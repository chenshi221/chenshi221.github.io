 Hugging Face Diffusers 库中 `train_instruct_pix2pix.py` 脚本实现了对InstructPix2Pix的训练。
 InstructPix2Pix 是一种能够根据文本指令编辑图像的扩散模型。其实现思路比较简单，修改标准扩散模型的 UNet的输入，使其能够接收一个额外的条件图像作为输入。

## 1. 模型组件加载

训练脚本首先会加载预训练好的 Stable Diffusion 模型的各个核心组件：

```python
# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
)
vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)
```

- **`noise_scheduler` (噪声调度器):** 在训练的“前向过程”中向图像中添加噪声，并在推理时指导“反向去噪过程”。这里加载的是 `DDPMScheduler`。[[扩散模型综述#1.1 去噪扩散概率模型(DDPM)]]
- **`tokenizer` (分词器):** 将输入的文本指令（如 "turn the horse into a zebra"）转换成模型能够理解的数字 ID。这里使用的是 `CLIPTokenizer`。
- **`text_encoder` (文本编码器):** 将分词器输出的数字 ID 转换成 embedding 向量表示。这里使用的是 `CLIPTextModel`。
- **`vae` (变分自编码器):** 用于将图像在像素空间和隐空间 (latent space) 之间进行转换。在隐空间上进行加噪和去噪比在像素空间上计算效率高得多。
- **`unet` (U-Net 模型):** 这是扩散模型的核心。它在每个时间步 (timestep) 预测噪声，然后从带噪的隐向量中减去预测的噪声，从而逐步恢复出清晰的图像。

## 2. UNet 修改：实现图像条件输入

InstructPix2Pix 的核心创新在于它能同时处理**带噪的目标图像**和**原始的条件图像**。为了实现这一点，脚本对标准的 UNet 进行了修改，将其输入通道从 4 个扩展到了 8 个。

![[Stable Diffusion结构#^f9c4dd]]

标准的 Stable Diffusion UNet 输入是 4 通道的隐向量（`latent`）。而 InstructPix2Pix 将**原始图像的隐向量**（4通道）与**带噪目标图像的隐向量**（4通道）在通道维度上进行拼接（concatenate），形成一个 8 通道的输入。

```python
# InstructPix2Pix uses an additional image for conditioning. To accommodate that,
# it uses 8 channels (instead of 4) in the first (conv) layer of the UNet.
logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
in_channels = 8
out_channels = unet.conv_in.out_channels
unet.register_to_config(in_channels=in_channels)

with torch.no_grad():
    # 创建一个新的输入卷积层，输入通道为 8
    new_conv_in = nn.Conv2d(
        in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    )
    # 将新卷积层的权重全部初始化为 0
    new_conv_in.weight.zero_()
    # 将预训练的原始权重（4个通道）复制到新卷积层的前4个通道
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    # 替换掉 U-Net 原本的输入卷积层
    unet.conv_in = new_conv_in
```

**权重初始化策略：**
- **前 4 个通道:** 复制预训练 UNet 的原始权重。这部分对应带噪的目标图像输入，保留了模型原有的生成能力。
- **后 4 个通道:** 权重初始化为零。这部分对应原始图像的条件输入。在训练开始时，模型对条件图像的处理是“中性”的，它将通过微调来学习如何利用这额外的 4 个通道信息来指导编辑过程。

在训练过程中，只有 `unet` 的权重会被更新，而 `vae` 和 `text_encoder` 会被冻结，以保留它们强大的特征提取能力。

```python
# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
```

## 3. 数据预处理

数据预处理的目标是为模型准备好成对的`(原始图像, 编辑指令, 编辑后图像)`数据。

```python
def preprocess_images(examples):
    original_images = np.concatenate(
        [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
    )
    edited_images = np.concatenate(
        [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
    )
    # We need to ensure that the original and the edited images undergo the same
    # augmentation transforms.
    images = np.stack([original_images, edited_images])
    images = torch.tensor(images)
    images = 2 * (images / 255) - 1
    return train_transforms(images)

def preprocess_train(examples):
    # Preprocess images.
    preprocessed_images = preprocess_images(examples)
    # Separate the original and edited images.
    original_images, edited_images = preprocessed_images
    original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
    edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

    examples["original_pixel_values"] = original_images
    examples["edited_pixel_values"] = edited_images

    # Preprocess the captions.
    captions = list(examples[edit_prompt_column])
    examples["input_ids"] = tokenize_captions(captions)
    return examples

# Set the training transforms
train_dataset = dataset["train"].with_transform(preprocess_train)
```

**关键步骤：**
1.  **统一变换:** 将原始图像和编辑后图像堆叠（`np.stack`）在一起，确保它们经历完全相同的数据增强（`train_transforms`），如随机裁剪、翻转等。这对于模型学习两者间的差异至关重要。
2.  **归一化:** 将图像像素值从 `[0, 255]` 范围归一化到 `[-1, 1]` 范围，这是扩散模型训练的标准实践。
3.  **文本分词:** 将编辑指令文本通过 `tokenizer` 转换为 `input_ids`。
