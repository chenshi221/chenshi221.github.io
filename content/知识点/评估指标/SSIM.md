#### SSIM (Structural Similarity Index) `↑`

*   **用途**: 衡量两张图像的**结构相似性**。
*   **计算步骤**: SSIM 从三个方面比较图像 `x` 和 `y`：
    1.  **亮度 (Luminance)**: 比较两张图像的平均像素值 `μ_x`, `μ_y`。
    2.  **对比度 (Contrast)**: 比较两张图像的像素值标准差 `σ_x`, `σ_y`。
    3.  **结构 (Structure)**: 比较两张图像的协方差 `σ_xy`。
    4.  **SSIM** = $l(x, y)^\alpha \cdot c(x, y)^\beta \cdot s(x, y)^\gamma$ (通常 `α,β,γ` 设为1)。
*   **意义**: SSIM 的取值范围是 `[-1, 1]`，越接近 1，表示两张图像在人眼看来结构越相似。它比 MSE 和 PSNR 更符合人类的视觉感知。