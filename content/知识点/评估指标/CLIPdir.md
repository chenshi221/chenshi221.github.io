1. 计算**图像变化向量**：$V_{img} = CLIP_{feat}(It) - CLIP_{feat}(Is)$
2. 计算**文本变化向量**：$V_{text} = CLIP_{feat}(Tt) - CLIP_{feat}(Ts)$
3. 计算这两个变化向量之间的余弦相似度。
