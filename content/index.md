---
title: 首页
---

<section class="home-hero">
  <p class="home-kicker">Obsidian -> Quartz -> GitHub Pages</p>
  <h1>Blog1</h1>
  <p class="home-lead">
    一个从个人 Obsidian vault 中筛选发布的公开知识库，主要记录机器学习、图像编辑论文、编程语言和推荐算法笔记。
  </p>
  <div class="home-actions">
    <a href="./论文阅读/论文">开始阅读</a>
    <a href="./tags">浏览标签</a>
  </div>
</section>

## 公开范围

这里发布的是从 Obsidian vault `notes` 中筛选出来的公开笔记。`组会`、`AIMbench`、`AIM-bench`、`模板` 等目录会在同步和构建时被排除。

<section class="home-grid">
  <a class="home-card" href="./论文阅读/论文">
    <span class="home-card-label">Research</span>
    <strong>论文阅读</strong>
    <span>图像编辑、扩散模型、Agent 和情感图像编辑方向的论文笔记。</span>
  </a>
  <a class="home-card" href="./知识点/DiT">
    <span class="home-card-label">Concepts</span>
    <strong>知识点</strong>
    <span>DiT、adaLN、空间对齐、非空间对齐和常用评估指标。</span>
  </a>
  <a class="home-card" href="./机器学习/机器学习">
    <span class="home-card-label">ML</span>
    <strong>机器学习</strong>
    <span>机器学习、PyTorch、NumPy、Pandas 等基础内容。</span>
  </a>
  <a class="home-card" href="./编程语言/python">
    <span class="home-card-label">Code</span>
    <strong>编程语言</strong>
    <span>Python、Java、JavaScript、C 和 Makefile 学习笔记。</span>
  </a>
  <a class="home-card" href="./推荐算法/推荐算法阅读笔记">
    <span class="home-card-label">Recsys</span>
    <strong>推荐算法</strong>
    <span>推荐算法方向的阅读记录和知识整理。</span>
  </a>
</section>

## 更新流程

在 `notes` 中修改笔记后，到 `blog1` 目录同步、构建并发布：

```bash
pnpm sync:notes
pnpm quartz build
git add -A
git commit -m "Update notes"
git push
```

> [!note]
> `sync:notes` 会保留这个首页，并跳过不会发布的目录。正式发布由 GitHub Actions 自动完成。
