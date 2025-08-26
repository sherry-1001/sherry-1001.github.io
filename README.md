# Sherry Jiang's blog

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Local Development

```bash
git clone https://github.com/sherry-1001/sherry-1001.github.io.git
cd sherry-1001.github.io
npm install

npm run start # This command starts a local development server and opens up a browser window.
# Most changes are reflected live without having to restart the server.
```

Open `http://localhost:3000/` in your browser.

## 注意事项

### 博客

在 `blog` 目录下写博客。

每次新建一个目录，名为 `年-月-日-标题`，例如`2025-07-30-pytorch-dynamo`。然后在里面新建文件 `index.md`，用 Markdown 格式写博客。

图片和 `index.md` 文件放在同一个目录里，用相对路径来引用，例如 `![架构图](./architecture.png)`。

### 笔记 (TODO)

在 `docs` 目录下写笔记。