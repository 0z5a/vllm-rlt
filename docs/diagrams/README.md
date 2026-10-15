# Mermaid 图形

直接用浏览器打开 `index.html`，切换三张图并缩放。无需联网。

- `.mmd`：可编辑 Mermaid 源码；同步到上一级 `unified_execution_zh.md` 的 Mermaid 代码块。
- `.svg`：矢量导出，适合放大。
- `.png`：2 倍分辨率导出，适合预览与分享。
- `mermaid-config.json`：统一主题与布局配置。

安装有 Mermaid CLI 时，在本目录重新生成：

```bash
for name in stage-architecture unified-forward pd-handoff; do
  for ext in svg png; do
    mmdc -i "$name.mmd" -o "$name.$ext" \
      -c mermaid-config.json -b white -w 1600 -s 2
  done
done
```

本次使用 Mermaid 11.17.2、Chrome 153 导出，已检查中文与布局。
自定义浏览器路径可通过 Mermaid CLI 的 `-p` 指定 Puppeteer 配置。
