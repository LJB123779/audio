# 音频合并与预览工具

一个基于 PyQt5 + pydub + pyqtgraph 的桌面工具，支持：
- 多个音频文件合并（可重复添加同一文件）
- 拖拽文件或文件夹到窗口添加
- 支持 MP3（需本地安装 FFmpeg）
- 在片段之间加入静音间隔
- 预览已合成的音频，带可拖动进度条
- 显示波形（类似声纹）与音量条

## 环境准备
1. 安装 Python 3.9+（Windows 推荐 64 位）
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 安装 FFmpeg（用于 MP3 读写）：
   - Windows: 下载 FFmpeg 压缩包并将 `bin` 目录加入系统 `PATH`。

## 运行
```bash
python main.py
```

## 提示
- 首次加载 MP3 时若失败，请确认 FFmpeg 可从命令行运行：
```bash
ffmpeg -version
```
- 拖拽文件夹将自动扫描其中的音频文件（mp3/wav/flac/...）。
