# Vox Timeline

Vox Timeline 是一个基于 Qwen3-TTS 的智能视频配音系统。它支持“脚本驱动”模式，允许用户通过简单的 JSON 脚本定义台词、语气和时间点，系统将自动生成高质量的语音，并智能处理音频重叠问题。

## 核心功能

*   **脚本驱动配音**: 通过 JSON 文件精确控制每句台词的内容、开始时间、说话人和语气。
*   **高质量 TTS**: 集成 Qwen3-TTS 模型，支持多语种和情感控制。
*   **全流程视频翻译**:
    *   **ASR**: 使用 Whisper 模型进行语音识别。
    *   **翻译**: 支持 Google TranslateGemma-4B 和 Helsinki-NLP Opus-MT 多种翻译模型。
    *   **配音**: 自动将翻译后的文本转换为语音。
    *   **视频合成**: 支持生成带硬字幕或软字幕的最终视频，自动对齐音画。
*   **智能时间轴**:
    *   **自动防重叠 (Auto-Shift)**: 当生成的语音长度超过设定的时间间隔时，后续音频会自动顺延，防止重叠。
    *   **高保真合成**: 使用 `Pydub` 进行音频处理，确保音质清晰，无削波或失真。
*   **调试支持**: 支持导出每个独立的语音片段以便排查问题。

## 快速开始

### 1. 安装依赖

本项目使用 `uv` 进行依赖管理。

**方式 A: 使用 uv (推荐)**
确保已安装 `uv`，然后同步环境：

```bash
uv sync
```

**方式 B: 使用 pip**
如果不使用 uv，也可以通过 requirements.txt 安装：

```bash
pip install -r requirements.txt
```

### 2. 准备脚本

创建一个 JSON 脚本文件（例如 `script.json`）：

```json
[
  {
    "start": 0.0,
    "text": "你好，这是第一句话。",
    "speaker": "Uncle_Fu",
    "instruct": "开心"
  },
  {
    "start": 3.0,
    "text": "这是第二句话，如果第一句太长，我会自动往后移。",
    "speaker": "Uncle_Fu",
    "instruct": "认真"
  }
]
```

### 3. 运行生成

**仅生成音频**:
```bash
python main.py --script script.json --output my_audio.wav
```

**生成音频并合成到视频**:
```bash
python main.py --script script.json --video input.mp4 --video-out output.mp4
```

**调试模式 (保存独立片段)**:
```bash
python main.py --script script.json --output debug.wav --debug-dir ./debug_clips
```

### 4. Web UI 可视化界面

项目提供了一个基于 Web 的图形界面，包含两个主要功能模块：

1.  **Script Dubbing (脚本配音)**: 输入 JSON 脚本，生成配音。
2.  **Video Translation (视频翻译配音)**: 上传视频，一键完成“识别 -> 翻译 -> 配音 -> 字幕 -> 合成”。
    *   支持选择翻译模型（Google Gemma 或 Helsinki-NLP）。
    *   支持选择字幕类型（硬字幕或软字幕）。

启动 Web UI:
```bash
python app.py
```

启动后，在浏览器访问 `http://127.0.0.1:7860` 即可使用。

## 项目结构

*   `src/core/tts.py`: TTS 引擎封装，负责单句生成与数据类型转换。
*   `src/core/audio.py`: 音频时间轴管理，实现了基于 Pydub 的合成与自动移位逻辑。
*   `src/pipeline/dubbing.py`: 业务流程控制器，串联脚本读取、生成与导出。
*   `main.py`: 命令行入口。

## 注意事项

1.  **显存占用**: Qwen3-TTS 模型（0.6B）需要约 2-4GB 显存。请确保 CUDA 环境正常。
2.  **Flash Attention**: 如果未安装 `flash-attn`，模型会回退到普通 PyTorch 实现，推理速度会稍慢，但不影响音质。
3.  **音频格式**:
    *   TTS 输出为 24000Hz 采样率。
    *   系统内部统一转换为 float32 格式进行处理，最终导出为标准 WAV (int16/float32)。
4.  **自动移位**: 脚本中的 `start` 时间仅代表“最早开始时间”。如果前一段语音还没结束，当前段落会被自动推迟。

## 许可证

MIT
