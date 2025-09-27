# ComfyUI-CosyVoice

ComfyUI-CosyVoice 是一个为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 设计的自定义节点集，集成了强大的 [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) 文本到语音合成模型，为您的 AI 工作流程提供高质量的语音生成能力。

## 功能特点

- **零样本声音克隆**：通过简短的参考音频，克隆任何人的声音
- **指令控制**：通过文本指令控制语音的风格、方言和情感
- **跨语言合成**：支持多种语言间的语音合成（中、英、日、韩等）
- **流式合成**：支持低延迟的流式语音生成
- **自动模型下载**：首次使用时自动下载所需模型
- **完整的音频处理**：包含音频加载、处理和保存功能

## 安装方法

### 1. 克隆仓库

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/FunAudioLLM/ComfyUI-CosyVoice.git
cd ComfyUI-CosyVoice
```

### 2. 安装依赖

确保您的 Python 环境已经安装了以下依赖：

```bash
pip install -r requirements.txt
```

如果您遇到 sox 兼容性问题：

```bash
# Ubuntu
apt-get install sox libsox-dev
# CentOS
yum install sox sox-devel
# Windows (通过chocolatey安装)
choco install sox
```

## 使用指南

启动 ComfyUI 后，您可以在节点菜单中找到 `CosyVoice2` 分类下的所有节点。

### 基本工作流程

1. **加载模型**：使用 `CosyVoice2Loader` 节点加载预训练模型
2. **准备参考音频**：使用 `LoadAudio` 节点加载用于声音克隆的参考音频
3. **选择合成方式**：选择 `CosyVoice2ZeroShot`、`CosyVoice2Instruct` 或 `CosyVoice2CrossLingual` 节点
4. **配置参数**：设置文本内容、语速等参数
5. **保存结果**：使用 `SaveAudio` 节点保存生成的音频

### 节点详解

#### CosyVoice2Loader

用于加载 CosyVoice2 模型的节点。

- **参数**：
  - `model_dir`：模型目录路径（默认：`pretrained_models/CosyVoice2-0.5B`）
  - `load_jit`：是否加载 JIT 编译模型（默认：`False`）
  - `load_trt`：是否加载 TensorRT 优化模型（默认：`False`）
  - `load_vllm`：是否加载 VLLM 加速模型（默认：`False`）
  - `fp16`：是否使用 FP16 精度（默认：`False`）
  - `device`：运行设备（`auto`/`cpu`/`cuda`，默认：`auto`）
  - `auto_download`：自动下载模型（默认：`True`）
- **输出**：
  - `model`：加载的 CosyVoice2 模型

#### LoadAudio

用于加载音频文件或处理音频数据的节点。

- **参数**：
  - `audio_path`：音频文件路径（默认：`./asset/zero_shot_prompt.wav`）
  - `audio_data`：可选的音频数据输入
  - `target_sample_rate`：目标采样率（8000-48000 Hz，默认：16000）
- **输出**：
  - `audio`：处理后的音频数据

#### SaveAudio

用于保存音频文件的节点。

- **参数**：
  - `audio`：要保存的音频数据
  - `save_path`：保存路径（默认：`./output.wav`）
- **输出**：
  - `path`：保存的文件路径

#### CosyVoice2ZeroShot

零样本语音合成节点，用于克隆声音。

- **参数**：
  - `model`：加载的 CosyVoice2 模型
  - `tts_text`：要合成的文本内容
  - `prompt_text`：与参考音频匹配的文本
  - `prompt_audio`：参考音频
  - `zero_shot_spk_id`：说话人ID（可选）
  - `stream`：是否启用流式合成（默认：`False`）
  - `speed`：语速（0.5-2.0，默认：1.0）
  - `text_frontend`：是否启用文本前端处理（默认：`True`）
- **输出**：
  - `audio`：合成的音频数据

#### CosyVoice2Instruct

指令语音合成节点，通过指令控制语音风格。

- **参数**：
  - `model`：加载的 CosyVoice2 模型
  - `tts_text`：要合成的文本内容
  - `instruct_text`：控制指令（如："用四川话说这句话"）
  - `prompt_audio`：参考音频
  - `zero_shot_spk_id`：说话人ID（可选）
  - `stream`：是否启用流式合成（默认：`False`）
  - `speed`：语速（0.5-2.0，默认：1.0）
  - `text_frontend`：是否启用文本前端处理（默认：`True`）
- **输出**：
  - `audio`：合成的音频数据

#### CosyVoice2CrossLingual

跨语言语音合成节点，支持多种语言间的语音合成。

- **参数**：
  - `model`：加载的 CosyVoice2 模型
  - `tts_text`：要合成的文本内容（支持多语言）
  - `prompt_audio`：参考音频
  - `zero_shot_spk_id`：说话人ID（可选）
  - `stream`：是否启用流式合成（默认：`False`）
  - `speed`：语速（0.5-2.0，默认：1.0）
  - `text_frontend`：是否启用文本前端处理（默认：`True`）
- **输出**：
  - `audio`：合成的音频数据

#### CosyVoice2ModelChecker

检查 CosyVoice2 模型是否已下载的节点。

- **参数**：
  - `model_dir`：模型目录路径
- **输出**：
  - `is_downloaded`：模型是否已下载

#### CosyVoice2ModelDownloader

下载 CosyVoice2 模型的节点。

- **参数**：
  - `model_type`：模型类型（`CosyVoice2-0.5B`/`CosyVoice-300M`/`CosyVoice-300M-SFT`/`CosyVoice-300M-Instruct`/`CosyVoice-ttsfrd`）
  - `local_dir`：本地保存目录
  - `download_ttsfrd`：是否下载 TTSFRD 资源（默认：`False`）
- **输出**：
  - `model_dir`：下载的模型目录

#### CosyVoice2SaveSpeaker

保存说话人信息的节点，用于零样本声音克隆。

- **参数**：
  - `model`：加载的 CosyVoice2 模型
  - `prompt_text`：与参考音频匹配的文本
  - `prompt_audio`：参考音频
  - `zero_shot_spk_id`：说话人ID
- **输出**：
  - `zero_shot_spk_id`：保存的说话人ID

## 模型管理

ComfyUI-CosyVoice 支持自动下载和管理模型。默认情况下，模型会保存在以下位置：

- 如果 ComfyUI 的模型目录可用，模型将保存在 `ComfyUI/models/CosyVoice/` 目录下
- 否则，模型将保存在当前目录的 `pretrained_models/` 目录下

## 高级功能

### 文本前端处理

启用 `text_frontend` 参数可以获得更好的文本规范化和处理效果，特别是对于包含数字、缩写和特殊字符的文本。

### 流式合成

启用 `stream` 参数可以降低延迟，特别适合需要实时响应的场景。

### 语速调整

通过 `speed` 参数可以调整合成语音的速度，范围为 0.5（半速）到 2.0（两倍速）。

## 常见问题

### 1. 首次使用时模型下载失败

如果自动下载失败，可以手动下载模型并放置到指定目录：

```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

### 2. 音频格式问题

CosyVoice2 要求输入音频为 16kHz 采样率的单声道音频。节点内部会自动处理不同格式的输入，但对于最佳效果，建议提前准备符合要求的音频文件。

### 3. 性能优化

- 对于 GPU 用户，可以启用 `fp16=True` 以获得更快的推理速度
- 对于高端 NVIDIA GPU 用户，可以尝试启用 `load_trt=True` 或 `load_vllm=True` 进一步提升性能

## 多语言支持

CosyVoice2 支持以下语言和方言：
- 中文
- 英文
- 日文
- 韩文
- 中文方言（粤语、四川话、上海话、天津话、武汉话等）

## 免责声明

本项目提供的内容仅用于学术目的，旨在展示技术能力。部分示例来源于互联网，如有任何内容侵犯了您的权益，请联系我们请求删除。

## 引用

如果您在研究中使用了本项目，请考虑引用以下论文：

```bibtex
@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}
```

## 致谢

感谢 [FunAudioLLM](https://github.com/FunAudioLLM) 团队开发的 CosyVoice 模型，以及所有为本项目做出贡献的开发者。