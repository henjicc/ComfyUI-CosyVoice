# ComfyUI-CosyVoice

[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) 的 ComfyUI 节点。除原项目代码外，其余均为 AI 编写，因此可能存在未知错误，使用时请留意。
## 功能特点

- **零样本声音克隆**：通过简短的参考音频，克隆任何人的声音
- **指令控制**：通过文本指令控制语音的风格、方言和情感
- **跨语言合成**：支持多种语言间的语音合成（中、英、日、韩等）
- **自动模型下载**：首次使用时自动下载所需模型

## 安装方法


```bash
cd custom_nodes
git clone https://github.com/henjicc/ComfyUI-CosyVoice.git
cd ComfyUI-CosyVoice
pip install -r requirements.txt
```

## 使用指南

启动 ComfyUI 后，可以在`CosyVoice2`分类下找到所有节点。
示例工作流：

- [零样本克隆](example_workflows/CosyVoice2_零样本克隆.json)
- [保存音色_使用音色生成](example_workflows/CosyVoice2_保存音色_使用音色生成.json)
- [跨语言合成](example_workflows/CosyVoice2_跨语言合成.json)
- [指令合成](example_workflows/CosyVoice2_指令合成.json)

## 多语言支持

CosyVoice2 支持以下语言和方言：
- 中文
- 英文
- 日文
- 韩文
- 中文方言（粤语、四川话、上海话、天津话、武汉话等）

## 免责声明

本项目提供的内容仅用于学术研究和技术展示目的，旨在推动语音合成技术的发展。请确保在使用本项目生成的语音内容时，遵守相关法律法规，不得用于任何违法违规用途。所有音频的生成和使用应获得相关权利人的明确授权，避免侵犯他人肖像权、声音权等合法权益。开发者不对因不当使用本项目而导致的任何法律后果承担责任。