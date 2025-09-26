# CosyVoice2节点模型目录修改总结

## 修改目的
确保CosyVoice2节点能够正确使用ComfyUI的模型目录，而不是默认的.cache文件夹。

## 修改内容

### 1. 添加folder_paths模块导入
在ComfyUI相关模块导入部分添加了`import folder_paths`语句，以便使用ComfyUI的模型目录管理功能。

### 2. 增强get_comfyui_model_dir函数
在get_comfyui_model_dir函数开头新增了两种基于folder_paths的模型目录获取方式：
- 优先使用`folder_paths.models_dir`
- 其次使用`folder_paths.base_path`构建models目录路径

### 3. 修改CosyVoice2Loader类
- 在INPUT_TYPES方法中，使用ComfyUI模型目录作为默认路径
- 添加了`os.makedirs(cosyvoice_model_dir, exist_ok=True)`语句，确保CosyVoice子目录存在
- 在load_model方法中，确保模型目录在ComfyUI模型目录下

### 4. 修改CosyVoice2ModelChecker类
- 在INPUT_TYPES方法中，使用ComfyUI模型目录作为默认路径

### 5. 修改CosyVoice2ModelDownloader类
- 在INPUT_TYPES方法中，使用ComfyUI模型目录作为默认路径
- 在download_model方法中，确保下载目录在ComfyUI模型目录下

## 效果
现在，当用户使用CosyVoice2节点时：
1. 模型将自动下载到ComfyUI模型目录下的CosyVoice子目录中
2. 节点将优先使用ComfyUI的模型目录管理功能
3. 如果ComfyUI模型目录不可用，将回退到其他路径获取方法

## 文件路径
d:\AI\ComfyUI_windows_TTS\ComfyUI\custom_nodes\ComfyUI-CosyVoice\CosyVoice2_node.py