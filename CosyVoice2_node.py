import os
import sys
import time
import torch
import torchaudio
import numpy as np
from typing import Dict, Any, Literal, Union, Generator

# 添加当前目录到Python路径中，以便导入cosyvoice模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加third_party/Matcha-TTS到路径中，以便导入CosyVoice
sys.path.append(os.path.join(current_dir, 'third_party', 'Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 尝试导入ComfyUI相关模块
try:
    import comfy
    from comfy.model_management import get_torch_device
    import folder_paths
    COMFYUI_AVAILABLE = True
    
    # 获取ComfyUI的模型目录
    def get_comfyui_model_dir():
        """获取ComfyUI的模型目录"""
        try:
            # 使用folder_paths获取模型目录
            if hasattr(folder_paths, 'models_dir'):
                return folder_paths.models_dir
            
            # 从folder_paths获取base_path，然后构建models目录
            if hasattr(folder_paths, 'base_path'):
                return os.path.join(folder_paths.base_path, "models")
            
            # 尝试从ComfyUI的配置中获取模型目录
            if hasattr(comfy, 'model_management') and hasattr(comfy.model_management, 'models_directory'):
                return comfy.model_management.models_directory
            
            # 尝试从ComfyUI的路径推断模型目录
            try:
                import comfy_path
                comfy_root = os.path.dirname(os.path.dirname(comfy_path.__file__))
                models_dir = os.path.join(comfy_root, 'models')
                
                # 如果模型目录不存在，尝试其他可能的路径
                if not os.path.exists(models_dir):
                    # 尝试在ComfyUI根目录下查找models目录
                    for root, dirs, files in os.walk(comfy_root):
                        if 'models' in dirs:
                            models_dir = os.path.join(root, 'models')
                            break
                
                return models_dir
            except ImportError:
                pass
            
            # 尝试从当前工作目录推断ComfyUI模型目录
            current_dir = os.getcwd()
            if 'ComfyUI' in current_dir:
                # 尝试找到ComfyUI根目录
                comfyui_root = current_dir
                while 'ComfyUI' in comfyui_root and os.path.basename(comfyui_root) != 'ComfyUI':
                    comfyui_root = os.path.dirname(comfyui_root)
                
                if os.path.basename(comfyui_root) == 'ComfyUI':
                    models_dir = os.path.join(comfyui_root, 'models')
                    if os.path.exists(models_dir):
                        return models_dir
            
            # 尝试从环境变量获取ComfyUI模型目录
            if 'COMFYUI_MODEL_PATH' in os.environ:
                return os.environ['COMFYUI_MODEL_PATH']
            
            # 尝试常见的ComfyUI模型目录路径
            common_paths = [
                os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'models'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))), 'models'),
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    return path
            
            return None
        except Exception as e:
            print(f"Warning: Failed to get ComfyUI model directory: {e}")
            return None
    
    # 获取ComfyUI模型目录
    COMFYUI_MODEL_DIR = get_comfyui_model_dir()
    if COMFYUI_MODEL_DIR:
        print(f"ComfyUI model directory found: {COMFYUI_MODEL_DIR}")
    else:
        print("Warning: ComfyUI model directory not found, using default paths")
except ImportError:
    COMFYUI_AVAILABLE = False
    COMFYUI_MODEL_DIR = None

# 定义CosyVoice2模型类型
class CosyVoice2ModelType:
    pass

# 定义音频数据类型
class CosyVoice2Audio:
    def __init__(self, waveform: torch.Tensor, sample_rate: int):
        self.waveform = waveform
        self.sample_rate = sample_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waveform": self.waveform,
            "sample_rate": self.sample_rate
        }

class CosyVoice2Loader:
    """加载CosyVoice2模型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取模型文件夹列表
        model_folders = get_cosyvoice_model_folders()
        
        # 设置默认模型
        default_model = model_folders[0] if model_folders else "CosyVoice2-0.5B"
        
        return {
            "required": {
                "model_name": (model_folders, {"default": default_model}),
            },
            "optional": {
                "load_jit": ("BOOLEAN", {"default": False}),
                "load_trt": ("BOOLEAN", {"default": False}),
                "load_vllm": ("BOOLEAN", {"default": False}),
                "fp16": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("COSYVOICE2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "CosyVoice2/Loaders"
    
    def load_model(self, model_name: str, load_jit: bool = False, load_trt: bool = False, 
                  load_vllm: bool = False, fp16: bool = False, device: str = "auto",
                  auto_download: bool = True):
        try:
            # 构建完整的模型目录路径
            if COMFYUI_MODEL_DIR:
                model_dir = os.path.join(COMFYUI_MODEL_DIR, "CosyVoice", model_name)
            else:
                model_dir = os.path.join("pretrained_models", model_name)
            
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                if auto_download:
                    # 尝试从modelscope下载
                    try:
                        from modelscope import snapshot_download
                        print(f"Model not found at {model_dir}. Attempting to download automatically...")
                        
                        # 根据model_name确定模型ID
                        model_id_map = {
                            "CosyVoice2-0.5B": "iic/CosyVoice2-0.5B",
                            "CosyVoice-300M": "iic/CosyVoice-300M",
                            "CosyVoice-300M-SFT": "iic/CosyVoice-300M-SFT",
                            "CosyVoice-300M-Instruct": "iic/CosyVoice-300M-Instruct",
                            "CosyVoice-ttsfrd": "iic/CosyVoice-ttsfrd",
                        }
                        
                        model_id = model_id_map.get(model_name)
                        if not model_id:
                            # 尝试直接使用model_name作为model_id
                            model_id = f"iic/{model_name}"
                        
                        # 下载模型
                        print(f"Downloading model {model_id}...")
                        model_dir = snapshot_download(model_id, local_dir=model_dir)
                        print(f"Model downloaded to {model_dir}")
                    except ImportError:
                        raise ImportError("modelscope not installed. Please install it with 'pip install modelscope' or provide a valid model directory.")
                    except Exception as e:
                        raise ValueError(f"Failed to download model {model_name}: {str(e)}")
                else:
                    raise FileNotFoundError(f"Model directory not found: {model_dir}. Enable auto_download to download automatically.")
            
            # 确定设备
            if device == "auto":
                if COMFYUI_AVAILABLE:
                    device = get_torch_device()
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 加载CosyVoice2模型
            print(f"Loading CosyVoice2 model from {model_dir}...")
            model = CosyVoice2(model_dir, load_jit=load_jit, load_trt=load_trt, 
                              load_vllm=load_vllm, fp16=fp16)
            
            # 将模型移动到指定设备
            if hasattr(model, 'llm') and model.llm is not None:
                model.llm = model.llm.to(device)
            if hasattr(model, 'flow') and model.flow is not None:
                model.flow = model.flow.to(device)
            if hasattr(model, 'hift') and model.hift is not None:
                model.hift = model.hift.to(device)
            
            # 保存设备信息
            model.device = device
            print(f"CosyVoice2 model loaded successfully on {device}")
            
            return (model,)
        except Exception as e:
            raise RuntimeError(f"Failed to load CosyVoice2 model: {str(e)}")

class CosyVoice2ZeroShot:
    """CosyVoice2零样本语音合成"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取说话人文件列表
        speaker_files = get_speaker_files()
        # 添加"无"选项作为默认选项
        speaker_options = ["无"] + speaker_files if speaker_files else ["无"]
        
        return {
            "required": {
                "model": ("COSYVOICE2_MODEL",),
                "tts_text": ("STRING", {"default": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "multiline": True}),
            },
            "optional": {
                "prompt_text": ("STRING", {"default": ""}),
                "prompt_audio": ("AUDIO",),
                "speaker_file": (speaker_options, {"default": "无"}),
                # "stream": ("BOOLEAN", {"default": False}),  # 隐藏流式输出选项
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "text_frontend": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "zero_shot"
    CATEGORY = "CosyVoice2/Inference"
    
    def zero_shot(self, model: CosyVoice2, tts_text: str, prompt_text: str = "", 
                 prompt_audio: Dict[str, Any] = None, speaker_file: str = "",
                 # stream: bool = False,  # 隐藏流式输出选项
                 speed: float = 1.0, text_frontend: bool = True, seed: int = 0):
        try:
            # 初始化变量
            prompt_speech_16k = None
            zero_shot_spk_id = ""
            
            # 处理说话人文件选项，将"无"转换为空字符串
            actual_speaker_file = "" if speaker_file == "无" else speaker_file
            
            # 如果提供了说话人文件，则加载说话人信息
            if actual_speaker_file and actual_speaker_file != "":
                # 获取说话人ID（去掉.pt扩展名）
                zero_shot_spk_id = os.path.splitext(actual_speaker_file)[0]
                
                # 确保说话人信息已加载到模型中
                speakers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
                speaker_file_path = os.path.join(speakers_dir, actual_speaker_file)
                
                if os.path.exists(speaker_file_path):
                    # 加载说话人信息
                    speaker_data = torch.load(speaker_file_path, map_location=model.device)
                    # 将说话人信息添加到模型中
                    model.frontend.spk2info[zero_shot_spk_id] = speaker_data
                    print(f"Loaded speaker info from {speaker_file_path}")
                else:
                    raise FileNotFoundError(f"Speaker file not found: {speaker_file_path}")
            
            # 如果提供了音频，则使用音频和文本（优先级高于保存的说话人）
            if prompt_audio is not None:
                # 获取音频和采样率
                prompt_speech_16k = prompt_audio["waveform"]
                sample_rate = prompt_audio["sample_rate"]
                
                # 确保音频采样率为16kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    prompt_speech_16k = resampler(prompt_speech_16k)
                
                # 确保音频形状为[T]格式（CosyVoice期望的格式）
                while prompt_speech_16k.dim() > 1:
                    # 持续压缩直到只剩一维
                    prompt_speech_16k = prompt_speech_16k[0]
                
                # 如果使用了音频，则不使用预保存的说话人ID
                zero_shot_spk_id = ""
            
            # 如果使用了保存的说话人且没有提供音频，需要特殊处理
            if zero_shot_spk_id and prompt_audio is None:
                # 使用保存的说话人信息进行推理，不传递prompt_speech_16k参数
                audio_generator = model.inference_zero_shot(
                    tts_text=tts_text,
                    prompt_text=prompt_text,
                    prompt_speech_16k=None,  # 使用保存的说话人时不传递音频
                    zero_shot_spk_id=zero_shot_spk_id,
                    # stream=stream,  # 隐藏流式输出选项
                    speed=speed,
                    text_frontend=text_frontend,
                    seed=seed
                )
            else:
                # 执行零样本推理
                audio_generator = model.inference_zero_shot(
                    tts_text=tts_text,
                    prompt_text=prompt_text,
                    prompt_speech_16k=prompt_speech_16k,  # 当zero_shot_spk_id不为空时，CosyVoice会忽略这个参数
                    zero_shot_spk_id=zero_shot_spk_id,
                    # stream=stream,  # 隐藏流式输出选项
                    speed=speed,
                    text_frontend=text_frontend,
                    seed=seed
                )
            
            # 收集所有音频片段
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk["tts_speech"])
            
            # 合并所有音频片段
            if audio_chunks:
                combined_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")
            
            # 确保音频形状为[B, C, T]格式
            if combined_audio.dim() == 1:  # [T]
                combined_audio = combined_audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            elif combined_audio.dim() == 2:  # [C, T]
                combined_audio = combined_audio.unsqueeze(0)  # [1, C, T]
            
            # 转换为ComfyUI的AUDIO格式
            audio_dict = {
                "waveform": combined_audio,
                "sample_rate": model.sample_rate
            }
            
            return (audio_dict,)
        except Exception as e:
            raise RuntimeError(f"Zero-shot inference failed: {str(e)}")

class CosyVoice2Instruct:
    """CosyVoice2指令语音合成"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取说话人文件列表
        speaker_files = get_speaker_files()
        # 添加"无"选项作为默认选项
        speaker_options = ["无"] + speaker_files if speaker_files else ["无"]
        
        return {
            "required": {
                "model": ("COSYVOICE2_MODEL",),
                "tts_text": ("STRING", {"default": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "multiline": True}),
                "instruct_text": ("STRING", {"default": "用四川话说这句话"}),
            },
            "optional": {
                "prompt_audio": ("AUDIO",),
                "speaker_file": (speaker_options, {"default": "无"}),
                # "stream": ("BOOLEAN", {"default": False}),  # 隐藏流式输出选项
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "text_frontend": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "instruct"
    CATEGORY = "CosyVoice2/Inference"
    
    def instruct(self, model: CosyVoice2, tts_text: str, instruct_text: str, 
                prompt_audio: Dict[str, Any] = None, speaker_file: str = "",
                # stream: bool = False,  # 隐藏流式输出选项
                speed: float = 1.0, text_frontend: bool = True, seed: int = 0):
        try:
            # 初始化变量
            prompt_speech_16k = None
            zero_shot_spk_id = ""
            
            # 处理说话人文件选项，将"无"转换为空字符串
            actual_speaker_file = "" if speaker_file == "无" else speaker_file
            
            # 如果提供了说话人文件，则加载说话人信息
            if actual_speaker_file and actual_speaker_file != "":
                # 获取说话人ID（去掉.pt扩展名）
                zero_shot_spk_id = os.path.splitext(actual_speaker_file)[0]
                
                # 确保说话人信息已加载到模型中
                speakers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
                speaker_file_path = os.path.join(speakers_dir, actual_speaker_file)
                
                if os.path.exists(speaker_file_path):
                    # 加载说话人信息
                    speaker_data = torch.load(speaker_file_path, map_location=model.device)
                    # 将说话人信息添加到模型中
                    model.frontend.spk2info[zero_shot_spk_id] = speaker_data
                    print(f"Loaded speaker info from {speaker_file_path}")
                else:
                    raise FileNotFoundError(f"Speaker file not found: {speaker_file_path}")
            
            # 如果提供了音频，则使用音频（优先级高于保存的说话人）
            if prompt_audio is not None:
                # 获取音频和采样率
                prompt_speech_16k = prompt_audio["waveform"]
                sample_rate = prompt_audio["sample_rate"]
                
                # 确保音频采样率为16kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    prompt_speech_16k = resampler(prompt_speech_16k)
                
                # 确保音频形状为[T]格式（CosyVoice期望的格式）
                while prompt_speech_16k.dim() > 1:
                    # 持续压缩直到只剩一维
                    prompt_speech_16k = prompt_speech_16k[0]
                
                # 如果使用了音频，则不使用预保存的说话人ID
                zero_shot_spk_id = ""
            
            # 如果使用了保存的说话人且没有提供音频，需要特殊处理指令文本
            if zero_shot_spk_id and prompt_audio is None:
                # 提取指令文本的token
                instruct_text_token, instruct_text_token_len = model.frontend._extract_text_token(instruct_text + '<|endofprompt|>')
                
                # 临时更新保存的说话人信息中的提示文本部分
                if zero_shot_spk_id in model.frontend.spk2info:
                    # 保存原始的提示文本
                    original_prompt_text = model.frontend.spk2info[zero_shot_spk_id].get('prompt_text', None)
                    original_prompt_text_len = model.frontend.spk2info[zero_shot_spk_id].get('prompt_text_len', None)
                    
                    # 更新提示文本部分
                    model.frontend.spk2info[zero_shot_spk_id]['prompt_text'] = instruct_text_token
                    model.frontend.spk2info[zero_shot_spk_id]['prompt_text_len'] = instruct_text_token_len
                    
                    try:
                        # 执行指令推理
                        audio_generator = model.inference_instruct2(
                            tts_text=tts_text,
                            instruct_text=instruct_text,
                            prompt_speech_16k=None,
                            zero_shot_spk_id=zero_shot_spk_id,
                            # stream=stream,  # 隐藏流式输出选项
                            speed=speed,
                            text_frontend=text_frontend,
                            seed=seed
                        )
                        
                        # 收集所有音频片段
                        audio_chunks = []
                        for chunk in audio_generator:
                            audio_chunks.append(chunk["tts_speech"])
                    finally:
                        # 恢复原始的提示文本
                        if original_prompt_text is not None:
                            model.frontend.spk2info[zero_shot_spk_id]['prompt_text'] = original_prompt_text
                            model.frontend.spk2info[zero_shot_spk_id]['prompt_text_len'] = original_prompt_text_len
                        else:
                            # 如果原来没有提示文本，则删除
                            if 'prompt_text' in model.frontend.spk2info[zero_shot_spk_id]:
                                del model.frontend.spk2info[zero_shot_spk_id]['prompt_text']
                            if 'prompt_text_len' in model.frontend.spk2info[zero_shot_spk_id]:
                                del model.frontend.spk2info[zero_shot_spk_id]['prompt_text_len']
                else:
                    raise RuntimeError(f"Speaker {zero_shot_spk_id} not found in spk2info")
            else:
                # 执行指令推理
                audio_generator = model.inference_instruct2(
                    tts_text=tts_text,
                    instruct_text=instruct_text,
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id=zero_shot_spk_id,
                    # stream=stream,  # 隐藏流式输出选项
                    speed=speed,
                    text_frontend=text_frontend,
                    seed=seed
                )
                
                # 收集所有音频片段
                audio_chunks = []
                for chunk in audio_generator:
                    audio_chunks.append(chunk["tts_speech"])
            
            # 合并所有音频片段
            if audio_chunks:
                combined_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")
            
            # 确保音频形状为[B, C, T]格式
            if combined_audio.dim() == 1:  # [T]
                combined_audio = combined_audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            elif combined_audio.dim() == 2:  # [C, T]
                combined_audio = combined_audio.unsqueeze(0)  # [1, C, T]
            
            # 转换为ComfyUI的AUDIO格式
            audio_dict = {
                "waveform": combined_audio,
                "sample_rate": model.sample_rate
            }
            
            return (audio_dict,)
        except Exception as e:
            raise RuntimeError(f"Instruct inference failed: {str(e)}")

class CosyVoice2CrossLingual:
    """CosyVoice2跨语言语音合成"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取说话人文件列表
        speaker_files = get_speaker_files()
        # 添加"无"选项作为默认选项
        speaker_options = ["无"] + speaker_files if speaker_files else ["无"]
        
        return {
            "required": {
                "model": ("COSYVOICE2_MODEL",),
                "tts_text": ("STRING", {"default": "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。", "multiline": True}),
            },
            "optional": {
                "prompt_audio": ("AUDIO",),
                "speaker_file": (speaker_options, {"default": "无"}),
                # "stream": ("BOOLEAN", {"default": False}),  # 隐藏流式输出选项
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "text_frontend": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "cross_lingual"
    CATEGORY = "CosyVoice2/Inference"
    
    def cross_lingual(self, model: CosyVoice2, tts_text: str, 
                     prompt_audio: Dict[str, Any] = None, speaker_file: str = "",
                     # stream: bool = False,  # 隐藏流式输出选项
                     speed: float = 1.0, 
                     text_frontend: bool = True, seed: int = 0):
        try:
            # 初始化变量
            prompt_speech_16k = None
            zero_shot_spk_id = ""
            
            # 处理说话人文件选项，将"无"转换为空字符串
            actual_speaker_file = "" if speaker_file == "无" else speaker_file
            
            # 如果提供了说话人文件，则加载说话人信息
            if actual_speaker_file and actual_speaker_file != "":
                # 获取说话人ID（去掉.pt扩展名）
                zero_shot_spk_id = os.path.splitext(actual_speaker_file)[0]
                
                # 确保说话人信息已加载到模型中
                speakers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
                speaker_file_path = os.path.join(speakers_dir, actual_speaker_file)
                
                if os.path.exists(speaker_file_path):
                    # 加载说话人信息
                    speaker_data = torch.load(speaker_file_path, map_location=model.device)
                    # 将说话人信息添加到模型中
                    model.frontend.spk2info[zero_shot_spk_id] = speaker_data
                    print(f"Loaded speaker info from {speaker_file_path}")
                else:
                    raise FileNotFoundError(f"Speaker file not found: {speaker_file_path}")
            
            # 如果提供了音频，则使用音频（优先级高于保存的说话人）
            if prompt_audio is not None:
                # 获取音频和采样率
                prompt_speech_16k = prompt_audio["waveform"]
                sample_rate = prompt_audio["sample_rate"]
                
                # 确保音频采样率为16kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    prompt_speech_16k = resampler(prompt_speech_16k)
                
                # 确保音频形状为[T]格式（CosyVoice期望的格式）
                while prompt_speech_16k.dim() > 1:
                    # 持续压缩直到只剩一维
                    prompt_speech_16k = prompt_speech_16k[0]
                
                # 如果使用了音频，则不使用预保存的说话人ID
                zero_shot_spk_id = ""
            
            # 如果使用了保存的说话人且没有提供音频，需要特殊处理
            if zero_shot_spk_id and prompt_audio is None:
                # 使用保存的说话人信息进行推理，不传递prompt_speech_16k参数
                audio_generator = model.inference_cross_lingual(
                    tts_text=tts_text,
                    prompt_speech_16k=None,  # 使用保存的说话人时不传递音频
                    zero_shot_spk_id=zero_shot_spk_id,
                    # stream=stream,  # 隐藏流式输出选项
                    speed=speed,
                    text_frontend=text_frontend,
                    seed=seed
                )
            else:
                # 执行跨语言推理
                audio_generator = model.inference_cross_lingual(
                    tts_text=tts_text,
                    prompt_speech_16k=prompt_speech_16k,  # 当zero_shot_spk_id不为空时，CosyVoice会忽略这个参数
                    zero_shot_spk_id=zero_shot_spk_id,
                    # stream=stream,  # 隐藏流式输出选项
                    speed=speed,
                    text_frontend=text_frontend,
                    seed=seed
                )
            
            # 收集所有音频片段
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk["tts_speech"])
            
            # 合并所有音频片段
            if audio_chunks:
                combined_audio = torch.cat(audio_chunks, dim=1)
            else:
                raise RuntimeError("No audio generated")
            
            # 确保音频形状为[B, C, T]格式
            if combined_audio.dim() == 1:  # [T]
                combined_audio = combined_audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            elif combined_audio.dim() == 2:  # [C, T]
                combined_audio = combined_audio.unsqueeze(0)  # [1, C, T]
            
            # 转换为ComfyUI的AUDIO格式
            audio_dict = {
                "waveform": combined_audio,
                "sample_rate": model.sample_rate
            }
            
            return (audio_dict,)
        except Exception as e:
            raise RuntimeError(f"Cross-lingual inference failed: {str(e)}")

class CosyVoice2ModelChecker:
    """检查CosyVoice2模型是否已下载"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取模型文件夹列表
        model_folders = get_cosyvoice_model_folders()
        
        # 设置默认模型
        default_model = model_folders[0] if model_folders else "CosyVoice2-0.5B"
        
        return {
            "required": {
                "model_name": (model_folders, {"default": default_model}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("model_exists", "model_path",)
    FUNCTION = "check_model"
    CATEGORY = "CosyVoice2/Utils"
    
    def check_model(self, model_name: str):
        try:
            # 构建完整的模型目录路径
            if COMFYUI_MODEL_DIR:
                model_dir = os.path.join(COMFYUI_MODEL_DIR, "CosyVoice", model_name)
            else:
                model_dir = os.path.join("pretrained_models", model_name)
            
            # 检查模型目录是否存在
            model_exists = os.path.exists(model_dir) and os.path.isdir(model_dir) and os.listdir(model_dir)
            
            if model_exists:
                return (True, model_dir)
            else:
                return (False, model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to check model: {str(e)}")

class CosyVoice2ModelDownloader:
    """下载CosyVoice2模型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 设置默认本地目录
        default_local_dir = "pretrained_models"
        
        # 如果ComfyUI模型目录可用，则使用它作为默认路径
        if COMFYUI_MODEL_DIR:
            # 创建CosyVoice模型子目录
            default_local_dir = os.path.join(COMFYUI_MODEL_DIR, "CosyVoice")
        
        return {
            "required": {
                "model_type": (["CosyVoice2-0.5B", "CosyVoice-300M", "CosyVoice-300M-SFT", "CosyVoice-300M-Instruct", "CosyVoice-ttsfrd"], {"default": "CosyVoice2-0.5B"}),
            },
            "optional": {
                "local_dir": ("STRING", {"default": default_local_dir}),
                "download_ttsfrd": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "download_model"
    CATEGORY = "CosyVoice2/Utils"
    OUTPUT_NODE = True
    
    def download_model(self, model_type: str, local_dir: str = "pretrained_models", 
                     download_ttsfrd: bool = False):
        try:
            # 如果ComfyUI模型目录可用且local_dir不是ComfyUI模型目录的子目录，则使用ComfyUI模型目录
            if COMFYUI_MODEL_DIR and not local_dir.startswith(COMFYUI_MODEL_DIR):
                # 确定模型名称
                model_name = os.path.basename(local_dir)
                if not model_name or model_name == "pretrained_models":
                    model_name = "CosyVoice"
                
                # 创建CosyVoice模型子目录
                cosyvoice_model_dir = os.path.join(COMFYUI_MODEL_DIR, model_name)
                os.makedirs(cosyvoice_model_dir, exist_ok=True)
                
                # 更新local_dir路径
                local_dir = cosyvoice_model_dir
                print(f"Using ComfyUI model directory: {local_dir}")
            
            # 确保目录存在
            os.makedirs(local_dir, exist_ok=True)
            
            # 尝试导入modelscope
            try:
                from modelscope import snapshot_download
            except ImportError:
                raise ImportError("modelscope not installed. Please install it with 'pip install modelscope'.")
            
            # 根据模型类型确定模型ID
            model_id_map = {
                "CosyVoice2-0.5B": "iic/CosyVoice2-0.5B",
                "CosyVoice-300M": "iic/CosyVoice-300M",
                "CosyVoice-300M-SFT": "iic/CosyVoice-300M-SFT",
                "CosyVoice-300M-Instruct": "iic/CosyVoice-300M-Instruct",
                "CosyVoice-ttsfrd": "iic/CosyVoice-ttsfrd",
            }
            
            model_id = model_id_map.get(model_type)
            if not model_id:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # 构建本地目录路径
            model_local_dir = os.path.join(local_dir, model_type)
            
            # 检查模型是否已存在
            if os.path.exists(model_local_dir) and os.listdir(model_local_dir):
                print(f"Model {model_type} already exists at {model_local_dir}")
                return (model_local_dir,)
            
            # 下载模型
            print(f"Downloading model {model_type}...")
            downloaded_path = snapshot_download(model_id, local_dir=model_local_dir)
            print(f"Model {model_type} downloaded to {downloaded_path}")
            
            # 如果需要下载ttsfrd且不是已经下载的模型
            if download_ttsfrd and model_type != "CosyVoice-ttsfrd":
                ttsfrd_id = "iic/CosyVoice-ttsfrd"
                ttsfrd_local_dir = os.path.join(local_dir, "CosyVoice-ttsfrd")
                
                if not os.path.exists(ttsfrd_local_dir) or not os.listdir(ttsfrd_local_dir):
                    print(f"Downloading ttsfrd resources...")
                    ttsfrd_path = snapshot_download(ttsfrd_id, local_dir=ttsfrd_local_dir)
                    print(f"ttsfrd resources downloaded to {ttsfrd_path}")
                else:
                    print(f"ttsfrd resources already exist at {ttsfrd_local_dir}")
            
            return (downloaded_path,)
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_type}: {str(e)}")

class CosyVoice2SaveSpeaker:
    """保存零样本说话人信息"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE2_MODEL",),
                "prompt_text": ("STRING", {"default": ""}),
                "prompt_audio": ("AUDIO",),
                "zero_shot_spk_id": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("speaker_file",)
    FUNCTION = "save_speaker"
    CATEGORY = "CosyVoice2/Utils"
    OUTPUT_NODE = True
    WEB_DIRECTORY = "./web"
    
    def save_speaker(self, model: CosyVoice2, prompt_text: str, prompt_audio: Dict[str, Any], 
                    zero_shot_spk_id: str, prompt=None, extra_pnginfo=None):
        try:
            # 检查说话人ID是否为空
            if not zero_shot_spk_id or zero_shot_spk_id.strip() == "":
                # 准备错误显示信息（使用中文）
                error_info = f"说话人保存失败！\n\n错误信息: 说话人ID不能为空\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                return {"ui": {"text": [error_info]}, "result": ("",)}
            
            # 获取音频和采样率
            prompt_speech_16k = prompt_audio["waveform"]
            sample_rate = prompt_audio["sample_rate"]
            
            # 确保音频采样率为16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                prompt_speech_16k = resampler(prompt_speech_16k)
            
            # 确保音频形状为[T]格式（CosyVoice期望的格式）
            if prompt_speech_16k.dim() == 3:  # [B, C, T]
                # 取第一个批次和第一个通道
                prompt_speech_16k = prompt_speech_16k[0, 0, :]
            elif prompt_speech_16k.dim() == 2:  # [B, T]或[C, T]
                # 假设是[B, T]，取第一个批次
                prompt_speech_16k = prompt_speech_16k[0, :]
            
            # 添加零样本说话人
            result = model.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)
            
            if not result:
                # 准备错误显示信息（使用中文）
                error_info = f"说话人保存失败！\n\n错误信息: Failed to add zero-shot speaker\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                return {"ui": {"text": [error_info]}, "result": ("",)}
            # 创建speakers目录（如果不存在）
            speakers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
            os.makedirs(speakers_dir, exist_ok=True)
            
            # 保存说话人信息到节点目录下的speakers文件夹，使用.pt格式
            speaker_file_path = os.path.join(speakers_dir, f"{zero_shot_spk_id}.pt")
            torch.save(model.frontend.spk2info[zero_shot_spk_id], speaker_file_path)
            print(f"Speaker info saved to {speaker_file_path}")
            
            # 准备显示信息（使用中文）
            save_info = f"说话人保存成功！\n\n说话人ID: {zero_shot_spk_id}\n文件路径: {speaker_file_path}\n保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # 返回保存的文件路径和UI信息
            return {"ui": {"text": [save_info]}, "result": (speaker_file_path,)}
        except Exception as e:
            # 准备错误显示信息（使用中文）
            error_info = f"说话人保存失败！\n\n错误信息: {str(e)}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            return {"ui": {"text": [error_info]}, "result": ("",)}

# 添加获取说话人文件的辅助函数
def get_speaker_files():
    """获取speakers目录下的所有.pt文件"""
    speakers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
    speaker_files = []
    if os.path.exists(speakers_dir):
        speaker_files = [f for f in os.listdir(speakers_dir) if f.endswith(".pt")]
    return speaker_files

# 添加获取CosyVoice模型文件夹的辅助函数
def get_cosyvoice_model_folders():
    """获取ComfyUI模型目录下CosyVoice文件夹中的所有模型文件夹"""
    model_folders = []
    if COMFYUI_MODEL_DIR:
        cosyvoice_model_dir = os.path.join(COMFYUI_MODEL_DIR, "CosyVoice")
        if os.path.exists(cosyvoice_model_dir):
            # 获取所有子目录（模型文件夹）
            model_folders = [f for f in os.listdir(cosyvoice_model_dir) 
                           if os.path.isdir(os.path.join(cosyvoice_model_dir, f))]
    
    # 如果没有找到模型文件夹，添加默认选项
    if not model_folders:
        model_folders = ["CosyVoice2-0.5B"]
    
    return model_folders

# 节点映射
NODE_CLASS_MAPPINGS = {
    "CosyVoice2Loader": CosyVoice2Loader,
    "CosyVoice2ModelChecker": CosyVoice2ModelChecker,
    "CosyVoice2ModelDownloader": CosyVoice2ModelDownloader,
    "CosyVoice2ZeroShot": CosyVoice2ZeroShot,
    "CosyVoice2Instruct": CosyVoice2Instruct,
    "CosyVoice2CrossLingual": CosyVoice2CrossLingual,
    "CosyVoice2SaveSpeaker": CosyVoice2SaveSpeaker,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoice2Loader": "Load CosyVoice2 Model",
    "CosyVoice2ModelChecker": "Check CosyVoice2 Model",
    "CosyVoice2ModelDownloader": "Download CosyVoice2 Model",
    "CosyVoice2ZeroShot": "CosyVoice2 Zero Shot",
    "CosyVoice2Instruct": "CosyVoice2 Instruct",
    "CosyVoice2CrossLingual": "CosyVoice2 Cross Lingual",
    "CosyVoice2SaveSpeaker": "Save Speaker to File",
}
