import os
from modelscope import snapshot_download

os.environ['MODELSCOPE_CACHE'] = "E:/cache/huggingface"
model_dir = snapshot_download("Qwen/Qwen2.5-7B-Instruct")
