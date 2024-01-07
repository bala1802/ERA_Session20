import torch
from diffusers import DiffusionPipeline
import config

def initialize_diffusion_model():
    pretrained_model_name_or_path = config.STABLE_DIFFUSION_MODEL
    pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
                                             torch_dtype=torch.float16).to(config.DEVICE)
    return pipe