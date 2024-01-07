import torch
from diffusers import DiffusionPipeline
import config

def initialize_diffusion_model():
    pretrained_model_name_or_path = config.STABLE_DIFFUSION_MODEL
    pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
                                             torch_dtype=torch.float16).to(config.DEVICE)
    
    pipe.load_textual_inversion("sd-concepts-library/dreams")
    pipe.load_textual_inversion("sd-concepts-library/midjourney-style") 
    pipe.load_textual_inversion("sd-concepts-library/moebius") 
    pipe.load_textual_inversion("sd-concepts-library/style-of-marc-allante") 
    pipe.load_textual_inversion("sd-concepts-library/wlop-style")

    return pipe