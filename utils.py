import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler

import config

def convert_latents_to_pil_images(pipe):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def populate_image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def initialize_lms_discrete_scheduler():
    scheduler = LMSDiscreteScheduler(beta_start = config.BETA_START, 
                                beta_end = config.BETA_END, 
                                beta_schedule = config.BETA_SCHEDULE, 
                                num_train_timesteps = config.NUM_TRAIN_TIMESTEPS)
    scheduler.set_timesteps(config.NUM_INFERENCE_STEPS)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)
    return scheduler