import torch
import gc

import utils
import model
import config
import image_generator as generator

def predict(prompt, pipe, loss_function=None):
    latents = []
    
    for seed_number, sd_concept in zip(config.SEEDS, config.STABLE_DIFUSION_CONCEPTS):
        # torch.mps.empty_cache()
        torch.gpu.empty_cache()
        gc.collect()
        # torch.mps.empty_cache()
        torch.gpu.empty_cache()

        prompt = [f'{prompt} {sd_concept}']
        latent = generator.generate_images(pipe=pipe, seed_number=seed_number, prompt=prompt, loss_function=loss_function)
        latents.append(latent)
    
    latents = torch.vstack(latents)
    images = utils.convert_latents_to_pil_images(pipe=pipe, latents=latents)
    grid = utils.populate_image_grid(images, 1, len(latents))
    return grid
