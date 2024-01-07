import torch
from tqdm.auto import tqdm

import config
import utils

def construct_text_embeddings(pipe, prompt):
    text_input = pipe.tokenizer(prompt, padding='max_length', 
                                max_length = pipe.tokenizer.model_max_length, truncation= True, 
                                return_tensors="pt")
    uncond_input = pipe.tokenizer([""] * config.BATCH_SIZE, padding="max_length", 
                                  max_length= text_input.input_ids.shape[-1], 
                                  return_tensors="pt")
    with torch.no_grad:
        text_input_embeddings = pipe.text_encoder(text_input.input_ids.to(config.DEVICE))[0]
    with torch.no_grad:
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(config.DEVICE))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_input_embeddings])
    return text_embeddings

def initialize_latents(seed_number, pipe, scheduler):
    generator = torch.seed(seed_number)
    latents = torch.randn((config.BATCH_SIZE, pipe.unet.config.in_channels, 
                           config.HEIGHT//8, config.width//8), 
                           generator = generator).to(torch.float16)


    latents = latents.to(config.DEVICE)
    latents = latents * scheduler.init_noise_sigma
    return latents

def run_prediction(pipe, text_embeddings, scheduler, latents, loss=None):
    for i, t in tqdm(enumerate(scheduler.timesteps), total = len(scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input.to(torch.float16), t, encoder_hidden_states=text_embeddings)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        if loss and i%5 == 0:
            latents = latents.detach().requires_grad_()
            #latents_x0 = scheduler.step(noise_pred,t, latents).pred_original_sample # this line does not work
            latents_x0 = latents - sigma * noise_pred

            denoised_images = pipe.vae.decode((1/ 0.18215) * latents_x0).sample / 2 + 0.5 # range(0,1)

            loss = utils.image_loss(denoised_images,loss) * config.LOSS_SCALE
            print(f"loss {loss}")

            cond_grad = torch.autograd.grad(loss, latents)[0]
            latents = latents.detach() - cond_grad * sigma**2
        
        latents = scheduler.step(noise_pred,t, latents).prev_sample

    return latents

def generate_images(pipe, seed_number, prompt, scheduler, loss=None):
    text_embeddings = construct_text_embeddings(pipe=pipe, prompt=prompt)
    latents = initialize_latents(seed_number=seed_number, pipe=pipe, scheduler=scheduler)
    latents = run_prediction(pipe, text_embeddings, scheduler, latents, loss)

    return latents