import torch
from tqdm.auto import tqdm
from diffusers import LMSDiscreteScheduler

import config

def construct_text_embeddings(pipe, prompt):
    text_input = pipe.tokenizer(prompt, padding='max_length', 
                                max_length = pipe.tokenizer.model_max_length, truncation= True, 
                                return_tensors="pt")
    uncond_input = pipe.tokenizer([""] * config.BATCH_SIZE, padding="max_length", 
                                  max_length= text_input.input_ids.shape[-1], 
                                  return_tensors="pt")
    with torch.no_grad():
        text_input_embeddings = pipe.text_encoder(text_input.input_ids.to(config.DEVICE))[0]
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(config.DEVICE))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_input_embeddings])
    return text_embeddings

def initialize_latent(seed_number, pipe, scheduler):
    generator = torch.manual_seed(seed_number)
    latent = torch.randn((config.BATCH_SIZE, pipe.unet.config.in_channels, 
                           config.HEIGHT//8, config.WIDTH//8), 
                           generator = generator).to(torch.float16)
    latent = latent.to(config.DEVICE)
    latent = latent * scheduler.init_noise_sigma
    return latent

def run_prediction(pipe, text_embeddings, scheduler, latent, loss_function=None):
    for i, t in tqdm(enumerate(scheduler.timesteps), total = len(scheduler.timesteps)):
        latent_model_input = torch.cat([latent] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input.to(torch.float16), t, encoder_hidden_states=text_embeddings)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        if loss_function and i%5 == 0:
            latent = latent.detach().requires_grad_()
            latent_x0 = latent - sigma * noise_pred

            denoised_images = pipe.vae.decode((1/ 0.18215) * latent_x0).sample / 2 + 0.5 # range(0,1)

            loss = loss_function(denoised_images) * config.LOSS_SCALE
            print(f"loss {loss}")

            cond_grad = torch.autograd.grad(loss, latent)[0]
            latent = latent.detach() - cond_grad * sigma**2
        
        latent = scheduler.step(noise_pred,t, latent).prev_sample

    return latent

def generate_images(pipe, seed_number, prompt, loss_function=None):

    scheduler = LMSDiscreteScheduler(beta_start = 0.00085, 
                                     beta_end = 0.012, 
                                     beta_schedule = "scaled_linear", 
                                     num_train_timesteps = 1000)
    scheduler.set_timesteps(config.NUM_INFERENCE_STEPS)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)

    text_embeddings = construct_text_embeddings(pipe=pipe, prompt=prompt)
    latent = initialize_latent(seed_number=seed_number, pipe=pipe, scheduler=scheduler)
    latent = run_prediction(pipe=pipe, text_embeddings=text_embeddings, 
                            scheduler=scheduler, latent=latent, 
                            loss_function=loss_function)

    return latent