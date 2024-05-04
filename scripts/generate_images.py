import sys
sys.path.append('/home/grads/hidir/unified-concept-editing')
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from image_utils import view_images
import os

seeds = np.random.randint(0, 5000, 1)
seeds = [421]
models_path = 'checkpoints/'
with torch.no_grad():
        for model_name in os.listdir(models_path):
                print(model_name)
                split = model_name.split('_')   
                occupation = split[-1]
                prompt = [f'a headshot of a guard']
                saved_model = models_path + model_name
                offset = 0
                for seed in seeds:
                        seed = int(seed)
                        save_path = f'appendix_checkpoints/appendix_images/'
                        os.makedirs(save_path, exist_ok=True)
                        device = 'cuda'
                        model_version = 'CompVis/stable-diffusion-v1-4'
                        model = StableDiffusionPipeline.from_pretrained(model_version, safety_checker=None).to(device)
                        model.unet.load_state_dict(torch.load(saved_model))
                        g = torch.Generator(device='cpu')
                        g.manual_seed(int(seed))
                        # g = None
                        num_tokens = len(prompt[0].split()) + 2
                        prompt_input = model.tokenizer(
                                prompt,
                                padding="max_length",
                                max_length=model.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt")
                        prompt_emb = model.text_encoder(prompt_input.input_ids.to(model.device))[0]
                        images = model(prompt_embeds=prompt_emb, negative_prompt = ['black and white, low quality'], num_images_per_prompt=51, num_inference_steps=20, generator = g).images
                        for m in range(len(images)):
                                images[m].save(save_path + str(offset + m) +' .png')
                                images[m].save(save_path + str(offset + m) +' .png')
                        torch.cuda.empty_cache()
                        del model
                        
