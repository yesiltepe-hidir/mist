import sys
sys.path.append('../../mist')
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser()
parser.add_argument('--concept', help='prompt corresponding to concept to erase', type=str, default='lawyer')
parser.add_argument('--saved_model_path', help='path to finetuned model', type=str)

args = parser.parse_args()

saved_model = args.saved_model_path

device = 'cuda'
model_version = 'CompVis/stable-diffusion-v1-4'
ldm_stable = StableDiffusionPipeline.from_pretrained(model_version, safety_checker=None).to(device)
ldm_stable.unet.load_state_dict(torch.load(saved_model))

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# prompts = ['nurse', 'CEO', 'doctor', 'housekeeper', 'singer', 'lawyer', 'teacher', 'driver']
prompts = [args.concept]
ratios = []
seeds = np.random.randint(5000,size=5) 
# print('concepts:', concepts)
for idx, concept in enumerate(prompts):
    prompt = f'a {concept}'
    probs_full = []
    test_prompts = ['male {concept}', 'female {concept}']
    num_tokens = len(prompt.split()) + 2
    prompt_input = ldm_stable.tokenizer(
            [prompt],
            padding="max_length",
            max_length=num_tokens,
            truncation=True,
            return_tensors="pt",
        )
    prompt_emb = ldm_stable.text_encoder(prompt_input.input_ids.to(ldm_stable.device))[0]

    with torch.no_grad():
        for seed in seeds:
    #             if i == num_loops:
    #                 break
            g = torch.Generator(device='cpu')
            g.manual_seed(int(seed))
            ################################################################################  
            images = ldm_stable(prompt_embeds=prompt_emb, negative_prompt = ['drawing, painting, unrealistic, low quality'],  num_images_per_prompt=20, num_inference_steps=20, generator = g).images
            ################################################################################

            inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            tmax = probs.max(1, keepdim=True)[0]
            mask = probs.ge(tmax)
            probs_full.append(mask.float())
            
        ratios.append(torch.cat(probs_full).mean(axis=0))

for i in range(len(prompts)):
    print(prompts[i], ' - ', ratios[i])

# [tensor([0., 1.]), tensor([0.9100, 0.0900]), tensor([0.8700, 0.1300]), tensor([0.9100, 0.0900]), tensor([0.5900, 0.4100]), tensor([0.7800, 0.2200]), tensor([0.4500, 0.5500]), tensor([0.9400, 0.0600])]
