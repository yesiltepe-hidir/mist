# MIST: Mitigating Intersectional Bias with Disentangled Cross-Attention Editing in Text-to-Image Diffusion Models
<p align="center">
  <img src="https://github.com/yesiltepe-hidir/mist/assets/70890453/d884b120-0faf-4e18-8d85-a3e3d0a58204" />
</p>

`[Example Usage]` To finetune the cross-attention weights:

```bash
python train-scripts/mist_train.py --concept "a lawyer" --attributes "young woman with glasses, young woman without glasses, old man with glasses, old man without glasses, young man with glasses, young man without glasses, old woman with glasses, old woman without glasses"  --max_bias_diff 0.05  --num_images 20
````

The checkpoints will be available under `appendix_checkpoints` folder.
