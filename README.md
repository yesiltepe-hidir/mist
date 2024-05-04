# MIST: Mitigating Intersectional Bias with Disentangled Cross-Attention Editing in Text-to-Image Diffusion Models
<p align="center">
  <img src="https://github.com/yesiltepe-hidir/mist/assets/70890453/d884b120-0faf-4e18-8d85-a3e3d0a58204" />
</p>

`[Example Usage]` To finetune the cross-attention weights:

```bash
python train-scripts/mist_train.py --concept "a lawyer" --attributes "young woman with glasses, young woman without glasses, old man with glasses, old man without glasses, young man with glasses, young man without glasses, old woman with glasses, old woman without glasses"  --max_bias_diff 0.05  --num_images 20
````

The checkpoints will be available under `checkpoints` folder.

`[Qualitative Results]` To generate images with the debiased weights:

Change the path variable in `generate_images.py` file line 2 with your path. Then run the following:
```bash
 python scripts/generate_images.py
```

`[Quantitative Results]` To get the test distributions:

Change the path variable in `eval_distributions.py` file line 2 with your path. Then run the following:

```bash
 python scripts/eval_distributions.py --concept "lawyer" --saved_model_path "<ROOT_PATH>/checkpoints/<MODEL_NAME>
```
