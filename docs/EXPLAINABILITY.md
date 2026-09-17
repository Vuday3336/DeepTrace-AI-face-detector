# DeepTrace — Explainability report

> Status: **TBD** — fill this in after running `ml/notebooks/03_robustness_explain_export.ipynb`.
> Galleries live in `docs/plots/explain/<model>/`; sanity-check numbers in `ml/reports/explain_<model>.json`.

## How to write this honestly

- Describe **observations on sampled images**, with counts: "In 8 of 12 sampled false positives on
  `crossgen_full`, heat concentrates on hair boundaries." Not: "the model detects hair artifacts."
- Samples are drawn with a fixed seed (no cherry-picking). The worst-error galleries are, by
  construction, the most confident mistakes — say so when quoting them.
- A heatmap shows **where** evidence was, not **why**. Grad-CAM is class-specific; attention rollout
  is class-agnostic.
- If the model-randomisation sanity check shows high correlation between the trained and random
  model's maps, the heatmaps mostly reflect image structure: report that and do not interpret them.

## Methods

| Model family | Offline (galleries) | Served in the app |
|---|---|---|
| CNN (EfficientNet-B0, ConvNeXt-Tiny) | Grad-CAM, Grad-CAM++ on the last feature map | Grad-CAM, closed-form from ONNX outputs (parity-tested against autograd) |
| ViT (ViT-S/16, CLIP ViT-B/16 probe) | Grad-CAM on last-block tokens (14×14 reshape), attention rollout | Attention rollout from ONNX attention outputs |

**Grad-CAM vs attention rollout.** Grad-CAM weights the final feature map by the gradient of the
"AI-generated" logit, so it highlights regions that pushed the score *toward fake*. Attention rollout
multiplies attention matrices through all layers to estimate how much each patch flows into the
classification token; it shows where the model *looked*, regardless of which way that evidence
pointed.

## Sanity check (Adebayo et al., 2018)

| Model | Method | Mean Spearman ρ (trained vs randomised weights) | Interpretation |
|---|---|---|---|
| effnet_b0 | Grad-CAM | TBD | TBD |
| effnet_b0 | Grad-CAM++ | TBD | TBD |
| clip_probe | Grad-CAM | TBD | TBD |
| clip_probe | Rollout | TBD | TBD |

ρ near 0 = maps depend on learned weights (good). ρ near 1 = maps would look the same for an
untrained model.

## Observations — in-distribution (StyleGAN)

- Correct AI-generated (TP): TBD
- Correct real (TN): TBD
- False positives (real called AI): TBD
- False negatives (AI called real): TBD

## Observations — cross-generator (diffusion)

- TBD (compare with the in-distribution patterns: does the model look at the same regions when it
  fails on SDXL / FLUX faces?)

## Frequency analysis

See `docs/plots/fft/`. TBD: do StyleGAN crops show periodic high-frequency peaks that diffusion crops
lack? Does that match the cross-generator drop in `docs/METRICS.md`?

## Limits of these explanations

- Heatmaps are computed on 224×224 crops and upsampled from a 7×7 (CNN) or 14×14 (ViT) grid, so they
  cannot localise pixel-level artifacts.
- They explain one model's behaviour on one image; they are not evidence about the image itself.
