---
title: DeepTrace
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Explainable AI-generated face detection (a detection aid, not proof)
---

# DeepTrace — explainable AI-generated face detector

Upload a photo; DeepTrace finds each face, estimates the calibrated probability that it is AI-generated,
and shows a heatmap of the regions that drove the decision.

**A detection aid, not proof.** It is weaker on images from newer generators and on heavily compressed
images. Never use a result as the sole basis for an accusation.

- Uploads are processed in memory and discarded; this Space has no accounts or history.
- The model was trained on FFHQ (non-commercial license) and StyleGAN faces.
- Source code, metrics and known limitations: see the GitHub repository linked in the Space settings.

Space variables: `DEEPTRACE_MODEL_REPO_ID` (model bundle repo on the Hub) and `DEEPTRACE_MODEL_REVISION`
(pin a commit hash).
