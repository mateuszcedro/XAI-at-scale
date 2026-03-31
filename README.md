# Scaling Vision Models Does Not Consistently Improve Localisation-Based Explanation Qualit
This repository contain research paper and supplementary code associated with the article *Scaling Vision Models Does Not Consistently Improve Localisation-Based Explanation Qualit* (Cedro, Chlebus, 2026)

## Abstract
Artificial intelligence models are increasingly scaled to improve predictive accuracy, yet it remains unclear whether scale improves the quality of post-hoc explanations. We investigate this relationship by evaluating 11 computer vision models representing increasing levels of depth and complexity within the ResNet, DenseNet, and Vision Transformer families, trained from scratch or pretrained, across three image datasets with ground-truth segmentation masks. For each model, we generate explanations using five post-hoc explainable AI methods and quantify mask alignment using two localisation metrics: Relevance Rank Accuracy (Arras et al., 2022) and the proposed Dual-Polarity Precision, which measures positive attributions inside the class mask and negative attributions outside it. Across datasets and methods, increasing architectural depth and parameter count does not improve explanation quality in most statistical comparisons, and smaller models often match or exceed deeper variants. While pretraining typically improves predictive performance and increases the dependence of explanations on learned weights, it does not consistently increase localisation scores. We also observe scenarios in which models achieve strong predictive performance while localisation precision is near zero, suggesting that performance metrics alone may not indicate whether predictions are based on the annotated regions. These results indicate that larger models do not reliably provide higher-quality explanations, and that explainability should therefore be assessed explicitly during model selection for safety-sensitive deployments.

## Overview
![ovewview](https://github.com/mateuszcedro/XAI-at-scale/blob/main/img/overview.png)

## Explanation Example
![example](https://github.com/mateuszcedro/XAI-at-scale/blob/main/img/example.png)

