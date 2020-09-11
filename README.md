# SSMBA: **S**elf-**S**upervised **M**anifold **B**ased Data **A**ugmentation

## Overview

Self-Supervised Manifold Based Data Augmentation or SSMBA is a semi-supervised data augmentation method that improves both in-domain and out-of-domain performance across multiple models and tasks. SSMBA relies on the assumption that the underlying data clusters around a lower dimensional manifold. A corruption function is applied to perturb a training example off the manifold, then a reconstruction function (typically a denoising autoencoder) is used to project the noised point back onto the
manifold. ![SSMBA perturbs and reconstructs examples to move along a manifold](img/ssmba.png)

## SSMBA in NLP

When applied in NLP settings, we apply masked language modeling (MLM) training noise as our corruption function. Specifically, we select a fraction of tokens to apply noise to, then of these tokens, either `<MASK>` them, replace them with a random token, or leave them unchanged. In the original BERT training regime, these percentages are 80% `<MASK>`, 10% random, and 10% unchanged. Once corrupted, we use a BERT model to predict each of the selected tokens and reconstruct the input. 
![Using SSMBA in NLP](img/nlp_example.png)

## How to Use SSMBA


