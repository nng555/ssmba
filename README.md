# SSMBA: **S**elf-**S**upervised **M**anifold **B**ased Data **A**ugmentation

## Overview

Self-Supervised Manifold Based Data Augmentation or SSMBA is a semi-supervised data augmentation method that improves both in-domain and out-of-domain performance across multiple models and tasks. SSMBA relies on the assumption that the underlying data clusters around a lower dimensional manifold. A corruption function is applied to perturb a training example off the manifold, then a reconstruction function (typically a denoising autoencoder) is used to project the noised point back onto the
manifold. ![SSMBA perturbs and reconstructs examples to move along a manifold](img/ssmba.png)<!-- .element height="50%" width="50%" -->
