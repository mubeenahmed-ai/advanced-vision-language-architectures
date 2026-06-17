<h1 align="center"><b>Advanced Deep Vision & Vision-Language Architectures</b></h1>

This repository serves as a comprehensive exploration and implementation of State-of-the-Art (SOTA) computer vision and multimodal architectures. It spans the evolution of visual modeling from deep convolutional networks to attention-based transformers, generative models, and zero-shot vision-language alignment.

## Repository Structure & Experiments

### 1. ResNet-152: Transfer Learning & Layer Freezing (`01-resnet-transfer-learning.ipynb`)
* **Objective:** Optimize compute efficiency during transfer learning.
* **Insights:** Conducted an ablation study comparing the validation accuracy of freezing the backbone vs. fine-tuning specific residual blocks, demonstrating the memory/performance trade-offs of deep CNN feature extractors.

### 2. Vision Transformers (ViT): Attention Pooling Mechanisms (`02-vit-pooling-analysis.ipynb`)
* **Objective:** Analyze how self-attention networks aggregate spatial information.
* **Insights:** Benchmarked the traditional `[CLS]` token classification approach against Mean Pooling across patch embeddings, visualizing the per-class accuracy variance in transformer architectures.

### 3. Variational Autoencoders (VAE): KL-Annealing (`03-vae-kl-annealing.ipynb`)
* **Objective:** Improve generative latent space continuity.
* **Insights:** Addressed the posterior collapse problem in standard VAEs by implementing KL-divergence annealing. Evaluated the visual fidelity of reconstructed vs. generated outputs under annealed conditions.

### 4. CLIP: Modality Gap & Zero-Shot Alignment (`04-clip-modality-gap.ipynb`)
* **Objective:** Evaluate joint vision-language embedding spaces.
* **Insights:** Implemented zero-shot classification on the STL-10 dataset using OpenAI's CLIP. Analyzed the "modality gap" using Procrustes alignment to determine if rigid transformations between image and text embeddings improve zero-shot accuracy over contrastive baselines.

## Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Key Libraries:** Torchvision, Hugging Face Transformers, OpenAI CLIP, Scikit-learn, Matplotlib.
* **Hardware:** CUDA-enabled GPU (T4/A100)

## Core Engineering Competencies Demonstrated
1. **Compute Optimization:** Managing gradients (`requires_grad`) and freezing layers for memory-efficient training.
2. **Architectural Modification:** Re-routing pooling mechanisms in Transformers and modifying the latent bottlenecks in Generative models.
3. **Multimodal Evaluation:** Handling aligned vector spaces and running zero-shot inference without fine-tuning.
