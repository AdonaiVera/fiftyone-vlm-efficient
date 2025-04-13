# 🌍 FiftyOne VLM Efficient
A comprehensive toolkit for improving Vision-Language Model (VLM) training data quality through state-of-the-art research techniques in FiftyOne

## 🎯 Project Vision

In the rapidly evolving field of Vision-Language Models (VLMs), data quality has emerged as a critical factor in model performance. This project aims to make advanced data quality improvement techniques accessible to anyone working with VLMs, integrating proven methods from top research labs like NVIDIA's VILA, LLaVA and more.

Our goal is to make state-of-the-art data quality techniques accessible to everyone - from individual developers and hobbyists to large companies. We provide easy-to-use plugins that help improve training data efficiency and quality for vision-language models, without requiring deep expertise in the field.

## 🚀 Key Features

### 1. Dataset Pruning
- **SIEVE Method** - State-of-the-art multimodal dataset pruning using image captioning models
  - Generates multiple captions for semantic similarity comparison
  - Fuses with CLIP scores for improved results
  - Efficient dataset filtering based on quality metrics
  - Paper: [SIEVE: Multimodal Dataset Pruning Using Image Captioning Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.pdf)

<img src="https://storage.googleapis.com/tecla/sieve.gif" alt="sieve-pluging" width="600"/>

## 🛠️ Installation

First, install the base requirements:

```shell
pip install -U fiftyone transformers>=4.30.0 accelerate einops timm torch sentence-transformers>=2.2.2 numpy>=1.24.0 Pillow>=9.5.0
pip install git+https://github.com/openai/CLIP.git
```

Then, install the VLM plugin:

```shell
fiftyone plugins download https://github.com/AdonaiVera/fiftyone-vlm-efficient
```

You can also install requirements via:

```shell
fiftyone plugins requirements @AdonaiVera/fiftyone-vlm-efficient --install
```

## 📚 Research-Backed Techniques

This toolkit integrates several state-of-the-art approaches from recent research:

1. **SIEVE Method** (CVPR 2024)
   - Multimodal dataset pruning using image captioning models
   - Semantic alignment optimization
   - CLIP score fusion
   - Paper: [SIEVE: Multimodal Dataset Pruning Using Image Captioning Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.pdf)

## 🔮 Future Work

We plan to integrate additional research-backed techniques:

1. **Pruning Methods**
   - [PruneVid: Efficient Video Pruning for Vision-Language Models](https://arxiv.org/abs/2412.16117)
   - [Fit and Prune (FitPrune): Efficient Vision-Language Model Training](https://arxiv.org/abs/2409.10197)
   - [ECoFLaP: Efficient Coarse-to-Fine Layer-wise Pruning for Vision-Language Models](https://github.com/ylsung/ECoFLaP)
   - [VLM-Compression: Strategies for Effective Sparsity and Performance Restoration](https://github.com/Shwai-He/VLM-Compression)

2. **Advanced Pre-processing**
   - [Dynamic-S2: Multi-scale Feature Extraction](https://arxiv.org/abs/2412.04468)
   - [VILA: Adaptive Batch Processing](https://github.com/NVlabs/VILA)
   - Quality-aware data augmentation techniques

3. **Prompt Engineering**
   - [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)
   - [COCO is All You Need](https://arxiv.org/abs/2401.08968)
   - Research-validated templates
   - Multi-granularity annotations
   - Dense annotation support

4. **Knowledge Distillation**
   - DeltaLoss metric for sample selection
   - Up to 50% dataset size reduction
   - Minimal performance impact
   - Paper: [VILA: Learning Image-Text Alignments](https://arxiv.org/abs/2412.04468)


## 📊 Usage Examples
This example demonstrates how to load and process the LAION GPT4Vision captions dataset:

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset from the Hugging Face Hub
dataset = load_from_hub(
    "laion/220k-GPT4Vision-captions-from-LIVIS",
    format="parquet",
    filepath="url",
    max_samples=10, 
)

# Launch the FiftyOne app for visualization
session = fo.launch_app(dataset)

# Keep the app running until manually closed
session.wait()
```

The LAION dataset contains high-quality GPT4Vision-generated captions that can be used to:
- Train and evaluate VLM models
- Benchmark caption quality
- Generate reference captions for quality assessment
- Fine-tune captioning models

## 📚 Citations

```bibtext
@article{Sieve,
  title={Sieve: Multimodal Dataset Pruning Using Image Captioning Models},
  author={Mahmoud, Anas and Elhoushi, Mostafa and Abbas, Amro and Yang, Yu and Ardalani, Newsha and Leather, Hugh and Morcos, Ari},
  journal={CVPR},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! If you're working on VLM training and have techniques that could improve data quality, please reach out.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.