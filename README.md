# 🎨 Pix2Pix GAN - Image-to-Image Translation

> A complete PyTorch implementation of Pix2Pix for image-to-image translation tasks (satellite→map, sketch→photo, day→night, etc.)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Training](https://img.shields.io/badge/Training-CPU%20%7C%20GPU-orange)]()

## 📸 Results & Demo

### Satellite → Map Translation

<table>
  <tr>
    <th>Input (Satellite)</th>
    <th>Generated (Map)</th>
    <th>Ground Truth</th>
  </tr>
  <tr>
    <td><img src="evaluation/input_0.png" alt="Input"></td>
    <td><img src="evaluation/y_gen_49.png" alt="Generated"></td>
    <td><img src="evaluation/label_0.png" alt="Ground Truth"></td>
  </tr>
</table>

### Training Progress

| Epoch 1 | Epoch 10 | Epoch 25 | Epoch 50 |
|---------|----------|----------|----------|
| ![](evaluation/y_gen_1.png) | ![](evaluation/y_gen_10.png) | ![](evaluation/y_gen_25.png) | ![](evaluation/y_gen_49.png) |

*Watch the model learn to generate accurate maps from satellite imagery!*

## ✨ Key Features

- **🏗️ U-Net Generator** with skip connections for high-quality outputs
- **🎯 PatchGAN Discriminator** for realistic texture generation
- **⚡ Mixed Precision Training** for 2x faster training on GPU
- **💻 CPU-Optimized Mode** for testing without expensive GPU
- **💾 Automatic Checkpointing** with seamless resume capability
- **📊 Real-time Monitoring** with loss tracking and sample generation
- **🎨 Easy Inference** script for generating predictions
- **📦 One-Click Dataset Download** for quick setup

## 🎯 What is Pix2Pix?

Pix2Pix is a conditional GAN that learns to map images from one domain to another. It consists of:
- **Generator (U-Net)**: Transforms input images to target domain
- **Discriminator (PatchGAN)**: Distinguishes real vs generated images

**Applications:**
- 🛰️ Satellite → Map
- 🏢 Building Labels → Photos
- ⚫⚪ Edges → Photos
- 🌙 Day → Night
- 🎨 Sketch → Photo

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Pix2Pix_from_scratch.git
cd Pix2Pix_from_scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup.py
```

### 2️⃣ Download Dataset

```bash
# Download maps dataset (satellite → map)
python download_data.py --dataset maps

# Or facades dataset (labels → building)
python download_data.py --dataset facades
```

### 3️⃣ Train Model

**GPU Training (Recommended):**
```bash
python train.py
```

**CPU Training (Faster Testing):**
```bash
python train_cpu.py  # ~90 minutes for 50 epochs
```

### 4️⃣ Generate Predictions

```bash
# Single image
python inference.py --checkpoint gen.pth.tar --input image.jpg

# Batch processing
python inference.py --checkpoint gen.pth.tar --input folder/ --output predictions/
```

## 📊 Training Results

### Loss Curves (50 epochs on CPU)

```
Epoch 1:  D_loss: 0.5662, G_loss: 59.3279
Epoch 10: D_loss: 0.2996, G_loss: 14.7441
Epoch 25: D_loss: 0.5234, G_loss: 13.3018
Epoch 50: D_loss: 0.4040, G_loss: 12.2581
```

✅ **Model converges successfully** - Generator loss decreased from 59.3 to 12.3  
✅ **Stable training** - Discriminator maintains balanced loss  
✅ **Fast convergence** - Good results visible after just 10 epochs

### Performance Benchmarks

| Configuration | Hardware | Time (50 epochs) | Quality |
|--------------|----------|------------------|---------|
| Optimized (128px) | CPU | **~90 minutes** | Medium |
| Full (256px) | CPU | ~6-8 hours | High |
| Full (256px) | GPU (RTX 3080) | **~45 minutes** | High |
| Full (256px) | Colab T4 | ~2 hours | High |

## 🏗️ Architecture

### Generator: U-Net

```
Input (3×256×256)
    ↓
[Encoder: 7 Conv Blocks] → Features: 64→128→256→512→512→512→512
    ↓
[Bottleneck: 512 features]
    ↓
[Decoder: 7 Deconv Blocks + Skip Connections]
    ↓
Output (3×256×256) with Tanh activation
```

**Key Features:**
- Skip connections preserve spatial information
- Dropout for regularization
- BatchNorm for stable training

### Discriminator: PatchGAN

```
Concat(Input, Target) (6×256×256)
    ↓
[5 Conv Blocks] → Features: 64→128→256→512
    ↓
Output: 30×30 patch classifications
```

**Why PatchGAN?**
- Classifies image patches instead of whole image
- Better texture quality
- Faster training

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Training Settings
IMAGE_SIZE = 256          # Image resolution (128, 256, or 512)
BATCH_SIZE = 16           # Batch size (reduce if OOM)
NUM_EPOCHS = 500          # Training epochs
LEARNING_RATE = 2e-4      # Learning rate

# Loss Weights
L1_LAMBDA = 100           # L1 reconstruction loss weight (↑ = more detail)

# Paths
TRAIN_DIR = "data/maps/train"
VAL_DIR = "data/maps/val"

# Hardware
NUM_WORKERS = 2           # DataLoader workers (0 for CPU)
SAVE_MODEL = True         # Save checkpoints every 5 epochs
LOAD_MODEL = False        # Resume from checkpoint
```

## 📁 Project Structure

```
Pix2Pix_from_scratch/
│
├── 🎯 Core Models
│   ├── generator_model.py        # U-Net generator
│   └── discriminator_model.py    # PatchGAN discriminator
│
├── 🚂 Training
│   ├── train.py                  # Main training (GPU/CPU)
│   ├── train_cpu.py              # CPU-optimized training
│   ├── config.py                 # Main configuration
│   └── config_cpu.py             # CPU configuration
│
├── 📊 Data
│   ├── dataset.py                # Dataset loader
│   └── download_data.py          # Dataset downloader
│
├── 🔧 Utils
│   ├── utils.py                  # Helper functions
│   ├── setup.py                  # Environment verification
│   └── inference.py              # Generate predictions
│
├── 📚 Documentation
│   ├── README.md                 # This file
│   ├── INSTALLATION.md           # Detailed setup guide
│   ├── CONTRIBUTING.md           # Contribution guidelines
│   └── PERFORMANCE_TIPS.md       # Optimization tips
│
└── 📦 Generated (gitignored)
    ├── data/                     # Datasets
    ├── evaluation/               # Training samples
    ├── predictions/              # Inference outputs
    └── *.pth.tar                # Model checkpoints
```

## 💡 Tips & Tricks

### Getting Better Results

1. **Train Longer**: 200-500 epochs for best quality
2. **Adjust L1_LAMBDA**: Higher values (150-200) preserve more details
3. **Use More Data**: More training samples = better generalization
4. **GPU Training**: 50-100x faster than CPU

### CPU Optimization

If you must use CPU:

```python
# config_cpu.py settings
IMAGE_SIZE = 128      # 4x faster than 256
BATCH_SIZE = 4        # Lower memory usage
NUM_EPOCHS = 50       # Reasonable time
NUM_WORKERS = 0       # Best for CPU
```

### Memory Issues

```python
BATCH_SIZE = 8        # or 4
IMAGE_SIZE = 128      # Reduce resolution
```

## 🌐 Cloud Training (Free GPU)

### Google Colab

```python
# In Colab notebook
!pip install albumentations tqdm
!git clone https://github.com/yourusername/Pix2Pix_from_scratch.git
%cd Pix2Pix_from_scratch
!python download_data.py --dataset maps
!python train.py
```

**Enable GPU**: Runtime → Change runtime type → T4 GPU

### Kaggle Notebooks

1. Create new notebook
2. Enable **GPU P100 accelerator**
3. Upload code and run

## 📈 Training Progress

The model learns progressively:

**Epoch 1-10**: Basic shapes and colors  
**Epoch 10-25**: Refined details and textures  
**Epoch 25-50**: High-quality, realistic outputs  
**Epoch 50+**: Fine-tuning and edge cases

Monitor progress in the `evaluation/` folder!

## 🎨 Use Cases

### Implemented Examples

- **🗺️ Maps**: Satellite imagery → Street maps
- **🏢 Facades**: Architectural labels → Building photos

### Potential Applications

- **🎨 Art**: Sketches → Paintings
- **🌓 Time**: Day → Night scenes
- **🏞️ Style**: Photo → Artistic rendering
- **🎮 Gaming**: Low-res → HD textures
- **🏥 Medical**: Different imaging modalities

## 🤝 Contributing

Contributions welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md)

**Priority areas:**
- Pre-trained model weights
- Additional datasets
- Performance optimizations
- Better visualizations
- Documentation improvements

## 📚 References

### Paper
**Image-to-Image Translation with Conditional Adversarial Networks**  
*Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros*  
CVPR 2017  
📄 [Paper](https://arxiv.org/abs/1611.07004) | 🌐 [Project Page](https://phillipi.github.io/pix2pix/)

### Citation

```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={CVPR},
  year={2017}
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `BATCH_SIZE` to 8 or 4 |
| Slow training | Use `train_cpu.py` or switch to GPU |
| Poor quality | Train longer, increase `L1_LAMBDA` |
| No images found | Run `python setup.py` to verify |
| Import errors | `pip install -r requirements.txt --upgrade` |

Full guide: [INSTALLATION.md](INSTALLATION.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

This is an educational implementation. For production use, consider the [official implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## 🙏 Acknowledgments

- **Phillip Isola et al.** for the groundbreaking Pix2Pix paper
- **PyTorch Team** for the amazing framework
- **Berkeley AI Research** for hosting datasets
- **Community** for feedback and contributions

## 📞 Contact & Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/abdelaziz2003vvb/Pix2Pix-GAN---Image-to-Image-Translation/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/abdelaziz2003vvb/Pix2Pix-GAN---Image-to-Image-Translation/disscussions)
- 📧 **Email**: ouayazza.abdelaziz@gmail.com

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

[⬆ Back to Top](#-pix2pix-gan---image-to-image-translation)

</div>