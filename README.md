# Pix2Pix GAN - Image-to-Image Translation

A complete PyTorch implementation of Pix2Pix for image-to-image translation tasks (satellite-to-map, sketch-to-photo, day-to-night, etc.).

![Pix2Pix](https://img.shields.io/badge/Model-Pix2Pix-blue) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red) ![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **U-Net Generator** with skip connections for high-quality outputs
- **PatchGAN Discriminator** for realistic texture generation
- **Mixed Precision Training** for faster training on GPU
- **CPU-Optimized Mode** for testing without GPU
- **Automatic Checkpointing** with resume capability
- **Progress Tracking** with real-time loss monitoring
- **Inference Script** for easy prediction generation
- **Dataset Downloader** for quick setup

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance Tips](#performance-tips)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training, optional)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Pix2Pix_from_scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### GPU Support (Optional but Recommended)

For CUDA-enabled PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## ⚡ Quick Start

### 1. Download Sample Dataset

```bash
# Download maps dataset (satellite → map)
python download_data.py --dataset maps

# Or download facades dataset (labels → building)
python download_data.py --dataset facades
```

### 2. Train the Model

**For GPU (Recommended):**
```bash
python train.py
```

**For CPU (Faster Testing):**
```bash
python train_cpu.py  # Uses optimized settings
```

### 3. Generate Predictions

```bash
# Single image
python inference.py --checkpoint gen.pth.tar --input image.jpg --output predictions/

# Batch processing
python inference.py --checkpoint gen.pth.tar --input input_folder/ --output predictions/
```

## 📁 Dataset Preparation

### Format Requirements

Images should be **horizontally concatenated**: `[input_image | target_image]`

- **Total width**: 1200px (600px input + 600px target)
- **Height**: Any (will be resized to 256×256)
- **Format**: PNG, JPG, or JPEG

### Directory Structure

```
data/
└── your_dataset/
    ├── train/
    │   ├── 1.jpg  # [input | target]
    │   ├── 2.jpg
    │   └── ...
    └── val/
        ├── 1.jpg
        └── ...
```

### Using Custom Data

1. Prepare paired images (input + target)
2. Concatenate them horizontally (600px each)
3. Place in `data/your_dataset/train/` and `data/your_dataset/val/`
4. Update `config.py`:
   ```python
   TRAIN_DIR = "data/your_dataset/train"
   VAL_DIR = "data/your_dataset/val"
   ```

## 🎓 Training

### Basic Training

```bash
# GPU training (full quality)
python train.py

# CPU training (optimized for speed)
python train_cpu.py
```

### Resume Training

```bash
# Set in config.py
LOAD_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"

# Then run training
python train.py
```

### Monitor Progress

- **Console**: Real-time loss metrics via tqdm
- **Evaluation folder**: Generated samples saved every epoch
- **Checkpoints**: Models saved every 5 epochs

### Training Time Estimates

| Configuration | Hardware | Time (500 epochs) |
|--------------|----------|-------------------|
| Full (256px, batch 16) | GPU (RTX 3080) | 2-3 hours |
| Full (256px, batch 16) | CPU | 48-72 hours |
| Optimized (128px, batch 4) | CPU | 12-18 hours |
| Quick test (128px, 10 epochs) | CPU | 20-40 minutes |

## 🎨 Inference

### Generate Single Prediction

```bash
python inference.py \
    --checkpoint gen.pth.tar \
    --input path/to/image.jpg \
    --output predictions/
```

### Batch Processing

```bash
python inference.py \
    --checkpoint gen.pth.tar \
    --input input_folder/ \
    --output predictions/
```

### Programmatic Usage

```python
from inference import load_generator, generate_prediction
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = load_generator("gen.pth.tar", device)
generate_prediction(gen, "input.jpg", "output.jpg", device)
```

## ⚙️ Configuration

Edit `config.py` to customize training:

```python
# Model parameters
IMAGE_SIZE = 256          # Image resolution (256, 128, or 512)
BATCH_SIZE = 16           # Batch size (reduce if OOM)
LEARNING_RATE = 2e-4      # Learning rate
L1_LAMBDA = 100           # L1 loss weight (100-200)

# Training settings
NUM_EPOCHS = 500          # Total epochs
SAVE_MODEL = True         # Save checkpoints
LOAD_MODEL = False        # Resume from checkpoint

# Dataset paths
TRAIN_DIR = "data/maps/train"
VAL_DIR = "data/maps/val"

# Hardware
NUM_WORKERS = 2           # DataLoader workers (0 for CPU)
```

### CPU-Optimized Settings

Use `config_cpu.py` for faster training on CPU:

```python
IMAGE_SIZE = 128          # Reduced resolution (4x faster)
BATCH_SIZE = 4            # Smaller batches
NUM_EPOCHS = 50           # Fewer epochs
NUM_WORKERS = 0           # Best for CPU
```

## 📂 Project Structure

```
Pix2Pix_from_scratch/
├── config.py                  # Main configuration
├── config_cpu.py              # CPU-optimized config
├── train.py                   # Main training script
├── train_cpu.py               # CPU-optimized training
├── inference.py               # Generate predictions
├── setup.py                   # Environment setup
├── download_data.py           # Download datasets
│
├── generator_model.py         # U-Net generator
├── discriminator_model.py     # PatchGAN discriminator
├── dataset.py                 # Dataset loader
├── utils.py                   # Helper functions
│
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── .gitignore                # Git ignore rules
│
├── data/                      # Datasets (gitignored)
│   └── maps/
│       ├── train/
│       └── val/
│
├── evaluation/                # Training samples (gitignored)
├── predictions/               # Inference outputs (gitignored)
│
└── *.pth.tar                  # Model checkpoints (gitignored)
```

## 🚀 Performance Tips

### GPU Training (Recommended)

**Free GPU Options:**
- [Google Colab](https://colab.research.google.com/) (T4/V100 GPU)
- [Kaggle Notebooks](https://www.kaggle.com/) (P100 GPU)

**Setup on Colab:**
```python
# Enable GPU: Runtime → Change runtime type → GPU
!pip install torch torchvision albumentations tqdm
!python train.py
```

### CPU Optimization

If you must use CPU:

1. **Use CPU-optimized script**: `python train_cpu.py`
2. **Reduce image size**: Set `IMAGE_SIZE = 128`
3. **Smaller batches**: Set `BATCH_SIZE = 4`
4. **Fewer epochs**: Set `NUM_EPOCHS = 50`
5. **Disable workers**: Set `NUM_WORKERS = 0`

### Memory Optimization

If you run out of memory:

```python
BATCH_SIZE = 8            # Reduce batch size
IMAGE_SIZE = 128          # Reduce image resolution
NUM_WORKERS = 0           # Reduce workers
```

## 📊 Results

### Sample Outputs

Training progression (satellite → map):

| Epoch 0 | Epoch 50 | Epoch 200 | Epoch 500 |
|---------|----------|-----------|-----------|
| ![](evaluation/y_gen_0.png) | ![](evaluation/y_gen_50.png) | ![](evaluation/y_gen_200.png) | ![](evaluation/y_gen_500.png) |

*Results improve significantly over training epochs*

### Model Architecture

**Generator (U-Net):**
- **Encoder**: 7 downsampling blocks (Conv → BatchNorm → LeakyReLU)
- **Bottleneck**: Single convolution layer
- **Decoder**: 7 upsampling blocks with skip connections
- **Output**: Tanh activation for [-1, 1] range

**Discriminator (PatchGAN):**
- 5 convolutional blocks
- Outputs 30×30 patch classifications
- Each patch determines if region is real/fake

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# In config.py
BATCH_SIZE = 8  # or 4
IMAGE_SIZE = 128
```

**2. No images found**
```bash
python setup.py  # Verify directory structure
ls data/maps/train  # Check if images exist
```

**3. Slow training on CPU**
```bash
python train_cpu.py  # Use optimized version
```

**4. Poor quality results**
- Train for more epochs (500+)
- Increase `L1_LAMBDA` (try 200)
- Check data quality and pairing
- Ensure images are properly aligned

**5. Import errors**
```bash
pip install -r requirements.txt --upgrade
```

**6. Checkpoint loading fails**
```python
# In config.py
LOAD_MODEL = False  # Start fresh
```

### Getting Help

1. Check [Issues](link-to-issues) for similar problems
2. Review `PERFORMANCE_TIPS.md` for optimization
3. Verify setup with `python setup.py`
4. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

## 📚 References

- **Paper**: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (Isola et al., 2017)
- **Project Page**: [Pix2Pix Official](https://phillipi.github.io/pix2pix/)
- **Original Implementation**: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={CVPR},
  year={2017}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This is an educational implementation. For production use, consider the official implementation.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Phillip Isola et al. for the original Pix2Pix paper
- PyTorch team for the excellent framework
- Berkeley AI Research for dataset hosting

## 📧 Contact

For questions or suggestions:
- Open an [Issue](link-to-issues)
- Email: [ouayazza.abdelaziz@gmail.com]

---

**Star ⭐ this repo if you find it helpful!**

Made with ❤️ by [Abdelaziz ouayazza]