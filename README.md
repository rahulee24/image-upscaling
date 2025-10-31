# Image Upscaling using SRGAN

This project implements a **Super-Resolution Generative Adversarial Network (SRGAN)** for upscaling low-resolution images to high-resolution images with enhanced detail and quality. The model uses a combination of adversarial training, perceptual loss, and content loss to generate photorealistic high-resolution images.

## 📋 Project Overview

The SRGAN architecture consists of:
- **Generator Network**: A deep residual network that upscales low-resolution images by 4x
- **Discriminator Network**: A convolutional network that distinguishes between real high-resolution images and generated images
- **VGG Feature Extractor**: Uses pre-trained VGG19 to compute perceptual loss for better visual quality

The model is trained using a combination of:
- **Adversarial Loss**: Helps generate realistic textures
- **Content Loss**: Ensures pixel-wise similarity
- **Perceptual Loss**: Maintains high-level feature similarity using VGG19

## 📁 File Structure and Description

### Core Model Files

- **`generator.py`**: Implements the Generator network with residual blocks and sub-pixel convolution layers for upscaling. Contains:
  - Residual blocks for deep feature extraction
  - Upsampling blocks for 4x resolution enhancement
  - Skip connections for gradient flow

- **`discriminator.py`**: Implements the Discriminator network that classifies images as real or generated. Uses:
  - Convolutional layers with increasing depth
  - Leaky ReLU activations
  - Dense layers for binary classification

- **`vgg_feature_extractor.py`**: Extracts features from pre-trained VGG19 network for computing perceptual loss. This helps maintain semantic content in generated images.

### Data and Training Files

- **`dataset.py`**: Handles data loading and preprocessing. Includes:
  - Image loading from dataset directory
  - Creation of low-resolution and high-resolution image pairs
  - Data augmentation and normalization
  - PyTorch DataLoader setup

- **`train.py`**: Main training script that:
  - Initializes Generator and Discriminator models
  - Implements the training loop with alternating updates
  - Computes combined loss (adversarial + content + perceptual)
  - Saves model checkpoints during training
  - Tracks and logs training metrics

- **`inference.py`**: Script for testing the trained model on new images:
  - Loads trained generator weights
  - Upscales input low-resolution images
  - Saves the super-resolved output images

### Configuration File

- **`reequire.txt`**: Lists all Python dependencies required to run the project (note: filename should be `requirements.txt`)

## 🚀 Installation and Setup

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for training)
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/rahulee24/image-upscaling.git
cd image-upscaling
```

### Step 2: Install Dependencies

```bash
pip install -r reequire.txt
```

Required packages include:
- PyTorch
- torchvision
- NumPy
- Pillow
- matplotlib (for visualization)

### Step 3: Download Dataset

For training, you'll need a high-resolution image dataset. Recommended options:

#### Option 1: DIV2K Dataset (Recommended)
- **Download Link**: [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- High-quality 2K resolution images
- 800 training images + 100 validation images
- Widely used benchmark for super-resolution tasks

#### Option 2: Flickr2K Dataset
- **Download Link**: [Flickr2K Dataset](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)
- 2,650 high-resolution images from Flickr

#### Option 3: CelebA-HQ (for faces)
- **Download Link**: [CelebA-HQ Dataset](https://github.com/tkarras/progressive_growing_of_gans)
- High-quality celebrity faces at 1024×1024 resolution

### Step 4: Prepare Your Data

Organize your dataset in the following structure:
```
image-upscaling/
├── data/
│   ├── train/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── test/
│       ├── test1.png
│       └── ...
├── dataset.py
├── generator.py
└── ...
```

## 🏋️ Training the Model

### Basic Training

```bash
python train.py
```

This will:
1. Load the training dataset from the specified directory
2. Initialize Generator and Discriminator networks
3. Train the model for the specified number of epochs
4. Save model checkpoints periodically
5. Display training loss metrics

### Training Parameters

You can modify training parameters in `train.py`:
- **Epochs**: Number of training iterations
- **Batch size**: Number of images per batch
- **Learning rate**: Optimization step size
- **Loss weights**: Balance between adversarial, content, and perceptual losses

### Monitoring Training

The training script will:
- Print loss values for Generator and Discriminator
- Save model checkpoints at regular intervals
- Generate sample outputs for visual inspection

## 🔍 Running Inference

Once the model is trained, use it to upscale new images:

```bash
python inference.py --input_image path/to/low_res_image.png --output_image path/to/output.png
```

### Parameters:
- `--input_image`: Path to the low-resolution input image
- `--output_image`: Path where the high-resolution output will be saved
- `--model_path`: (Optional) Path to trained generator weights

### Example Usage

```bash
# Upscale a single image
python inference.py --input_image ./test/input.png --output_image ./results/upscaled.png

# Batch process multiple images
python inference.py --input_dir ./test/ --output_dir ./results/
```

## 📊 Model Architecture Details

### Generator
- **Input**: Low-resolution image (64×64 or similar)
- **Output**: High-resolution image (256×256, 4x upscale)
- **Architecture**: 
  - Initial convolution layer
  - 16 residual blocks with skip connections
  - 2 upsampling blocks (2x each = 4x total)
  - Final convolution layer

### Discriminator
- **Input**: High-resolution image (real or generated)
- **Output**: Probability score (real vs fake)
- **Architecture**:
  - 8 convolutional layers with increasing filters
  - Batch normalization and LeakyReLU activations
  - Dense layers for final classification

## 📈 Results and Performance

The trained SRGAN model can:
- Upscale images by 4x resolution
- Recover fine details and textures
- Generate photorealistic images
- Work on various image types (faces, landscapes, objects)

## 🛠️ Troubleshooting

**Issue**: Out of memory error during training  
**Solution**: Reduce batch size in `train.py`

**Issue**: Slow training speed  
**Solution**: Ensure you're using GPU acceleration (CUDA)

**Issue**: Poor quality results  
**Solution**: Train for more epochs or adjust loss weights

## 📚 References and Resources

- [SRGAN Paper](https://arxiv.org/abs/1609.04802): "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/): High-quality training data
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html): Framework reference

## 📝 Notes

- Training SRGAN requires significant computational resources (GPU recommended)
- The model works best with diverse training data
- Experiment with different loss weight combinations for optimal results
- Pre-training with pixel-wise loss before adversarial training can improve convergence

## 👨‍💨 Author

**Rahul Roy** - [GitHub Profile](https://github.com/rahulee24)

## 📄 License

This project is open-source and available for educational and research purposes.

---

**Dataset Resources**:
- [DIV2K High-Resolution Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - Primary recommendation
- [Flickr2K Dataset](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) - Additional training data
- [Set5 & Set14](https://github.com/jbhuang0604/SelfExSR) - Standard test sets for benchmarking
