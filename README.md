# GanForge

**Indian Institute of Technology Kanpur — ACA Summer Project**
Machine Learning, Deep Learning, and Generative AI

---

## About

GanForge is a structured, end-to-end documentation of my journey through the IIT Kanpur ACA Summer Project on Machine Learning, Deep Learning, and Generative AI. The name is a reference to the final destination of the curriculum — Generative Adversarial Networks (GANs) — the capstone of this learning arc.

This repository is organized as a progressive curriculum: starting from foundational pixel manipulation with OpenCV, advancing through deep learning theory and model training, and culminating in a full CycleGAN and Neural Style Transfer project. Every folder represents a distinct milestone in the program.

**Roll No:** 241212
**Institution:** Indian Institute of Technology Kanpur

---

## Repository Structure

```
GanForge/
│
├── Week_1/                    # OpenCV and Matplotlib fundamentals
│   ├── assignment-1.ipynb     # Image analysis notebook
│   ├── cat.jpg                # Sample input image
│   ├── dog.jpg                # Sample input image
│   ├── cv2_dog.png            # OpenCV-rendered output
│   └── mpl_dog.png            # Matplotlib-rendered output
│
├── Week_2/                    # Advanced image processing pipeline
│   ├── Task-1.py              # Full 8-stage image processing script
│   └── Assignment 2 sol/      # PDF solutions for Tasks 1 to 4
│       ├── Task-1.pdf
│       ├── Task-2.pdf
│       ├── Task-3.pdf
│       └── Task-4.pdf
│
├── Week_3/                    # Deep learning-based image retrieval
│   ├── SASIS.py               # Builds ResNet18 Annoy index
│   ├── FSI.py                 # Queries index and generates image grids
│   ├── 3Q1.pdf                # Theory questions and answers
│   └── 3Q2.pdf                # Theory questions and answers
│
├── MidEval/                   # Mid-Evaluation: Face Mask Detection (CNN)
│   └── Face_mask_detection/
│       ├── project.ipynb      # CNN model training and evaluation notebook
│       └── test1.jpg          # Sample test image
│
├── EndEval/                   # End-Evaluation: Generative AI (Group Project)
│   ├── EndEval_code.ipynb     # CycleGAN and Neural Style Transfer notebook
│   ├── EndEval_Project1_doc.pdf
│   ├── presentation_endeval_project1_ganforge.pdf
│   └── requirements.txt
│
└── LICENSE                    # Apache License 2.0
```

---

## Week 1 — Introduction to OpenCV and Matplotlib

This week focused on foundational image input/output operations — the starting point for all computer vision work. The primary goal was understanding how images are represented as numerical arrays and how different libraries interpret and display them.

### Tasks Covered

| Task | Description |
|------|-------------|
| Image I/O | Reading `.jpg` images using both `cv2.imread()` and `matplotlib.image.imread()` |
| Display | Rendering images side-by-side via `plt.subplot` |
| Shape Analysis | Examining image dimensions (H x W x C) and data type |
| Pixel Intensity | Plotting RGB channel histograms to analyze tonal distribution |

### Key Observation

OpenCV reads images in BGR format, while Matplotlib expects RGB. This distinction is a critical consideration that affects all downstream visualization.

### Files

- `Week_1/assignment-1.ipynb` — Main Jupyter notebook
- `Week_1/cv2_dog.png` — OpenCV display output
- `Week_1/mpl_dog.png` — Matplotlib display output

**Libraries Used:** Python, OpenCV, Matplotlib, NumPy

---

## Week 2 — Advanced Image Processing with OpenCV

A comprehensive, 8-stage image processing pipeline implemented in Python and OpenCV. A synthetic colorful test image is programmatically generated and passed through each transformation stage, requiring no external input image.

### Pipeline Stages

| Stage | Function | Technique |
|-------|----------|-----------|
| 1 | Color Space Conversion | RGB to HSV, channel splitting |
| 2 | Histogram Equalization | Contrast enhancement on grayscale |
| 3 | Binary Thresholding | Inverse binary threshold at pixel value 127 |
| 4 | Gray Level Reduction | Posterization — 4 gray levels (bit reduction) |
| 5 | Edge Detection | Laplacian filter and Scharr filter |
| 6 | Median Denoising | Salt-and-pepper noise removal |
| 7 | Unsharp Masking | Image sharpening via Gaussian blur subtraction |
| 8 | LAB Color Space | Luminance-Chrominance decomposition |

### Files

- `Week_2/Task-1.py` — Complete 8-stage processing script
- `Week_2/Assignment 2 sol/` — PDF task solutions (Tasks 1 to 4)

**Libraries Used:** Python, OpenCV, NumPy, Matplotlib

---

## Week 3 — Feature-Based Image Similarity Search

A two-script system implementing content-based image retrieval (CBIR) using deep feature embeddings. ResNet18 (pretrained on ImageNet) serves as the feature extractor, and the Annoy library builds an approximate nearest-neighbour index for fast similarity search at scale.

### System Overview

```
[Dog Images] --> ResNet18 (fc = Identity) --> 512-dim Embeddings --> Annoy Index
                                                                          |
[Query Image] --> ResNet18 --> Query Embedding --> Top-24 Nearest --------+
                                                        |
                                             5x5 Grid of Similar Images
```

### Scripts

**SASIS.py — Search and Similarity Indexing System**
- Iterates over all dog images in the training set
- Extracts 512-dimensional embeddings from ResNet18 (final fully connected layer replaced with `nn.Identity()`)
- Adds each embedding to an Annoy Index (angular distance metric, 10 trees)
- Saves the built index as `dog_index.ann`

**FSI.py — Feature Similarity Index (Retrieval)**
- Loads the pre-built Annoy index
- For each query image, retrieves 24 nearest neighbours
- Renders a 5x5 image grid (query highlighted with a red border, plus 24 similar images)
- Saves each grid to `ImageDump/image_{i}.png`

### Files

- `Week_3/SASIS.py` — Index builder
- `Week_3/FSI.py` — Similarity retrieval and grid generator
- `Week_3/3Q1.pdf` / `Week_3/3Q2.pdf` — Theory Q&A documents

**Libraries Used:** Python, PyTorch, Torchvision (ResNet18), Annoy, PIL

---

## Mid-Evaluation — Face Mask Detection

A complete computer vision pipeline for multi-class face mask classification, trained on a dataset with XML bounding box annotations. The project covers the full machine learning workflow: data preprocessing, model design, training, evaluation, and export.

### Problem Statement

Classify facial images into three categories:

- `with_mask` — Mask worn correctly
- `without_mask` — No mask worn
- `mask_weared_incorrect` — Mask worn incorrectly

### Pipeline

```
XML Annotations --> Face Cropping (OpenCV) --> Resize and Normalize
                                                      |
                                             Custom CNN Model
                                            (Conv2D > MaxPool > Dense)
                                                      |
                                         Multi-class Softmax Output
                                                      |
                                         model.save('mask_classifier_model.h5')
```

### CNN Architecture

| Layer | Details |
|-------|---------|
| Conv2D | Feature extraction with ReLU activations |
| MaxPooling2D | Spatial downsampling |
| Flatten | Conversion to feature vector |
| Dense | Fully connected classification layers |
| Output | Softmax activation (3 classes) |

### Files

- `MidEval/Face_mask_detection/project.ipynb` — Full training and evaluation notebook
- `MidEval/Face_mask_detection/test1.jpg` — Sample inference image

**Libraries Used:** Python, TensorFlow, Keras, OpenCV, Scikit-learn

---

## End-Evaluation — Generative AI and Neural Style Transfer (Group Project)

The culminating group project implementing state-of-the-art Generative AI architectures. It combines two distinct generative AI paradigms.

### Project 1A — CycleGAN (Monet to Photo Translation)

CycleGAN enables unpaired image-to-image translation, learning to convert photographs into Monet-style paintings and back without requiring paired training examples.

**Architecture:**

```
Photo --> Generator G --> Fake Monet --> Generator F --> Reconstructed Photo
                              |
                        Discriminator D_Y
                        (Real vs. Fake Monet)
```

- **Cycle Consistency Loss** ensures `F(G(x)) = x` (and vice versa)
- **Adversarial Loss** (GAN loss) trains the generators to fool the discriminators
- Trained on the `monet2photo` dataset

### Project 1B — Neural Style Transfer (Picasso styled Mona Lisa)

A custom PyTorch implementation using a frozen VGG19 backbone to transfer the artistic style of Pablo Picasso's *The Weeping Woman* onto Leonardo da Vinci's *Mona Lisa*.

**Optimization:** Gradient descent on the output image to minimize the combined loss:

```
L_total = (alpha x L_content) + (beta x L_style)
```

- **Content Loss:** Mean squared error between feature maps of the content image and the generated image (deep VGG19 layers)
- **Style Loss:** Mean squared error between Gram matrices of the style image and the generated image (shallow VGG19 layers)

### Files

| File | Description |
|------|-------------|
| `EndEval/EndEval_code.ipynb` | Complete notebook: CycleGAN and Neural Style Transfer |
| `EndEval/EndEval_Project1_doc.pdf` | Full project documentation |
| `EndEval/presentation_endeval_project1_ganforge.pdf` | Project presentation slides |
| `EndEval/requirements.txt` | All environment dependencies |

**Libraries Used:** Python, PyTorch, Torchvision (VGG19), PIL, Matplotlib, NumPy, Visdom

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A CUDA-capable GPU is recommended for the End-Evaluation training tasks
- Jupyter Lab, VS Code, or Kaggle Notebooks

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ramcharan601/GanForge.git
cd GanForge

# 2. Install core dependencies
pip install torch torchvision opencv-python matplotlib scikit-learn annoy pillow tensorflow

# 3. For the End-Evaluation project
pip install -r EndEval/requirements.txt
```

### Running the Projects

| Project | Entry Point |
|---------|-------------|
| Week 1 | Open `Week_1/assignment-1.ipynb` in Jupyter |
| Week 2 | `python Week_2/Task-1.py` |
| Week 3 (build index) | `python Week_3/SASIS.py` — run this first |
| Week 3 (retrieval) | `python Week_3/FSI.py` — run after SASIS |
| Mid-Evaluation | Open `MidEval/Face_mask_detection/project.ipynb` in Jupyter |
| End-Evaluation | Open `EndEval/EndEval_code.ipynb` in Jupyter or Kaggle |

**Note for Week 3:** Update the `images_folder` path in both `SASIS.py` and `FSI.py` to point to your local dog image dataset directory before running.

---

## Tech Stack Summary

| Category | Technologies |
|----------|-------------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch, TensorFlow, Keras |
| Computer Vision | OpenCV, Torchvision, PIL |
| ML Utilities | Scikit-learn, NumPy |
| Similarity Search | Annoy (Approximate Nearest Neighbors) |
| Visualization | Matplotlib, Visdom |
| Pretrained Models | ResNet18, VGG19 |
| Generative Models | Custom CNN, CycleGAN |
| Notebooks | Jupyter Notebook / Lab |

---

## Learning Progression

```
Week 1       Week 2            Week 3               MidEval          EndEval
  |             |                 |                    |                |
Image I/O --> Classical CV --> Deep Feature Search --> CNN Classifier --> GANs and Style Transfer
(OpenCV)    (Filters, Edge   (ResNet18 + Annoy CBIR) (TensorFlow/    (CycleGAN + VGG19
            Detection, HSV)                           Keras)           PyTorch)
```

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for full terms and conditions.

---

*Created by Ram Charan (Roll No: 241212) — IIT Kanpur ACA Summer Project*
