# GanForge
**Indian Institute of Technology Kanpur (IITK) | ACA**

Welcome to **GanForge**, a comprehensive repository documenting my journey through the **IIT Kanpur ACA Summer Project** on Machine Learning, Deep Learning, and Generative AI. 

This repository serves as a structured log of coursework, coding assignments, and hands-on projects designed to build foundational and advanced skills in artificial intelligence. Throughout this program, the focus transitions from basic data manipulation and mathematical modeling to implementing robust deep learning architectures (like CNNs) using PyTorch, and exploring the mechanics of Generative Adversarial Networks (GANs).

**Key Learning Objectives Covered:**
- **Foundational ML:** Data preprocessing, statistical modeling, and basic machine learning algorithms.
- **Deep Learning Architectures:** Building and training neural networks from scratch using PyTorch.
- **Computer Vision:** Image processing, feature extraction, and implementing CNNs for real-world tasks like face mask detection.
- **Generative AI Concepts:** Understanding the underlying theory and application of modern generative models.

## Repository Structure

This repository is organized progressively by weekly assignments, culminating in a comprehensive End-Evaluation project.

- **`Week_2/`**: Advanced scripting and image processing tasks using OpenCV — HSV conversion, histogram equalization, edge detection, denoising, and LAB color space.
- **`Week_3/`**: Implementation of algorithms including Feature Similarity Index (FSI) for image retrieval and SASIS (Search and Similarity Indexing System) using ResNet18 and Annoy index.
- **`EndEval/`**: The final group evaluation focusing on Generative AI, featuring CycleGAN (Monet-to-Photo) and Neural Style Transfer (Picasso-styled Mona Lisa).

---

## Week 2: Image Processing with OpenCV

**Description:**
Advanced image processing pipeline using OpenCV and Matplotlib.

**Tasks Covered:**
- HSV color space conversion and channel splitting
- Histogram equalization on grayscale images
- Binary thresholding (inverse)
- Gray level reduction (posterization)
- Edge detection using Laplacian and Scharr filters
- Median denoising
- Unsharp masking (image sharpening)
- LAB color space conversion

**Files:**
- `Task-1.py`: Complete image processing pipeline
- `Assignment 2 sol/`: PDF solutions for Tasks 1–4

---

## Week 3: Feature-based Image Similarity Search

**Description:**
Implementation of a two-stage image similarity retrieval system using deep learning feature embeddings.

**Key Components:**
- **SASIS.py**: Builds an Annoy index from ResNet18 (pretrained on ImageNet) feature embeddings of dog images
- **FSI.py**: Queries the Annoy index for each image and saves a 5×5 grid of nearest-neighbour images

**Tech Stack:** Python, PyTorch, Torchvision, Annoy, PIL

---

## 🎨 End-Evaluation: Generative AI & Style Transfer

**Description:**
The culminating group project focusing on advanced Generative AI architectures. It implements CycleGAN for image-to-image translation (converting photos to Monet-style paintings) and leverages a pre-trained VGG19 model for Neural Style Transfer.

**Key Features:**
- **CycleGAN**: Bidirectional image translation trained on the `monet2photo` dataset.
- **Neural Style Transfer**: A custom PyTorch implementation applying the artistic style of Pablo Picasso's *The Weeping Woman* to Leonardo da Vinci's *Mona Lisa*.
- **Tech Stack**: Python, PyTorch, Torchvision, PIL, Matplotlib.

**Files Included:**
- `EndEval_code.ipynb`: The main notebook containing both the CycleGAN and Neural Style Transfer implementations.
- `EndEval_Project1_doc.pdf`: Detailed project documentation.
- `presentation_endeval_project1_ganforge.pdf`: Project presentation slides.
- `requirements.txt`: Environment dependencies.

---

## Getting Started

To explore the projects or run the code locally:

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd GanForge
   ```
2. Install the required dependencies (ensure you have PyTorch installed):
   ```bash
   pip install torch torchvision opencv-python matplotlib scikit-learn annoy pillow
   ```
3. Open the Jupyter Notebooks using your preferred environment (Jupyter Lab, VS Code, Kaggle, etc.).

---
*Created by Student (Roll No: 241212)*
