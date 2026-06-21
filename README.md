# GanForge
**Indian Institute of Technology Kanpur (IITK) | ACA**

Welcome to **GanForge**, a comprehensive repository documenting my journey through the **IIT Kanpur ACA Summer Project** on Machine Learning, Deep Learning, and Generative AI.

This repository serves as a structured log of coursework, coding assignments, and hands-on projects designed to build foundational and advanced skills in artificial intelligence. Throughout this program, the focus transitions from basic data manipulation and image processing to implementing deep learning architectures using PyTorch and TensorFlow, and exploring the mechanics of Generative Adversarial Networks (GANs).

**Key Learning Objectives Covered:**
- **Foundational CV:** Image I/O, pixel intensity analysis, and color space manipulation using OpenCV and Matplotlib.
- **Classical Image Processing:** HSV conversion, histogram equalization, edge detection, denoising, and unsharp masking.
- **Deep Feature Search:** Content-based image retrieval using ResNet18 embeddings and Annoy approximate nearest-neighbour indexing.
- **Computer Vision:** Implementing a custom CNN for real-world multi-class classification — Face Mask Detection.
- **Generative AI Concepts:** Understanding and applying CycleGAN for unpaired image translation and Neural Style Transfer using VGG19.

## Repository Structure

This repository is organized progressively by weekly assignments, culminating in a comprehensive End-Evaluation project.

- **`Week_1/`**: Introduction to OpenCV and Matplotlib — reading images, exploring dimensions, and plotting pixel intensity histograms.
- **`Week_2/`**: Advanced image processing pipeline — HSV conversion, histogram equalization, edge detection, denoising, unsharp masking, and LAB color space.
- **`Week_3/`**: Implementation of a two-stage image similarity retrieval system (SASIS + FSI) using ResNet18 feature embeddings and Annoy indexing.
- **`MidEval/Face_mask_detection/`**: The core highlight of this repository. A Convolutional Neural Network (CNN) built from scratch using TensorFlow/Keras for Face Mask Detection.
- **`EndEval/`**: The final group evaluation focusing on Generative AI, featuring CycleGAN (Monet-to-Photo) and Neural Style Transfer (Picasso-styled Mona Lisa).

---

## Highlight: Face Mask Detection (Mid-Evaluation)

**Description:** A Deep Learning Computer Vision project designed to detect whether individuals in an image are wearing a face mask correctly, incorrectly, or not at all. The model is built using TensorFlow/Keras and leverages a custom Convolutional Neural Network (CNN) architecture trained on annotated face image data.

**Key Features:**
- **Multi-class Classification:** Three output categories — `with_mask`, `without_mask`, and `mask_weared_incorrect`.
- **Dataset:** Trained on the Face Mask Detection dataset with PASCAL VOC XML bounding box annotations; face regions are cropped using OpenCV before classification.
- **Architecture:** Custom CNN with Conv2D, MaxPooling2D, Flatten, and Dense layers with Softmax output.
- **Tech Stack:** Python, TensorFlow, Keras, OpenCV, Scikit-learn.

**Files Included:**
- `MidEval/Face_mask_detection/project.ipynb`: The complete source code, data preprocessing, model training, and evaluation metrics.

---

## Highlight: Generative AI and Style Transfer (End-Evaluation)

**Description:** The culminating group project focusing on advanced Generative AI architectures. It implements CycleGAN for image-to-image translation (converting photos to Monet-style paintings) and leverages a pre-trained VGG19 model for Neural Style Transfer.

**Key Features:**
- **CycleGAN:** Bidirectional image translation trained on the `monet2photo` dataset. Uses cycle consistency loss and adversarial loss to learn mappings between unpaired image domains.
- **Neural Style Transfer:** A custom PyTorch implementation applying the artistic style of Pablo Picasso's *The Weeping Woman* to Leonardo da Vinci's *Mona Lisa*. Content and style losses are computed from VGG19 feature maps and Gram matrices respectively.
- **Tech Stack:** Python, PyTorch, Torchvision, PIL, Matplotlib.

**Files Included:**
- `EndEval/EndEval_code.ipynb`: The main notebook containing both the CycleGAN and Neural Style Transfer implementations.
- `EndEval/EndEval_Project1_doc.pdf`: Detailed project documentation.
- `EndEval/presentation_endeval_project1_ganforge.pdf`: Project presentation slides.
- `EndEval/requirements.txt`: Environment dependencies.

---

## Getting Started

To explore the projects or run the code locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ramcharan601/GanForge.git
   cd GanForge
   ```

2. Install the required dependencies (ensure you have PyTorch and TensorFlow installed):
   ```bash
   pip install torch torchvision opencv-python matplotlib scikit-learn annoy pillow tensorflow
   ```

3. Open the Jupyter Notebooks using your preferred environment (Jupyter Lab, VS Code, Kaggle, etc.).

---

*Created by Ram Charan (Roll No: 241212)*
