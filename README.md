# Enhanced Detection of Lumpy Skin Disease Using CycleGAN-Based Data Augmentation and Deep Learning
## Overview
This project presents a deep learning-based framework for detecting Lumpy Skin Disease (LSD) in cattle. The study addresses the challenge of dataset imbalance by employing CycleGAN-based data augmentation to generate synthetic images and improve classification performance.
## Objectives
1. Address dataset imbalance using CycleGAN-based augmentation
2. Generate realistic synthetic images for minority classes
3. Train and evaluate multiple pre-trained CNN models
4. Improve classification accuracy and generalization capability
## Dataset
- Total images: 1,024
- Normal: 700
- Lumpy Skin: 324
- Image resolution: 256 × 256
## Methodology
### Data Augmentation
CycleGAN is used for unpaired image-to-image translation to generate synthetic images for both classes. This helps balance the dataset and enhances diversity.
## Classification Models
The following pre-trained convolutional neural network architectures are evaluated:
- DenseNet121
- InceptionV3
- VGG19
- MobileNetV2
- ResNet50
- Xception
## Experimental Setup
- Framework: PyTorch
- Optimizer: Adam
- Learning rate:
  - CycleGAN: 0.0002
  - Classifiers: 1e-4
- Batch size:
- CycleGAN: 8
- Classifiers: 32
- Epochs:
 - CycleGAN: 200
 - Classifiers: 30
## Results
The inclusion of CycleGAN-generated synthetic data improved the performance of all evaluated models.

Best performing model: VGG19

- Accuracy: 98.75%
- Precision: 0.98
- Recall: 0.99
- F1-score: 0.99
- Mean Squared Error: 0.01
## Key Findings
- Significant improvement using augmented dataset
- Reduced overfitting
- Enhanced generalization capability
- Synthetic data improves minority class detection

## Installation

git clone https://github.com/sarrehatasmin/LSD_Detection_CycleGAN.git
cd LSD_Detection_CycleGAN

## Usage
Open the Jupyter Notebook to run the project:
CYCLEGAN_LSD.ipynb
Run all cells step by step to:
- Train the CycleGAN model
- Generate synthetic images
- Train classification models
- Evaluate results
## Project Structure
```
LSD_Detection_CycleGAN/
├── Classification_models/
├── Results/
├── CYCLEGAN_LSD.ipynb
├── model_utils.py
└── README.md
```
## Contributions
- First application of CycleGAN for LSD veterinary disease detection
- Effective handling of class imbalance
- Demonstrated improvement across multiple CNN models
- High-performance classification with limited real data
## Future Work
- Integration with Explainable AI (Grad-CAM, SHAP)
- Deployment on mobile/edge devices
- Extension to other livestock diseases
- Integration with federated learning systems
## Citation
Khaton, S., Rikta, S. T., You, W., Uddin, K. M. M.
Enhanced Detection of Lumpy Skin Disease Using CycleGAN-Based Data Augmentation and Deep Learning Classifiers.
## License
This project is intended for research and academic purposes.
