POTATO LEAF DISEASE DETECTION AND CLASSIFICATION USING VGG16S (PYTORCH + CUDA)

This project implements a deep learning-based image classification system to detect and classify potato leaf diseases using transfer learning with a modified VGG16 model. The approach utilizes PyTorch and leverages CUDA for GPU acceleration to enhance training efficiency and model performance.

PROJECT OVERVIEW

The main objective of this project is to identify and classify potato leaf diseases into different categories using computer vision.
The model assists in early diagnosis and prevention of potato crop diseases by analyzing leaf images.

Classification Categories:

Healthy

Early Blight

Late Blight

Workflow Summary:

Data preparation with image resizing, rotation, flipping, color jitter, and normalization.

Dataset split into 80% training and 20% validation.

VGG16 used as base model with modified classifier:

Fully connected layer (512 × 7 × 7 → 256)

ReLU activation

Dropout (0.5)

Output layer (256 → number of classes)

Training in two phases:

Phase 1: Classifier training (10 epochs, LR = 0.001)

Phase 2: Fine-tuning (50 epochs, LR = 0.000001)

Evaluation metrics include accuracy, precision, recall, F1-score, AUC, specificity, MCC, and Cohen’s kappa.

External image testing supported with class probability display.

DATASET STRUCTURE

The dataset must be inside a folder named “plantvillage” with subfolders for each class:

plantvillage
	 Early_Blight
	 Healthy
 	 Late_Blight

Each folder contains the respective leaf images.

REQUIRED LIBRARIES

Python 3.8 or later
PyTorch
Torchvision
NumPy
Matplotlib
Scikit-learn
Pillow (PIL)
tqdm

A GPU with CUDA support is recommended for training.

HOW TO RUN

Place your dataset in the folder “plantvillage”.

Run the main script (for example: python potato_vgg16s.py).

The model will:

Train in two phases

Save the final model as “vgg16s_potato_model_final.pth”

Generate two charts:

“loss_curve_final.png” (training loss curve)

“final_metrics_final.png” (evaluation metrics)

You can test external images such as healthy.jpg, early.jpg, or late2.webp to get predictions and confidence scores.

OUTPUT FILES

vgg16s_potato_model_final.pth — Trained model file
loss_curve_final.png — Training and validation loss curve
final_metrics_final.png — Evaluation metrics bar chart

EVALUATION METRICS

Accuracy — Overall prediction correctness
Precision — Correct positive predictions
Recall — Correctly identified positive samples
F1-Score — Balance between precision and recall
AUC — Area under ROC curve
Specificity — Ability to detect healthy leaves
MCC — Balanced measure of prediction quality
Cohen’s Kappa — Agreement measure between prediction and truth

AUTHORS

22MID0309 - Kalai Selvan G
22MID0097 - Manjunathan V