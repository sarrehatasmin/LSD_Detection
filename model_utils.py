import os
import shutil
import torch
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, roc_auc_score, roc_curve


def combine_datasets(original_dir, synthetic_dir, output_dir):
    """
    Combine original and synthetic datasets by moving images to a common folder.
    Arguments:
        original_dir (str): Path to the original dataset (Normal Skin, Lumpy Skin).
        synthetic_dir (str): Path to the synthetic images (Normal Skin, Lumpy Skin).
        output_dir (str): Path to the output directory for combined dataset.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Normal Skin'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Lumpy Skin'), exist_ok=True)
    
    original_normal_dir = os.path.join(original_dir, 'Normal Skin')
    original_lumpy_dir = os.path.join(original_dir, 'Lumpy Skin')
    
    
    synthetic_normal_dir = os.path.join(synthetic_dir, 'healthy_skin')  
    synthetic_lumpy_dir = os.path.join(synthetic_dir, 'lumpy_skin')     
    
    
    def is_image(file_name):
        return file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    if os.path.exists(original_normal_dir):
        for img_name in os.listdir(original_normal_dir):
            if is_image(img_name):  
                shutil.copy(os.path.join(original_normal_dir, img_name), 
                            os.path.join(output_dir, 'Normal Skin', img_name))

    if os.path.exists(synthetic_normal_dir):
        for img_name in os.listdir(synthetic_normal_dir):
            if is_image(img_name):  
                shutil.copy(os.path.join(synthetic_normal_dir, img_name), 
                            os.path.join(output_dir, 'Normal Skin', img_name))

    
    if os.path.exists(original_lumpy_dir):
        for img_name in os.listdir(original_lumpy_dir):
            if is_image(img_name):  
                shutil.copy(os.path.join(original_lumpy_dir, img_name), 
                            os.path.join(output_dir, 'Lumpy Skin', img_name))

    if os.path.exists(synthetic_lumpy_dir):
        for img_name in os.listdir(synthetic_lumpy_dir):
            if is_image(img_name):  
                shutil.copy(os.path.join(synthetic_lumpy_dir, img_name), 
                            os.path.join(output_dir, 'Lumpy Skin', img_name))

    print(f"Combined dataset saved to {output_dir}")

original_dir = '/workspace/archive/Lumpy Skin Images Dataset' 
synthetic_dir = '/workspace/archive/synthetic_images'
output_dir = '/workspace/archive/combined_dataset'  

combine_datasets(original_dir, synthetic_dir, output_dir)

def split_dataset(input_dir, output_dir, train_size=0.7, valid_size=0.15, test_size=0.15):
    """
    Split the dataset into train, validation, and test sets.
    Arguments:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the output directory where the splits will be saved.
        train_size (float): Proportion of the dataset to include in the train split.
        valid_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
    """

    os.makedirs(os.path.join(output_dir, 'train', 'Normal Skin'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'Lumpy Skin'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid', 'Normal Skin'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid', 'Lumpy Skin'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'Normal Skin'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'Lumpy Skin'), exist_ok=True)

    def split_and_copy(src_dir, dest_dir):
        image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        

        train_files, temp_files = train_test_split(image_files, train_size=train_size, random_state=42)
        valid_files, test_files = train_test_split(temp_files, test_size=test_size / (valid_size + test_size), random_state=42)

        
        for file_list, split in zip([train_files, valid_files, test_files], ['train', 'valid', 'test']):
            for file_name in file_list:
                shutil.copy(os.path.join(src_dir, file_name), os.path.join(dest_dir, split, os.path.basename(src_dir), file_name))


    normal_skin_dir = os.path.join(input_dir, 'Normal Skin')
    lumpy_skin_dir = os.path.join(input_dir, 'Lumpy Skin')

    
    split_and_copy(normal_skin_dir, output_dir)

    
    split_and_copy(lumpy_skin_dir, output_dir)

    print(f"Dataset split and saved to {output_dir}")

input_dir = '/workspace/archive/combined_dataset'  
output_dir = '/workspace/archive/split_dataset'  

split_dataset(input_dir, output_dir)

def count_images_in_directory(directory):
    """Count the number of images in a directory."""
    return len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))])

def copy_images(src_dir, dst_dir):
    """Copy images from source directory to destination directory."""
    for img in os.listdir(src_dir):
        if img.endswith(('.jpg', '.png', '.jpeg')):
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(dst_dir, img)
            
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

original_normal_skin_dir = '/workspace/archive/Lumpy Skin Images Dataset/Normal Skin'
original_lumpy_skin_dir = '/workspace/archive/Lumpy Skin Images Dataset/Lumpy Skin'
augmented_normal_skin_dir = '/workspace/archive/combined_dataset/Normal Skin'
augmented_lumpy_skin_dir = '/workspace/archive/combined_dataset/Lumpy Skin'

combined_normal_skin_dir = '/workspace/archive/combined_dataset/Normal Skin'
combined_lumpy_skin_dir = '/workspace/archive/combined_dataset/Lumpy Skin'

os.makedirs(combined_normal_skin_dir, exist_ok=True)
os.makedirs(combined_lumpy_skin_dir, exist_ok=True)


original_normal_skin_count = count_images_in_directory(original_normal_skin_dir)
original_lumpy_skin_count = count_images_in_directory(original_lumpy_skin_dir)


copy_images(original_normal_skin_dir, combined_normal_skin_dir)
copy_images(original_lumpy_skin_dir, combined_lumpy_skin_dir)


augmented_normal_skin_count = count_images_in_directory(augmented_normal_skin_dir)
augmented_lumpy_skin_count = count_images_in_directory(augmented_lumpy_skin_dir)


copy_images(augmented_normal_skin_dir, combined_normal_skin_dir)
copy_images(augmented_lumpy_skin_dir, combined_lumpy_skin_dir)

combined_normal_skin_count = count_images_in_directory(combined_normal_skin_dir)
combined_lumpy_skin_count = count_images_in_directory(combined_lumpy_skin_dir)

print(f"Original Normal Skin images: {original_normal_skin_count}")
print(f"Original Lumpy Skin images: {original_lumpy_skin_count}")
print(f"Total original images: {original_normal_skin_count + original_lumpy_skin_count}")

print(f"\nCombined Normal Skin images: {combined_normal_skin_count}")
print(f"Combined Lumpy Skin images: {combined_lumpy_skin_count}")
print(f"Total combined images: {combined_normal_skin_count + combined_lumpy_skin_count}")

def evaluate_model(model, data_loader, dataset_name, device):
    model.eval()
    true_labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return true_labels, predictions, probabilities

def plot_confusion_matrices(true_labels_1, preds_1, name_1, 
                            true_labels_2, preds_2, name_2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_plot_label = ['Normal skin', 'Lumpy skin']

    for ax, (true_labels, preds, dataset_name) in zip(axes, 
        [(true_labels_1, preds_1, name_1), (true_labels_2, preds_2, name_2)]):
        
        cm = confusion_matrix(true_labels, preds)
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_plot_label, yticklabels=cm_plot_label, ax=ax)
        ax.set_title(f"Confusion Matrix - {dataset_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    
    plt.tight_layout()
    plt.show()

def print_metrics(true_labels_1, preds_1, name_1, 
                  true_labels_2, preds_2, name_2):
    for true_labels, preds, dataset_name in [(true_labels_1, preds_1, name_1), 
                                             (true_labels_2, preds_2, name_2)]:
        mse = mean_squared_error(true_labels, preds)
        error_rate = 1 - (np.sum(np.array(true_labels) == np.array(preds)) / len(true_labels))
        print(f"Metrics for {dataset_name} Dataset:")
        print(classification_report(true_labels, preds, zero_division=0, target_names=['Normal skin', 'Lumpy skin']))
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Error Rate: {error_rate:.4%}\n")

def plot_auc_roc(true_labels_1, probs_1, name_1, 
                 true_labels_2, probs_2, name_2):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (true_labels, probs, dataset_name) in zip(axes, 
        [(true_labels_1, probs_1, name_1), (true_labels_2, probs_2, name_2)]):
        
        true_labels_one_hot = np.zeros((len(true_labels), len(set(true_labels))))
        for i, label in enumerate(true_labels):
            true_labels_one_hot[i, label] = 1

        probabilities = np.array(probs)
        auc_scores = []

        for i, class_label in enumerate(sorted(set(true_labels))):
            fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], probabilities[:, i])
            auc = roc_auc_score(true_labels_one_hot[:, i], probabilities[:, i])
            auc_scores.append(auc)
            ax.plot(fpr, tpr, label=f"Class {class_label} (AUC = {auc:.2f})")

        ax.set_title(f"AUC-ROC Curve - {dataset_name}")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend(loc="lower right")
        ax.grid()
    
    plt.tight_layout()
    plt.show()

