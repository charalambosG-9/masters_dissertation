from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
from sklearn.metrics import roc_curve, auc

data_dir = '/mnt/parscratch/users/acp23cg/'

# Create directories
if not os.path.exists('graphs/'):
    os.makedirs('graphs/')

if not os.path.exists('conf_matrices/'):
    os.makedirs('conf_matrices/')

trained_model_path = os.path.join(data_dir, 'models')

if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)

torch_tensors_path = os.path.join(data_dir, 'torch_tensors')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, filename = 'Confusion Matrix'):

    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

    fig, ax = plt.subplots(figsize = (20, 20))
    disp.plot(ax=ax, cmap='Blues', colorbar = False)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 60) 
    ax.set_xlabel('Predicted Label', fontsize = 60) 
    ax.set_ylabel('True Label', fontsize = 60)       

    for texts in ax.texts:
        texts.set_fontsize(84)  

    plt.savefig('conf_matrices/' + filename + '.jpg')
    plt.close()

# Print metrics
def print_metrics(labels, predictions, conf_matrix = False, classes = ['AD', 'CN', 'MCI'], filename = None):

    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average = 'macro')
    precision = precision_score(labels, predictions, average = 'macro')
    f1 = f1_score(labels, predictions, average = 'macro')
    cm = confusion_matrix(labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if conf_matrix:
        print(f"Confusion Matrix:\n{cm}\n")
        if filename is not None:
            plot_confusion_matrix(cm, classes, filename)


# Function to plot uncertainty distribution
def plot_uncertainty_distribution(mri_uncertainties, pet_uncertainties, final_uncertainties, file_name):

    plt.figure(figsize = (40, 12))

    # Plot MRI uncertainties
    plt.subplot(1, 3, 1)
    plt.hist(mri_uncertainties, bins = 50, alpha=  0.75)
    plt.title('MRI Model Prediction Uncertainty', fontsize = 40)
    plt.xlabel('Uncertainty', fontsize = 35)
    plt.ylabel('Frequency', fontsize = 35)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    # Plot PET uncertainties
    plt.subplot(1, 3, 2)
    plt.hist(pet_uncertainties, bins = 50, alpha = 0.75)
    plt.title('PET Model Prediction Uncertainty', fontsize = 40)
    plt.xlabel('Uncertainty', fontsize = 35)
    plt.ylabel('Frequency', fontsize = 35)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    # Plot final uncertainties
    plt.subplot(1, 3, 3)
    plt.hist(final_uncertainties, bins = 50, alpha = 0.75)
    plt.title('Final Prediction Uncertainty', fontsize = 40)
    plt.xlabel('Uncertainty', fontsize = 35)
    plt.ylabel('Frequency', fontsize = 35)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    plt.tight_layout()
    plt.savefig(file_name)
    # plt.show()
    plt.close()

# Fuction to plot ROC AUC
def plot_roc_auc(labels_one_hot, mri_probs, pet_probs, final_probs, num_classes, class_names, file_name):
    plt.figure(figsize = (40, 12))

    for i in range(num_classes):
        # Calculate ROC curve and ROC area for MRI model
        fpr_mri, tpr_mri, _ = roc_curve(labels_one_hot[:, i], mri_probs[:, i])
        roc_auc_mri = auc(fpr_mri, tpr_mri)
        
        # Plot ROC curve for MRI model
        plt.subplot(1, 3, 1)
        plt.plot(fpr_mri, tpr_mri, lw = 2, label = f'Class {class_names[i]} (area = {roc_auc_mri:.2f})')

    plt.subplot(1, 3, 1)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlabel('False Positive Rate', fontsize = 35)
    plt.ylabel('True Positive Rate', fontsize = 35)
    plt.title('MRI Model ROC', fontsize = 40)
    plt.legend(loc="lower right", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    for i in range(num_classes):
        # Calculate ROC curve and ROC area for PET model
        fpr_pet, tpr_pet, _ = roc_curve(labels_one_hot[:, i], pet_probs[:, i])
        roc_auc_pet = auc(fpr_pet, tpr_pet)
        
        # Plot ROC curve for PET model
        plt.subplot(1, 3, 2)
        plt.plot(fpr_pet, tpr_pet, lw = 2, label = f'Class {class_names[i]} (area = {roc_auc_pet:.2f})')

    plt.subplot(1, 3, 2)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlabel('False Positive Rate', fontsize = 35)
    plt.ylabel('True Positive Rate', fontsize = 35)
    plt.title('PET Model ROC', fontsize = 40)
    plt.legend(loc="lower right", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    for i in range(num_classes):
        # Calculate ROC curve and ROC area for final model
        fpr_final, tpr_final, _ = roc_curve(labels_one_hot[:, i], final_probs[:, i])
        roc_auc_final = auc(fpr_final, tpr_final)
        
        # Plot ROC curve for final model
        plt.subplot(1, 3, 3)
        plt.plot(fpr_final, tpr_final, lw = 2, label = f'Class {class_names[i]} (area = {roc_auc_final:.2f})')

    plt.subplot(1, 3, 3)
    plt.plot([0, 1], [0, 1], color = 'navy', lw=  2, linestyle = '--')
    plt.xlabel('False Positive Rate', fontsize = 35)
    plt.ylabel('True Positive Rate', fontsize = 35)
    plt.title('Final Model ROC', fontsize = 40)
    plt.legend(loc = "lower right", fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()