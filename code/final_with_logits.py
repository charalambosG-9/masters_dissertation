import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import gc
import time
from models import MRI_model, PET_model
from myconfig import trained_model_path, torch_tensors_path, print_metrics, plot_roc_auc, plot_uncertainty_distribution

# Test loop
def test_loop(mri_model, pet_model, test_loader, num_samples):
    mri_predictions = []
    pet_predictions = []
    mri_uncertainties = []
    pet_uncertainties = []
    mri_probs = []
    pet_probs = []

    if num_samples == 1:
        print("Testing without MC Dropout")
    else:
        print(f"Testing with MC Dropout ({num_samples} samples)")
        for module in mri_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        for module in pet_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    with torch.no_grad():
        for images_mri, images_pet, labels in test_loader:
            images_mri, images_pet, labels = images_mri.to(device), images_pet.to(device), labels.to(device)

            mri_logits_list = []
            pet_logits_list = []

            mri_softmax_list = []
            pet_softmax_list = []

            for _ in range(num_samples):
                mri_outputs = mri_model(images_mri)  
                pet_outputs = pet_model(images_pet)  

                mri_logits_list.append(mri_outputs)
                pet_logits_list.append(pet_outputs)

                mri_softmax_list.append(F.softmax(mri_outputs, dim = 1))
                pet_softmax_list.append(F.softmax(pet_outputs, dim = 1))

            mri_logits_stack = torch.stack(mri_logits_list)
            pet_logits_stack = torch.stack(pet_logits_list)

            mri_softmax_stack = torch.stack(mri_softmax_list)
            pet_softmax_stack = torch.stack(pet_softmax_list)

            mri_logits_mean = mri_logits_stack.mean(axis = 0)
            pet_logits_mean = pet_logits_stack.mean(axis = 0)

            mri_softmax_mean = mri_softmax_stack.mean(axis = 0)
            pet_softmax_mean = pet_softmax_stack.mean(axis = 0)

            mri_prediction = torch.argmax(F.softmax(mri_logits_mean, dim = 1), axis = 1).cpu().numpy()
            pet_prediction = torch.argmax(F.softmax(pet_logits_mean, dim = 1), axis = 1).cpu().numpy()

            if num_samples == 1:
                mri_uncertainty = np.zeros(len(mri_prediction))
                pet_uncertainty = np.zeros(len(pet_prediction))
            else:
                mri_uncertainty = mri_logits_stack.var(axis = 0).mean(axis = 1).cpu().numpy()
                pet_uncertainty = pet_logits_stack.var(axis = 0).mean(axis = 1).cpu().numpy()

            mri_predictions.extend(mri_prediction)
            pet_predictions.extend(pet_prediction)
            mri_uncertainties.extend(mri_uncertainty)
            pet_uncertainties.extend(pet_uncertainty)
            mri_probs.extend(F.softmax(mri_softmax_mean, dim = 1).cpu().numpy())  
            pet_probs.extend(F.softmax(pet_softmax_mean, dim = 1).cpu().numpy())

        del images_mri, images_pet, labels, mri_outputs, pet_outputs
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return (mri_predictions, pet_predictions, 
            mri_uncertainties, pet_uncertainties,
            mri_probs, pet_probs)

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
x_test_mri = torch.load(os.path.join(torch_tensors_path, 'x_test_mri.pt'))
x_test_pet = torch.load(os.path.join(torch_tensors_path, 'x_test_pet.pt'))

y_test = torch.load(os.path.join(torch_tensors_path, 'y_test.pt'))

num_classes = len(np.unique(y_test))

mri_model = MRI_model(x_test_mri.shape, num_classes).to(device)
pet_model = PET_model(x_test_pet.shape, num_classes).to(device)
    
mri_model.load_state_dict(torch.load(os.path.join(trained_model_path, 'mri_model.pt')))
pet_model.load_state_dict(torch.load(os.path.join(trained_model_path, 'pet_model.pt')))

if device == 'cuda':
    mri_model = nn.DataParallel(mri_model)
    pet_model = nn.DataParallel(pet_model)

# Test phase
test_dataset = torch.utils.data.TensorDataset(x_test_mri, x_test_pet, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class_names = ['AD', 'CN', 'MCI']

def make_predictions(num_samples, roc_curve_file_name, uncertainty_file_name):

    print('###############################')

    mri_model.eval()
    pet_model.eval()

    (mri_predictions, pet_predictions, 
    mri_uncertainties, pet_uncertainties,
    mri_probs, pet_probs) = test_loop(mri_model, pet_model, test_loader, num_samples)

    # Convert probabilities to numpy arrays
    mri_probs = np.array(mri_probs)
    pet_probs = np.array(pet_probs)

    # One-hot encode the labels
    y_test_one_hot = np.eye(num_classes)[y_test]

    print('\nMRI Model')
    print_metrics(y_test, mri_predictions, conf_matrix = True, classes = class_names, filename = 'MRI_confusion_matrix_'+str(num_samples)+'_samples_logits')

    print('\nPET Model')
    print_metrics(y_test, pet_predictions, conf_matrix = True, classes=class_names, filename='PET_confusion_matrix_'+str(num_samples)+'_samples_logits')

    print('\nMRI predictions')
    print(mri_predictions)

    print('\nPET predictions')
    print(pet_predictions)

    # Final predictions
    final_predictions = []
    final_uncertainties = []
    final_probs = []

    num_from_mri = 0
    num_from_pet = 0
    
    for i in range(len(mri_predictions)):
        if mri_uncertainties[i] < pet_uncertainties[i]:
            final_predictions.append(mri_predictions[i])
            final_uncertainties.append(mri_uncertainties[i])
            final_probs.append(mri_probs[i])
            num_from_mri += 1
        else:
            final_predictions.append(pet_predictions[i])
            final_uncertainties.append(pet_uncertainties[i])
            final_probs.append(pet_probs[i])
            num_from_pet += 1

    final_probs = np.array(final_probs)

    # Plot ROC AUC for MRI and PET models
    plot_roc_auc(y_test_one_hot, mri_probs, pet_probs, final_probs, num_classes, class_names, os.path.join('graphs', roc_curve_file_name))

    print('\nFinal predictions')
    print_metrics(y_test, final_predictions, conf_matrix = True, classes = class_names, filename = 'final_confusion_matrix_'+str(num_samples)+'_samples_logits')
    print('')
    print(final_predictions)

    print(f'\nNumber of samples where MRI model was chosen: {num_from_mri}')
    print(f'Number of samples where PET model was chosen: {num_from_pet}')

    print('\nActual labels')
    print(y_test)

    print('\nMRI uncertainties')
    print(mri_uncertainties)

    print('\nPET uncertainties')
    print(pet_uncertainties)

    print('\nFinal uncertainties')
    print(final_uncertainties)

    plot_uncertainty_distribution(mri_uncertainties, pet_uncertainties, final_uncertainties, os.path.join('graphs', uncertainty_file_name))

make_predictions(1, 'roc_auc_1_sample_logits.jpg', 'uncertainty_1_sample_logits.jpg')
make_predictions(50, 'roc_auc_50_samples_logits.jpg', 'uncertainty_50_samples_logits.jpg')
make_predictions(500, 'roc_auc_500_samples_logits.jpg', 'uncertainty_500_samples_logits.jpg')
make_predictions(1000, 'roc_auc_1000_samples_logits.jpg', 'uncertainty_1000_samples_logits.jpg')

print(f'Time taken: {time.time() - start_time}')
