# For testing accuracy over different number of samples
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import gc
from models import MRI_model, PET_model
from myconfig import trained_model_path, torch_tensors_path

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
                # mri_uncertainty = mri_logits_stack.var(axis = 0).mean(axis = 1).cpu().numpy()
                # pet_uncertainty = pet_logits_stack.var(axis = 0).mean(axis = 1).cpu().numpy()
                mri_uncertainty = mri_softmax_stack.var(axis = 0).mean(axis = 1).cpu().numpy()
                pet_uncertainty = pet_softmax_stack.var(axis = 0).mean(axis = 1).cpu().numpy()

            mri_predictions.extend(mri_prediction)
            pet_predictions.extend(pet_prediction)
            mri_uncertainties.extend(mri_uncertainty)
            pet_uncertainties.extend(pet_uncertainty)
            mri_probs.extend(mri_softmax_mean.cpu().numpy())
            pet_probs.extend(pet_softmax_mean.cpu().numpy())

        del images_mri, images_pet, labels, mri_outputs, pet_outputs
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return (mri_predictions, pet_predictions, 
            mri_uncertainties, pet_uncertainties,
            mri_probs, pet_probs)

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
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

class_names = ['AD', 'CN', 'MCI']

def make_predictions(num_samples):

    print('###############################')

    mri_model.eval()
    pet_model.eval()

    (all_mri_predictions, all_pet_predictions, 
    all_mri_uncertainties, all_pet_uncertainties,
    all_mri_probs, all_pet_probs) = test_loop(mri_model, pet_model, test_loader, num_samples)

    # Final predictions
    final_predictions = []

    for i in range(len(all_mri_predictions)):
        if all_mri_uncertainties[i] < all_pet_uncertainties[i]:
            final_predictions.append(all_mri_predictions[i])
        else:
            final_predictions.append(all_pet_predictions[i])

    accuracy = accuracy_score(y_test, final_predictions)

    return accuracy

plt.figure(figsize = (20, 10))

num_samples = range(100, 2100, 100)

accuracies = []

for num_sample in num_samples:
    accuracy = make_predictions(num_sample)
    accuracies.append(accuracy)

plt.plot(num_samples, accuracies, marker = 'o')
plt.xlabel('Number of samples', fontsize = 28)
plt.ylabel('Accuracy', fontsize = 28)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.title('Accuracy vs Number of Samples', fontsize = 32)

plt.savefig('graphs/accuracy_vs_samples.jpg')
plt.close()

