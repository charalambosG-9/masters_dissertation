import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import  train_test_split
from collections import Counter
import skimage.transform as skTrans
import scipy.ndimage as ndi
import torch
from myconfig import data_dir
from augmentations import data_augmentation

# Resize images
def resize_image(img, target_shape, background_threshold = 0.01):
    
    background_mask = img < background_threshold
    
    factors = [t / s for t, s in zip(target_shape, img.shape)]
    
    img_resized = ndi.zoom(img, factors, order = 3)
    
    mask_resized = ndi.zoom(background_mask, factors, order = 0)
    
    img_resized[mask_resized] = 0
    
    return img_resized

# Percentile normalization
def percentile_normalization(data, lower_percentile = 1, upper_percentile = 99):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    normalized_data = np.clip(data, lower_bound, upper_bound)
    normalized_data = (normalized_data - lower_bound) / (upper_bound - lower_bound)
    return normalized_data

# Get image data from files
def get_data_from_files(labels, file_dir, target_shape):
    
    all_data = []
    all_labels = []
    images = []

    for i, label in enumerate(labels):
        files = [f for f in os.listdir(os.path.join(file_dir, label)) if f.endswith('.nii')]
        files.sort()
        for file in files:
            img = nib.load(os.path.join(file_dir, label, file))
            img_data = img.get_fdata()
            resized_data =  skTrans.resize(img_data, target_shape, order = 3, preserve_range = True)
            all_data.append(resized_data)
            all_labels.append(i)
            images.append(img)

    # Convert to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    images = np.array(images)

    return all_data, all_labels, images

labels = ['AD', 'CN', 'MCI']

mri_data_dir = os.path.join(data_dir, 'Data/MPR__GradWarp')

# mri_target_shape = (256, 256, 166)
mri_target_shape = (128, 128, 60)

all_data_mri, all_labels_mri, images_mri = get_data_from_files(labels, mri_data_dir, mri_target_shape)

print('\nMRI Data shape:', all_data_mri.shape)
print('\nMRI Labels shape:', all_labels_mri.shape)

pet_data_dir = os.path.join(data_dir, 'Data/Co-registered,_Averaged')

pet_target_shape = (128, 128, 60)

all_data_pet, all_labels_pet, images_pet = get_data_from_files(labels, pet_data_dir, pet_target_shape)

print('\nPET Data shape:', all_data_pet.shape)
print('\nPET Labels shape:', all_labels_pet.shape)

if np.array_equal(all_labels_mri, all_labels_pet):
    print("\nLabels are the same for both modalities")
else:
    print("\nLabels are not the same for both modalities")
    exit()

x_train_mri, x_test_mri, x_train_pet, x_test_pet, y_train, y_test = train_test_split(all_data_mri, all_data_pet, all_labels_mri, 
                                                                                     test_size = 0.2, random_state = 42, stratify = all_labels_mri)

print('\nTraining data labels MRI:', Counter(y_train))
print('\nTesting data labels MRI:', Counter(y_test))

temp_label_array = y_train

x_train_mri, y_train = data_augmentation(x_train_mri, y_train, num_per_sample = 10, target_shape = mri_target_shape, crop_size = (112, 112, 48)) # 224, 224, 160

print('\nTraining data shape after augmentation MRI:', x_train_mri.shape)

x_train_pet, _ = data_augmentation(x_train_pet, temp_label_array, num_per_sample = 10, target_shape = pet_target_shape, crop_size = (112, 112, 48))

print('\nTraining data shape after augmentation PET:', x_train_pet.shape)

# Percentage normalization
x_train_mri = percentile_normalization(x_train_mri)
x_test_mri = percentile_normalization(x_test_mri)

x_train_pet = percentile_normalization(x_train_pet)
x_test_pet = percentile_normalization(x_test_pet)

# Permute data
x_train_mri = np.transpose(x_train_mri, (0, 3, 1, 2))
x_test_mri = np.transpose(x_test_mri, (0, 3, 1, 2))

x_train_pet = np.transpose(x_train_pet, (0, 3, 1, 2))
x_test_pet = np.transpose(x_test_pet, (0, 3, 1, 2))

# Add channel dimension
x_train_mri = x_train_mri[:, np.newaxis, :, :, :]
x_test_mri = x_test_mri[:, np.newaxis, :, :, :]

x_train_pet = x_train_pet[:, np.newaxis, :, :, :]
x_test_pet = x_test_pet[:, np.newaxis, :, :, :]

print('\nTraining data shape after permutating and adding channel dimension MRI:', x_train_mri.shape)
print('\nTest data shape after permutating and adding channel dimension MRI:', x_test_mri.shape)

print('\nTraining data shape after permutating and adding channel dimension PET:', x_train_pet.shape)
print('\nTest data shape after permutating and adding channel dimension PET:', x_test_pet.shape)

# Convert data to PyTorch tensors
x_train_mri = torch.from_numpy(x_train_mri).float()
x_test_mri = torch.from_numpy(x_test_mri).float()

x_train_pet = torch.from_numpy(x_train_pet).float()
x_test_pet = torch.from_numpy(x_test_pet).float()

y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

target_path = os.path.join(data_dir, 'torch_tensors')

if not os.path.exists(target_path):
    os.makedirs(target_path)

torch.save(x_train_mri, os.path.join(target_path, 'x_train_mri.pt'))
torch.save(x_test_mri, os.path.join(target_path, 'x_test_mri.pt'))

torch.save(x_train_pet, os.path.join(target_path, 'x_train_pet.pt'))
torch.save(x_test_pet, os.path.join(target_path, 'x_test_pet.pt'))

torch.save(y_train, os.path.join(target_path, 'y_train.pt'))
torch.save(y_test, os.path.join(target_path, 'y_test.pt'))

