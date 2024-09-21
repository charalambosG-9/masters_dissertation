import random
import numpy as np
import skimage.transform as skTrans
import scipy.ndimage as ndi
import scipy.ndimage as ndi
import skimage.transform as skTrans
from skimage import exposure

def augment_image(img, label, crop_size, target_shape, augmentation_probs):

    # Apply augmentations with specified probabilities
    if random.random() < augmentation_probs.get('crop', 0):
        img = random_crop(img, crop_size)
        img = skTrans.resize(img, target_shape, preserve_range=True)

    if random.random() < augmentation_probs.get('scale', 0):
        scale_factor = random.uniform(0.9, 1.1)
        img_scaled = skTrans.resize(img, (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor), int(img.shape[2] * scale_factor)), preserve_range=True)
        img = skTrans.resize(img_scaled, target_shape, preserve_range=True)

    if random.random() < augmentation_probs.get('flip', 0):
        flip_axis = random.choice([0, 1, 2])
        img = np.flip(img, axis=flip_axis)

    if random.random() < augmentation_probs.get('rotate', 0):
        angle = random.uniform(-10, 10)
        axes = random.sample([0, 1, 2], 2)
        img = ndi.rotate(img, angle, axes=axes, reshape=False, mode='nearest')

    if random.random() < augmentation_probs.get('blur', 0):
        sigma = random.uniform(0.5, 1.5)
        img = ndi.gaussian_filter(img, sigma=sigma)

    if random.random() < augmentation_probs.get('elastic', 0):
        alpha = random.uniform(0.4, 1) 
        sigma = random.uniform(0.02, 0.05)  
        img = elastic_transform(img, alpha, sigma)

    if random.random() < augmentation_probs.get('brightness', 0):
        img = np.clip(img, 0, None)
        img = exposure.adjust_gamma(img, gamma=random.uniform(0.95, 1.05))

    if random.random() < augmentation_probs.get('contrast', 0):
        img = np.clip(img, 0, None)
        img = exposure.equalize_hist(img)

    if random.random() < augmentation_probs.get('random_erasing', 0):
        img = random_erasing(img)

    if random.random() < augmentation_probs.get('dropout', 0):
        img = dropout(img)

    return img, label

def affine_transform(image, matrix):
    return ndi.affine_transform(image, matrix[:3, :3], offset=matrix[:3, 3], output_shape=image.shape, order=1, mode='nearest')

def random_crop(image, crop_size):
    start_x = random.randint(0, image.shape[0] - crop_size[0])
    start_y = random.randint(0, image.shape[1] - crop_size[1])
    start_z = random.randint(0, image.shape[2] - crop_size[2])
    return image[start_x:start_x + crop_size[0], start_y:start_y + crop_size[1], start_z:start_z + crop_size[2]]

def histogram_equalization(image):
    return exposure.equalize_hist(image)

def random_erasing(image, erasing_prob=0.1, max_erase_fraction=0.2):
    if random.random() < erasing_prob:
        shape = image.shape
        erase_fraction = random.uniform(0.01, max_erase_fraction)
        erase_size = [int(s * erase_fraction) for s in shape]
        start_x = random.randint(0, shape[0] - erase_size[0])
        start_y = random.randint(0, shape[1] - erase_size[1])
        start_z = random.randint(0, shape[2] - erase_size[2])
        image[start_x:start_x + erase_size[0], start_y:start_y + erase_size[1], start_z:start_z + erase_size[2]] = 0
    return image

def dropout(image, dropout_prob=0.1):
    mask = np.random.rand(*image.shape) > dropout_prob
    return image * mask

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    distorted_image = ndi.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted_image

def data_augmentation(data, labels, num_per_sample = 5 , target_shape=(256, 256, 166), crop_size=(160, 160, 128), augmentation_probs=None):
    augmented_data = []
    augmented_labels = []

    if augmentation_probs is None:
        augmentation_probs = {
            'crop': 0.3,
            'scale': 0.2,
            'flip': 0.3,
            'rotate': 0.3,
            'blur': 0.2,
            'elastic': 0.2,
            'brightness': 0.2,
            'contrast': 0.2,
            'random_erasing': 0.2,
            'dropout': 0.2,
        }

    for img, label in zip(data, labels):

        for _ in range(num_per_sample):
            augmented_img, augmented_label = augment_image(img, label, crop_size, target_shape, augmentation_probs)
            augmented_data.append(augmented_img)
            augmented_labels.append(augmented_label)

    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)

    combined_data = np.concatenate((data, augmented_data), axis=0)
    combined_labels = np.concatenate((labels, augmented_labels), axis=0)

    return combined_data, combined_labels