import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import gc
import time
import optuna
from myconfig import trained_model_path, torch_tensors_path, print_metrics
from models import MRI_model

start_time = time.time()

# Load data
x_train = torch.load(os.path.join(torch_tensors_path, 'x_train_mri.pt'))
x_test = torch.load(os.path.join(torch_tensors_path, 'x_test_mri.pt'))

y_train = torch.load(os.path.join(torch_tensors_path, 'y_train.pt'))
y_test = torch.load(os.path.join(torch_tensors_path, 'y_test.pt'))

print(f"Data loaded in {time.time() - start_time:.2f} seconds")

# Objective function for Bayesian Optimization
def objective(trial):
    # Define the search space
    learning_rate = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Initialize the model with the suggested dropout rate
    model = MRI_model(x_train.shape, len(np.unique(y_train)), dropout=dropout_rate).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss()

    # Train-validation split
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(x_train_split, y_train_split)
    val_dataset = torch.utils.data.TensorDataset(x_val_split, y_val_split)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    lowest_val_loss = np.inf
    patience = 0

    for epoch in range(50):  # You can reduce the number of epochs for faster optimization
        model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(outputs.detach(), axis=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(all_labels, all_predictions)

        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predictions = torch.argmax(outputs, axis=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        val_accuracy = accuracy_score(all_labels, all_predictions)

        # If validation loss improves, save the model
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        
        # Early stopping condition
        if patience > 10:
            break

        print(f'Epoch {epoch + 1}, Running loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}')
        print_metrics(all_labels, all_predictions, conf_matrix=True, classes=['AD', 'CN', 'MCI'], filename=None)

    # Return the validation loss (you could also return accuracy if preferred)
    return val_loss

# Create an Optuna study to minimize validation loss
study = optuna.create_study(direction='minimize')

# Run optimization for a specified number of trials
study.optimize(objective, n_trials=30)

# Retrieve the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")
print(f"Best validation loss: {study.best_value}")

# Final training with optimized hyperparameters
learning_rate = best_params['lr']
weight_decay = best_params['weight_decay']
dropout_rate = best_params['dropout']

# Initialize the model with the best hyperparameters
model = MRI_model(x_train.shape, len(np.unique(y_train)), dropout=dropout_rate).cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
criterion = nn.CrossEntropyLoss()

# Train-validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

lowest_val_loss = np.inf
patience = 0

# Train the model with the optimized hyperparameters
for epoch in range(50):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = torch.argmax(outputs.detach(), axis=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(all_labels, all_predictions)

    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            predictions = torch.argmax(outputs, axis=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    val_accuracy = accuracy_score(all_labels, all_predictions)

    print(f'Epoch {epoch + 1}, Running loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}')
    print_metrics(all_labels, all_predictions, conf_matrix=True, classes=['AD', 'CN', 'MCI'], filename=None)

    # Save the best model
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience = 0
    else:
        patience += 1

    train_accuracies.append(train_accuracy)
    train_losses.append(running_loss / len(train_loader))
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)

    del images, labels, outputs
    torch.cuda.empty_cache()
    gc.collect()

    if patience > 15:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Plot training/validation metrics
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy Fold')
plt.plot(val_accuracies, label='Validation Accuracy Fold')
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.title('Accuracy Metrics MRI Model', fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss Fold')
plt.plot(val_losses, label='Validation Loss Fold')
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('Loss', fontsize=24)
plt.title('Loss Metrics MRI Model', fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)

plt.savefig('graphs/training_metrics_MRI.jpg')
plt.close()

# Test the model on test set
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model.load_state_dict(best_model_state)

if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), os.path.join(trained_model_path, 'mri_model.pt'))
else:
    torch.save(model.state_dict(), os.path.join(trained_model_path, 'mri_model.pt'))

all_predictions = []
all_labels = []
test_loss = 0.0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        predictions = torch.argmax(outputs, axis=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

    del images, labels, outputs
    torch.cuda.empty_cache()
    gc.collect()

print('')
print_metrics(all_labels, all_predictions, conf_matrix=True, classes=['AD', 'CN', 'MCI'], filename='MRI_training_confusion_matrix')
