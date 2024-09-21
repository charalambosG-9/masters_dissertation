import torch
import torch.nn as nn
import torch.nn.functional as F

class MRI_model(nn.Module):
    def __init__(self, input_shape, num_classes=2, dropout=0.5):
        super(MRI_model, self).__init__()

        self.conv1 = nn.Conv3d(input_shape[1], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.conv7 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        
        self.bn = nn.BatchNorm3d(128)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy_input = torch.zeros((1,) + input_shape[1:])
            dummy_output = self.forward_conv(dummy_input)
            output_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = self.bn(x)
        x = self.dropout(x)

        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

    
class PET_model(nn.Module):
    def __init__(self, input_shape, num_classes=2, dropout=0.5):
        super(PET_model, self).__init__()

        self.conv1 = nn.Conv3d(input_shape[1], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        self.conv7 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        
        self.bn = nn.BatchNorm3d(256)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy_input = torch.zeros((1,) + input_shape[1:])
            dummy_output = self.forward_conv(dummy_input)
            output_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = self.bn(x)
        x = self.dropout(x)

        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x