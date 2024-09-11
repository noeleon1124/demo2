import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Hyper-parameters
num_epochs = 8
learning_rate = 1e-3
channels = 1  # Number of input channels (for grayscale images, this is 1)
depth = 64  # Depth for convolution layers

# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
Y = lfw_people.target

# Preprocessing
X = X[:, np.newaxis, :, :]  # Add channel dimension
X = X.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
print("X_min:", X.min(), "X_max:", X.max())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the BasicBlock and ConvNetwork modules
class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()

    def forward(self, x):
        out = self.norm1(self.features(x))
        return self.act_layer(out)

class ConvNetwork(nn.Module):
    def __init__(self, dim, in_channels, num_classes=10):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling_depth = 2 * self.dim

        self.block1 = BasicBlock(self.in_channels, self.dim, 3, stride=2)
        self.block2 = BasicBlock(self.dim, self.dim * 2, 3, stride=2)

        self.ave_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(self.pooling_depth, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        out = self.ave_pool(x)
        out = out.view(out.size(0), -1)  # Flatten

        return self.classifier(out)

# Instantiate the ConvNetwork model with the defined hyperparameters
num_classes = len(lfw_people.target_names)
model = ConvNetwork(dim=depth, in_channels=channels, num_classes=num_classes)

# Define the loss function and optimizer with the learning rate from the hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluate the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=lfw_people.target_names)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)