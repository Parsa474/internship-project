import math
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

def gen_train():
    for x_train, y_train in train_loader:
        I = x_train[0].numpy()
        K = np.zeros((1, 28, 28, 1), np.float32)
        I = (I / 255.0)
        K[0, ..., 0] = I.astype(np.float32)

        pts1 = np.float32([[(0, 0), (28, 0), (0, 28), (28, 28)]])

        rx = np.random.randint(0, 5, 1)[0]
        ry = np.random.randint(0, 5, 1)[0]
        pts2 = np.float32([[(5 + rx, ry), (20 + rx, ry), (0, 28), (28, 28)]])
        M = cv2.getPerspectiveTransform(pts2, pts1)
        Y = cv2.warpPerspective(K[0, :, :, 0], M, (28, 28))[None, ..., None]

        yield torch.from_numpy(Y), torch.from_numpy(K)

def gen_test():
    for x_test, y_test in test_loader:
        I = x_test[0].numpy()
        K = np.zeros((1, 28, 28, 1), np.float32)
        I = (I / 255.0)
        K[0, ..., 0] = I.astype(np.float32)

        pts1 = np.float32([[(0, 0), (28, 0), (0, 28), (28, 28)]])
        rx = np.random.randint(0, 5, 1)[0]
        ry = np.random.randint(0, 5, 1)[0]
        pts2 = np.float32([[(5 + rx, ry), (20 + rx, ry), (0, 28), (28, 28)]])
        M = cv2.getPerspectiveTransform(pts2, pts1)
        Y = cv2.warpPerspective(K[0, :, :, 0], M, (28, 28))[None, ..., None]

        yield torch.from_numpy(Y), torch.from_numpy(K)

g_train = gen_train()
x_train, y_train = next(g_train)
plt.title('input (perspectived)')
plt.imshow(x_train[0, ..., 0], cmap='gray')
plt.figure()
plt.title('output (original)')
plt.imshow(y_train[0, ..., 0], cmap='gray')
plt.show()

import torch.nn as nn

class PerspectiveCorrectionCNN(nn.Module):
    def __init__(self):
        super(PerspectiveCorrectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input channels set to 1 for grayscale images
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 784)  # Output size matches input size (28x28)

    def forward(self, x):
        x = torch.relu(self.conv1(x.permute(0, 3, 1, 2)))  # Transpose the dimensions
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation to output values in [0, 1]
        return x

model = PerspectiveCorrectionCNN()

import torch.optim as optim

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate

# Training loop
num_epochs = 10  # You can adjust the number of epochs
for epoch in range(num_epochs):
    model.train()
    for x_train, y_train in gen_train():
        optimizer.zero_grad()
        # Transpose and reshape the target tensor to match the output shape
        y_train = y_train.permute(0, 3, 1, 2).view(-1, 784)
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_test, y_test in gen_test():
            # Transpose and reshape the target tensor to match the output shape
            y_test = y_test.permute(0, 3, 1, 2).view(-1, 784)
            outputs = model(x_test)
            loss = criterion(outputs, y_test)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_loss:.4f}')

# Save the trained model if needed
torch.save(model.state_dict(), 'perspective_correction_model.pth')
