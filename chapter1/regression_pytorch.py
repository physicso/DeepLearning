import torch
import torch.nn as nn
import torch.optim as optim

# Convert the training data to PyTorch tensors.
train_X = torch.tensor([30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=torch.float32).unsqueeze(1)
train_Y = torch.tensor([320.0, 360.0, 400.0, 455.0, 490.0, 546.0, 580.0], dtype=torch.float32).unsqueeze(1)
train_X /= 100.0  # Normalize the data simply.
train_Y /= 100.0


# Build a linear fitting model consisting of two parameters: slope and bias, which is equivalent to a fully connected layer.
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


model = LinearRegression()

# Define the loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Model training.
for epoch in range(10):
    # Forward propagation
    outputs = model(train_X)
    # Compute the loss.
    loss = criterion(outputs, train_Y)
    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print the loss of the current epoch.
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, loss.item()))
