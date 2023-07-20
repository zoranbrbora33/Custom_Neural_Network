from torch import nn
import torch.nn.functional as F


class MyConvNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolutional layer: input channels=1 (grayscale), output channels=10, kernel size=(5, 5), padding=2
        self.__conv1 = nn.Conv2d(1, 10, (5, 5), padding=2)

        # Second convolutional layer: input channels=10, output channels=20, kernel size=(5, 5), padding=2
        self.__conv2 = nn.Conv2d(10, 20, (5, 5), padding=2)

        # Max pooling layer with kernel size=(2, 2)
        self.__pool = nn.MaxPool2d((2))

        # Fully connected layer: input features=980 (after max pooling), output features=50
        self.__fc1 = nn.Linear(980, 50)

        # Fully connected layer: input features=50, output features=10 (for 10 classes)
        self.__fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # First convolutional layer with ReLU activation and max pooling
        x = self.__conv1(x)
        x = F.relu(x)
        x = self.__pool(x)

        # Second convolutional layer with ReLU activation and max pooling
        x = self.__conv2(x)
        x = F.relu(x)
        x = self.__pool(x)

        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 980)

        # First fully connected layer with ReLU activation
        x = self.__fc1(x)
        x = F.relu(x)

        # Second fully connected layer (output layer)
        x = self.__fc2(x)

        return x
