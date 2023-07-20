import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import math
from my_conv_nn import MyConvNeuralNetwork

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 0.01
NUM_OF_EPOCHS = 20
ITERATION_LOGGING_FREQUENCY = 100
EXPERIMENT_NAME = 'runs/custom_net_mnist'
MODELS_PATH = 'models'

# Global variable to keep track of overall iterations
overall_iteration = 0

# Create a SummaryWriter to log training data for TensorBoard
writer = SummaryWriter(EXPERIMENT_NAME)

# Check if GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define data preprocessing transforms
preprocess = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

# Download MNIST dataset and apply preprocessing
mnist_dataset_train = MNIST(
    'data', train=True, download=True, transform=preprocess)
mnist_dataset_validation = MNIST(
    'data', train=False, download=True, transform=preprocess)

# Display the first three training images with their labels
fig, axes = plt.subplots(3)
for i in range(3):
    image, label = mnist_dataset_train[i]
    image = image.squeeze()
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(str(label))

# Create data loaders for training and validation datasets
mnist_dataloader_train = DataLoader(
    mnist_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
mnist_dataloader_validation = DataLoader(
    mnist_dataset_validation, batch_size=BATCH_SIZE)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Initialize the neural network model
model = MyConvNeuralNetwork().to(device)

# Define the optimizer for training the model
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Function to train the model for one epoch


def train(dataloader, model, loss_fn, optimizer, epoch):
    global overall_iteration
    running_loss = 0
    dataset_size = len(dataloader.dataset)
    for iteration_in_epoch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        overall_iteration += 1

        # Log training loss and learning rate to TensorBoard every `ITERATION_LOGGING_FREQUENCY` iterations
        if (iteration_in_epoch + 1) % ITERATION_LOGGING_FREQUENCY == 0:
            average_loss = running_loss / ITERATION_LOGGING_FREQUENCY
            learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar('training_loss', average_loss, overall_iteration)
            writer.add_scalar('learning_rate', learning_rate,
                              overall_iteration)
            running_loss = 0

            # Print progress for each iteration
            print('Epoch {} / {} [{} / {}] Loss: {:.4f} Learning rate: {}'
                  .format(epoch, NUM_OF_EPOCHS,
                          iteration_in_epoch + 1,
                          math.ceil(dataset_size // BATCH_SIZE),
                          average_loss,
                          learning_rate))

# Function to validate the model after each epoch


def validate(dataloader, model, epoch_num):
    print('Start validating...')
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predictions = nn.functional.softmax(predictions, dim=1)
            predicted_classes = predictions.argmax(1)
            correct_predicted_classes = (
                predicted_classes == labels).sum().item()
            total_correct += correct_predicted_classes

        accuracy = total_correct / len(dataloader.dataset)

        # Log validation accuracy to TensorBoard
        writer.add_scalar('validation accuracy', accuracy, epoch_num)

        # Print validation accuracy for this epoch
        print('Validation accuracy: {:.2f}'.format(accuracy))

        # Save the model after each epoch with a filename that includes the epoch number
        torch.save(model, MODELS_PATH +
                   '/model_after_epoch_{}.pt'.format(epoch_num))

    print('Done with validation!')


# Main training loop
for epoch in range(1, NUM_OF_EPOCHS + 1):

    # Perform training for one epoch
    train(mnist_dataloader_train, model, loss_fn, optimizer, epoch)

    # Validate the model after each epoch
    validate(mnist_dataloader_validation, model, epoch)
