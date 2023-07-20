# Import required libraries
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt

# Define the input size for the image (width, height)
IMAGE_INPUT_SIZE = (28, 28)

# Check if GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to the input image
image_path = '6.png'

# Open the image and convert it to grayscale
image = Image.open(image_path).convert('L')

# Load the pre-trained model and move it to the specified device (CPU or GPU)
model = torch.load('models/model_after_epoch_10.pt').to(device)

# Set the model to evaluation mode (important for models with batch normalization or dropout)
model.eval()

# Create a series of image preprocessing transformations
preprocess = transforms.Compose([transforms.Resize(IMAGE_INPUT_SIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5], std=[0.5])])

# Apply the preprocessing transformations to the input image
input_tensor = preprocess(image)

# Move the preprocessed image tensor to the specified device (CPU or GPU)
input_tensor = input_tensor.to(device)

# Add a batch dimension to the input tensor (required for models with batch processing)
input_tensor = torch.unsqueeze(input_tensor, 0)

# Forward pass through the model to obtain predictions
output = model(input_tensor)

# Apply softmax activation to the output probabilities to get class probabilities
output = nn.functional.softmax(output, 1)

# Get the predicted class index (the class with the highest probability)
predicted_class = output.argmax(1).item()

# Get the probability of the predicted class
probability = output[0][predicted_class]

# Display the original image using matplotlib
plt.imshow(image, cmap='gray')

# Set the title of the plot with the predicted class and its probability
plt.title('class: {}\nprobability: {:.4f}'.format(
    predicted_class, probability))

# Show the plot
plt.show()
