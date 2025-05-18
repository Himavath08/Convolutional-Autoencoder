# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.
## DESIGN STEPS
### STEP 1:
Import necessary libraries including PyTorch, torchvision, and matplotlib.
### STEP 2:
Load the MNIST dataset with transforms to convert images to tensors.
### STEP 3:
Add Gaussian noise to training and testing images using a custom function.
### STEP 4:
Define the architecture of a convolutional autoencoder:
Encoder: Conv2D layers with ReLU + MaxPool
Decoder: ConvTranspose2D layers with ReLU/Sigmoid
### STEP 5:
Initialize model, define loss function (MSE) and optimizer (Adam).
### STEP 6:
Train the model using noisy images as input and original images as target.
### STEP 7:
Visualize and compare original, noisy, and denoised images.
## PROGRAM
### Name:Himavath M
### Register Number:212223240053
```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 7, 7)
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B, 1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
````
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
````
## OUTPUT
### Model Summary
![image](https://github.com/user-attachments/assets/7510de8d-c8c1-416b-b806-e5fb862829a3)

### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/user-attachments/assets/13817891-c4e7-4a36-b140-fd9a80666b03)

## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
