import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A helper convolutional block for the Discriminator.
    This is a basic block: Conv -> BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True)) # LeakyReLU is standard for GAN discriminators
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    """
    The Discriminator Model (The "Art Critic").
    This is a VGG-style classifier. Its job is to take an image
    (256x256) and output a single number (probability) from 0 to 1
    (0 = Fake, 1 = Real).
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        # This architecture progressively downsamples the image
        # while increasing the number of features.
        # Stride=2 halves the image size.
        
        # Input: (3, 256, 256)
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=False), # (64, 256, 256)
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),  # (64, 128, 128)
            
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1), # (128, 128, 128)
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1), # (128, 64, 64)
            
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1), # (256, 64, 64)
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1), # (256, 32, 32)
            
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1), # (512, 32, 32)
            ConvBlock(512, 512, kernel_size=3, stride=2, padding=1), # (512, 16, 16)
            
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1), # (1024, 16, 16)
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1), # (1024, 8, 8)
        )

        # This is the "Classifier" part.
        # It takes the final 1024x8x8 feature map and flattens it.
        self.classifier = nn.Sequential(
            # AdaptiveAvgPool2d flattens the HxW dimensions to 1x1
            # (1024, 8, 8) -> (1024, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # A 1x1 convolution acts like a Fully Connected (Linear) layer
            # on the 1024 features.
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
            
            # Sigmoid squishes the final score to be between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Pass the image through the feature-extracting "body"
        x = self.blocks(x)
        
        # 2. Pass the features through the "classifier" head
        x = self.classifier(x)
        
        # 3. Flatten the output from (Batch, 1, 1, 1) to (Batch)
        # This makes it compatible with the BCELoss function
        return torch.flatten(x)


# --- A small test to run if you execute this file directly ---
if __name__ == "__main__":
    # The Discriminator expects a 256x256 image
    HR_IMAGE_SIZE = 256
    
    # Create a discriminator
    disc = Discriminator()
    
    # Create a random "image" tensor
    # (1 batch, 3 channels, 256x256 size)
    test_hr_image = torch.randn(1, 3, HR_IMAGE_SIZE, HR_IMAGE_SIZE)
    
    print(f"--- Discriminator Test ---")
    print(f"Input shape: {test_hr_image.shape}")
    
    # Pass the image through the discriminator
    with torch.no_grad():
        prediction = disc(test_hr_image)
        
    print(f"Output shape: {prediction.shape}")
    print(f"Output value (probability): {prediction.item():.4f}")
    
    # Check if the output shape is correct (1 value per batch item)
    assert prediction.shape == (1,)
    # Check if the value is a probability (between 0 and 1)
    assert 0.0 <= prediction.item() <= 1.0
    
    print("\nDiscriminator test passed!")