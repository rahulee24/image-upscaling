import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A single Residual Block.
    This is the core component of the Generator (Pillar 2).
    It consists of: Conv -> BatchNorm -> PReLU -> Conv -> BatchNorm -> Elementwise-Sum
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # The main convolutional block
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(), # Parametric ReLU, learns the negative slope
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        """
        The forward pass for the residual block.
        x -> [BLOCK] -> residual
        return x + residual (This is the "skip connection")
        """
        residual = self.block(x)
        return x + residual

class UpsampleBlock(nn.Module):
    """
    The Upsampling Block.
    It uses a combination of Conv2d and PixelShuffle to increase
    the image resolution (Height and Width) by 2x.
    """
    def __init__(self, in_features, upscale_factor):
        super(UpsampleBlock, self).__init__()
        
        # 1. A convolution that creates (upscale_factor ** 2) more channels.
        # e.g., for 2x, it creates 4x the channels (64 -> 256)
        self.conv = nn.Conv2d(in_features, in_features * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        
        # 2. The PixelShuffle layer.
        # This rearranges the 256 channels into a 2x2 block of pixels,
        # effectively doubling the height and width and reducing the
        # channels back to 64.
        # (Batch, 256, H, W) -> (Batch, 64, H*2, W*2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    """
    The main Generator Model (The "Artist").
    It combines all the building blocks (Pillars 1 & 2).
    """
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()

        if upscale_factor % 2 != 0 or upscale_factor // 2 not in [1, 2, 4]:
             raise ValueError("Upscale factor must be 2, 4, or 8.")
        
        num_upsample_blocks = upscale_factor // 2

        # --- 1. Initial Convolution (The "Head") ---
        # Takes the 3-channel LR image and maps it to 64 features.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # --- 2. Residual Blocks (The "Body") ---
        # 16 identical residual blocks, as defined above.
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(in_features=64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # --- 3. Bridge Convolution (Connects Body to Tail) ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # --- 4. Upsampling Blocks (The "Tail") ---
        # Uses PixelShuffle to upscale.
        # If upscale_factor=4, we need two 2x upsample blocks.
        upsampling = []
        for _ in range(num_upsample_blocks):
            upsampling.append(UpsampleBlock(in_features=64, upscale_factor=2))
        self.upsampling = nn.Sequential(*upsampling)

        # --- 5. Final Output Convolution ---
        # Takes the 64 upscaled features and collapses them
        # back into a 3-channel (RGB) image.
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh() # Tanh squishes the output to be between -1 and 1
        )

    def forward(self, x):
        """ The full data path for the Generator """
        # 1. Pass through the head
        x1 = self.conv1(x)
        
        # 2. Pass through the body
        res_out = self.res_blocks(x1)
        
        # 3. Add global skip connection (input of body + output of body)
        x = x1 + self.conv2(res_out)
        
        # 4. Pass through the upsampling tail
        x = self.upsampling(x)
        
        # 5. Pass through the final output layer
        x = self.conv_out(x)
        
        return x

# --- A small test to run if you execute this file directly ---
if __name__ == "__main__":
    # Test for 4x upscaling
    # Input size: (Batch, Channels, Height, Width)
    LR_IMAGE_SIZE = 64
    UPSCALE_FACTOR = 4
    HR_IMAGE_SIZE = LR_IMAGE_SIZE * UPSCALE_FACTOR

    # Create a generator
    gen = Generator(upscale_factor=UPSCALE_FACTOR)
    
    # Create a random "image" tensor
    # (1 batch, 3 channels, 64x64 size)
    test_lr_image = torch.randn(1, 3, LR_IMAGE_SIZE, LR_IMAGE_SIZE)
    
    print(f"--- Generator Test (4x) ---")
    print(f"Input shape:  {test_lr_image.shape}")
    
    # Pass the image through the generator
    with torch.no_grad(): # We don't need to calculate gradients for this test
        test_hr_image = gen(test_lr_image)
    
    print(f"Output shape: {test_hr_image.shape}")
    
    # Check if the output shape is correct
    assert test_hr_image.shape == (1, 3, HR_IMAGE_SIZE, HR_IMAGE_SIZE)
    
    print("\nGenerator test passed!")