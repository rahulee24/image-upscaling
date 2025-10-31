import torch
import torch.nn as nn
import torchvision.models as models

# --- Pillar 4: The Perceptual Loss (The "Secret Sauce") ---
# We are creating a "model" that is just the first 36 layers
# of a pre-trained VGG-19 network. We will not train this model.
# We will only use it to extract "features" from images.

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load a pre-trained VGG-19 model (trained on ImageNet)
        vgg19_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Get its "features" section (all the convolutional layers)
        # We take the first 36 layers, which is up to the 'conv5_4' layer
        # as specified in the SRGAN paper.
        self.features = nn.Sequential(
            *list(vgg19_model.features.children())[:36]
        ).eval() # .eval() puts it in "evaluation" mode
        
        # --- CRITICAL ---
        # We must "freeze" this model. We are not training VGG.
        # We are only using it as a fixed tool to measure features.
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, img):
        # Pass the image through the VGG layers and return the features
        return self.features(img)

if __name__ == "__main__":
    # Test the feature extractor
    extractor = VGGFeatureExtractor()
    
    # A fake 256x256 HR image
    fake_hr_image = torch.randn(1, 3, 256, 256)
    
    features = extractor(fake_hr_image)
    
    print(f"Input image shape: {fake_hr_image.shape}")
    print(f"Output feature shape: {features.shape}")
    # The output shape will be (BatchSize, 512, 16, 16)
    # This is the (Channels, H, W) of the 'conv5_4' layer
    assert features.shape == (1, 512, 16, 16)
    print("\nVGG Feature Extractor test passed!")