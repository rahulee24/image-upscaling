import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm

# Import our custom modules
# (Make sure your file names are simple: generator.py, discriminator.py, etc.)
from generator import Generator
from discriminator import Discriminator
from vgg_feature_extractor import VGGFeatureExtractor
from dataset import ImageDataset, CROP_SIZE, UPSCALE_FACTOR

# --- NEW LINE FOR THE FIX ---
# This is needed for Windows multiprocessing
from multiprocessing import freeze_support

# --- Configuration (Hyperparameters) ---
# These are fine to leave at the global level
# as they are just variable definitions.

# Training parameters
NUM_EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE_G = 1e-4  # Learning rate for Generator
LEARNING_RATE_D = 1e-4  # Learning rate for Discriminator
B1 = 0.9  # Adam optimizer beta1
B2 = 0.999 # Adam optimizer beta2

# Loss weights (these are key to SRGAN)
LAMBDA_CONTENT = 1.0     # Weight for Perceptual (VGG) Loss
LAMBDA_ADV = 1e-3      # Weight for Adversarial Loss (how much to "fool" the discriminator)
LAMBDA_PIXEL = 1e-2      # Weight for simple Pixel-wise (L1) Loss

# Dataset parameters
# We get CROP_SIZE and UPSCALE_FACTOR from dataset.py
HR_IMAGE_SIZE = CROP_SIZE
LR_IMAGE_SIZE = HR_IMAGE_SIZE // UPSCALE_FACTOR
NUM_WORKERS = 4  # Number of parallel threads for data loading. Set to 0 if this gives errors.

# Paths (MAKE SURE THESE FOLDERS EXIST)
DATASET_PATH = "./data/DIV2K_train_HR"  # Path to your high-resolution training images
MODEL_SAVE_PATH = "./models"            # Folder to save model checkpoints
IMAGE_SAVE_PATH = "./images"            # Folder to save training preview images
PRETRAINED_GEN_PATH = None              # Optional: path to a pretrained generator (e.g., from a previous run)

# Create output folders if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

# -----------------------------------------------------------------
# --- ALL EXECUTABLE CODE MUST GO INSIDE THIS BLOCK ---
# -----------------------------------------------------------------

def main():
    """
    Main training function.
    """
    
    # --- Device Setup ---
    # Check if a GPU (CUDA) is available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    print("Loading dataset...")
    dataset = ImageDataset(DATASET_PATH, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # Helps speed up CPU to GPU data transfer
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    print(f"Dataset loaded with {len(dataset)} images.")

    # --- Initialize Models ---
    print("Initializing models...")
    # Generator
    generator = Generator(in_channels=3, n_residual_blocks=16, upscale_factor=UPSCALE_FACTOR).to(device)
    
    # Discriminator
    discriminator = Discriminator(in_channels=3).to(device)
    
    # VGG Feature Extractor (for Perceptual Loss)
    # We set requires_grad=False because we *never* train the VGG model.
    vgg_extractor = VGGFeatureExtractor().to(device)
    vgg_extractor.eval()
    
    # Load pretrained generator if specified
    if PRETRAINED_GEN_PATH:
        try:
            generator.load_state_dict(torch.load(PRETRAINED_GEN_PATH))
            print(f"Loaded pretrained generator weights from {PRETRAINED_GEN_PATH}")
        except FileNotFoundError:
            print(f"WARNING: Pretrained generator path not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading pretrained generator: {e}. Starting from scratch.")

    # --- Loss Functions ---
    # 1. Adversarial Loss (how "real" or "fake" the image is)
    criterion_adv = nn.BCEWithLogitsLoss().to(device) # Binary Cross-Entropy
    
    # 2. Content/Perceptual Loss (VGG features)
    criterion_content = nn.L1Loss().to(device) # L1 Loss (Mean Absolute Error)
    
    # 3. Pixel Loss (simple pixel-by-pixel comparison)
    criterion_pixel = nn.L1Loss().to(device)

    # --- Optimizers ---
    # We have two separate optimizers: one for the Generator...
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(B1, B2))
    # ...and one for the Discriminator.
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(B1, B2))

    # --- The Training Loop ---
    print("Starting training...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Set models to training mode
        generator.train()
        discriminator.train()
        
        # Use tqdm for a nice progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{NUM_EPOCHS}]")
        
        for i, batch in pbar:
            # Get the Low-Res (lr) and High-Res (hr) images from the dataloader
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)
            
            # Create "real" and "fake" labels for the adversarial loss
            # We use "soft" labels (0.9 instead of 1.0) for stability, a common GAN trick
            real_labels = torch.full((lr_images.size(0), 1), 0.9, device=device, dtype=torch.float32)
            fake_labels = torch.full((lr_images.size(0), 1), 0.1, device=device, dtype=torch.float32)

            # =======================================
            #  (1) Train the Generator
            # =======================================
            
            # We only update the generator's weights, so zero its gradients
            optimizer_g.zero_grad()
            
            # Generate fake High-Res (SR) images
            # sr_images = "Super-Resolution" images
            sr_images = generator(lr_images)
            
            # --- Calculate Generator Losses ---
            
            # 1. Pixel Loss (how far are the pixels from the real HR image?)
            loss_pixel = criterion_pixel(sr_images, hr_images)
            
            # 2. Content/Perceptual Loss (how far are the VGG *features* from the real HR image?)
            # We pass both images through the VGG model to get their features
            sr_features = vgg_extractor(sr_images)
            hr_features = vgg_extractor(hr_images)
            loss_content = criterion_content(sr_features, hr_features)
            
            # 3. Adversarial Loss (how well did we "fool" the discriminator?)
            # We pass the *fake* images through the discriminator...
            sr_pred = discriminator(sr_images)
            # ...and see how close its prediction was to the "real" label.
            # The generator wants the discriminator to think its fakes are real.
            loss_adv = criterion_adv(sr_pred.flatten(), real_labels.flatten())
            
            # --- Combine Generator Losses ---
            # Total loss is a weighted sum of the three
            loss_g = (LAMBDA_CONTENT * loss_content) + \
                     (LAMBDA_ADV * loss_adv) + \
                     (LAMBDA_PIXEL * loss_pixel)
            
            # Backpropagate and update generator weights
            loss_g.backward()
            optimizer_g.step()

            # =======================================
            #  (2) Train the Discriminator
            # =======================================
            
            # We only update the discriminator's weights, so zero its gradients
            optimizer_d.zero_grad()
            
            # --- Calculate Discriminator Losses ---
            
            # 1. Loss on REAL images
            # Pass the *real* HR images through the discriminator
            hr_pred = discriminator(hr_images)
            # Calculate the loss vs. the "real" labels
            loss_d_real = criterion_adv(hr_pred.flatten(), real_labels.flatten())
            
            # 2. Loss on FAKE images
            # Pass the *fake* (generated) images through the discriminator
            # We detach() sr_images to prevent gradients from flowing back to the generator
            sr_pred = discriminator(sr_images.detach())
            # Calculate the loss vs. the "fake" labels
            loss_d_fake = criterion_adv(sr_pred.flatten(), fake_labels.flatten())
            
            # --- Combine Discriminator Losses ---
            # Total loss is the average of the real and fake losses
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            # Backpropagate and update discriminator weights
            loss_d.backward()
            optimizer_d.step()
            
            # --- Update Progress Bar ---
            # Show the key losses in the tqdm progress bar
            pbar.set_postfix_str(
                f"[D loss: {loss_d.item():.4f}] "
                f"[G loss: {loss_g.item():.4f} "
                f"(Content: {loss_content.item():.4f}, Adv: {loss_adv.item():.4f}, Pixel: {loss_pixel.item():.4f})]"
            )
        
        # --- End of Epoch ---
        print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] finished.")
        
        # Save a preview of the generator's output
        if epoch % 5 == 0:  # Save every 5 epochs
            # Save the last batch's result
            # We stack the LR, the generated SR, and the real HR for comparison
            lr_resized = nn.functional.interpolate(lr_images, scale_factor=UPSCALE_FACTOR, mode='bicubic')
            comparison_grid = torch.cat((lr_resized, sr_images, hr_images), dim=0)
            
            save_path = os.path.join(IMAGE_SAVE_PATH, f"epoch_{epoch:03d}.png")
            save_image(comparison_grid, save_path, nrow=BATCH_SIZE, normalize=True)
            print(f"Saved training preview to {save_path}")

        # Save model checkpoints
        if epoch % 10 == 0:  # Save every 10 epochs
            # Save Generator
            gen_path = os.path.join(MODEL_SAVE_PATH, f"srgan_generator_epoch_{epoch:03d}.pth")
            torch.save(generator.state_dict(), gen_path)
            
            # Save Discriminator
            disc_path = os.path.join(MODEL_SAVE_PATH, f"srgan_discriminator_epoch_{epoch:03d}.pth")
            torch.save(discriminator.state_dict(), disc_path)
            
            print(f"Saved model checkpoints to {MODEL_SAVE_PATH}")

    # --- End of Training ---
    print("Training finished.")
    
    # Save the final generator model
    final_gen_path = os.path.join(MODEL_SAVE_PATH, "srgan_generator_final.pth")
    torch.save(generator.state_dict(), final_gen_path)
    print(f"Saved final generator model to {final_gen_path}")


# This is the "proper idiom" the error message was talking about.
if __name__ == '__main__':
    # --- NEW LINE FOR THE FIX ---
    # This must be called inside the __name__ == "__main__" block
    freeze_support()
    
    # Run the main training function
    main()

