import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os

# Import your model definition
# We need this to re-create the model structure before loading its weights
try:
    from generator import Generator
except ImportError:
    print("ERROR: Could not import Generator from generator.py.")
    print("Please make sure generator.py is in the same directory.")
    exit()

# --- Configuration ---
MODEL_PATH = "./models/srgan_generator_final.pth"
# We'll look for this base name with .png, .jpg, or .jpeg extensions
TEST_IMG_NAME = "test_lr"
OUTPUT_IMG_PATH = "./test_hr_result.png"
# --- End Configuration ---

def main():
    """
    Main function to run the inference.
    """
    # Check if a GPU (CUDA) is available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize and Load the Model
    # We must first create an instance of the Generator, just as we defined it
    # (with the same upscale_factor from training, which was 4)
    model = Generator(upscale_factor=4).to(device)

    # Load the trained weights (the .pth file) into the model structure
    try:
        # We add weights_only=True for security, as recommended by PyTorch
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Did you run train.py and did it save the final model?")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("This might happen if your generator.py is different from the one used for training.")
        return
        
    # Set the model to "evaluation" mode (disables things like dropout, etc.)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")

    # 2. Load and Prepare the Low-Resolution Image
    
    # Find the test image, checking for .png, .jpg, or .jpeg
    test_img_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        path = f"./{TEST_IMG_NAME}{ext}"
        if os.path.exists(path):
            test_img_path = path
            break
            
    if test_img_path is None:
        print(f"ERROR: Test image not found (tried {TEST_IMG_NAME}.png, {TEST_IMG_NAME}.jpg, etc.)")
        print(f"Please add your low-res image as '{TEST_IMG_NAME}.png' or '{TEST_IMG_NAME}.jpg'")
        return

    print(f"Loading test image from: {test_img_path}")
    lr_image = Image.open(test_img_path).convert("RGB")

    # Convert the PIL Image (range [0, 255]) to a PyTorch Tensor (range [0.0, 1.0])
    lr_tensor = ToTensor()(lr_image).to(device)
    # Add a "batch" dimension (from [C, H, W] to [B, C, H, W])
    # The model expects a batch of images, even if it's just one.
    lr_tensor = lr_tensor.unsqueeze(0) 
    print(f"Loaded low-res image. Shape: {lr_tensor.shape}")

    # 3. Run Inference
    print("Running model... (this may take a moment)")
    # We run this inside torch.no_grad() to save memory and computations,
    # as we don't need to calculate gradients for inference.
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    print("Model inference complete.")

    # 4. Post-process and Save the Output Image
    
    # Un-normalize the output tensor
    # The generator's Tanh() outputs [-1, 1]. We remap this to [0, 1].
    sr_tensor = (sr_tensor + 1.0) / 2.0

    # Clamp values to the valid [0.0, 1.0] range
    sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

    # Remove the "batch" dimension (from [B, C, H, W] to [C, H, W])
    sr_tensor = sr_tensor.squeeze(0)
    
    # Move tensor from GPU back to CPU for saving
    sr_tensor_cpu = sr_tensor.cpu()

    # Convert the PyTorch Tensor back to a PIL Image
    sr_image = ToPILImage()(sr_tensor_cpu)

    # Save the final high-resolution image
    sr_image.save(OUTPUT_IMG_PATH)
    
    print("-" * 30)
    print(f"SUCCESS: Upscaled image saved to {OUTPUT_IMG_PATH}")
    print(f"Original shape: {lr_tensor.shape[2:]}, New shape: {sr_image.size}")
    print("-" * 30)


if __name__ == "__main__":
    main()

