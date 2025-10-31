import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import glob

# --- Configuration ---
# These are the *default* values. The training script will import these.
# We define them here so both scripts can share the same constants.
CROP_SIZE = 96      # The size of the high-res (HR) patch to crop
UPSCALE_FACTOR = 4  # The factor to upscale (e.g., 4x)

class ImageDataset(Dataset):
    """
    Custom Dataset for loading and transforming high-resolution images
    and generating their low-resolution counterparts on-the-fly.
    """
    def __init__(self, hr_image_dir, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR):
        """
        Args:
            hr_image_dir (str): Path to the directory containing high-resolution images.
            crop_size (int): The size to randomly crop the HR images to (e.g., 96x96).
            upscale_factor (int): The factor to downscale and then upscale (e.g., 4).
        """
        self.hr_image_dir = hr_image_dir
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        
        # Get a list of all image file paths
        # We use glob to find all .png, .jpg, and .jpeg images
        self.image_files = sorted(glob.glob(os.path.join(hr_image_dir, "*.*[jpJpPnNgGg]")))
        
        # --- Define Transformations ---
        
        # Transform for High-Resolution (HR) images:
        # 1. Randomly crop to `crop_size`
        # 2. Convert to a PyTorch Tensor
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor()  # Converts from (H, W, C) [0, 255] to (C, H, W) [0.0, 1.0]
        ])
        
        # Transform for Low-Resolution (LR) images:
        # This is applied *after* the HR image is cropped and converted to a Tensor.
        # 1. Resize (downscale) to `crop_size / upscale_factor`
        # 2. Resize (upscale) back to `crop_size` using 'bicubic' interpolation.
        #    This creates the blurry, low-res input that the generator will receive.
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(), # Convert tensor back to PIL Image to use transforms.Resize
            transforms.Resize(
                (self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor() # Convert back to tensor
        ])
        
        # We'll also create a transform to scale the LR image back up to HR size
        # for visualization, but without the detail. This is what a "non-AI"
        # upscaler would do. This is NOT used for training, but is useful for
        # the test block at the bottom.
        self.lr_display_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (self.crop_size, self.crop_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor()
        ])


    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Gets a single data point (LR-HR image pair).
        This is called by the DataLoader.
        """
        # Load the high-resolution image
        img_path = self.image_files[index]
        hr_image = Image.open(img_path).convert("RGB")
        
        # Apply the HR transform (random crop + to tensor)
        hr_image_tensor = self.hr_transform(hr_image)
        
        # Apply the LR transform (downscale)
        # Note: We create the LR image *from* the cropped HR tensor
        lr_image_tensor = self.lr_transform(hr_image_tensor)
        
        # Return a dictionary containing the LR and HR images
        return {"lr": lr_image_tensor, "hr": hr_image_tensor}


# --- Test Block ---
# This code only runs if you execute `python dataset.py` directly.
# It's a "sanity check" to make sure the dataset is working.
if __name__ == "__main__":
    
    # --- IMPORTANT ---
    # Put the path to your downloaded DIV2K data here!
    # Or just create a folder "data/DIV2K_train_HR" and put a few .jpg files in it to test.
    TEST_DATA_PATH = "./data/DIV2K_train_HR"  # Or any folder with .jpg images
    
    print(f"Testing dataset from: {TEST_DATA_PATH}")

    try:
        dataset = ImageDataset(TEST_DATA_PATH)
        if len(dataset) == 0:
            raise Exception(f"No images found in test path. Please update TEST_DATA_PATH in dataset.py or create the folder: {TEST_DATA_PATH}")
        
        # Use torch.utils.data.DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"Batch loaded successfully.")
        print(f"LR image batch shape: {batch['lr'].shape}")
        print(f"HR image batch shape: {batch['hr'].shape}")
        
        # Try to save a preview
        try:
            from torchvision.utils import save_image
            
            # Get the first LR image from the batch and apply the "display" transform
            lr_display = dataset.lr_display_transform(batch['lr'][0])
            
            # Get the first HR image from the batch
            hr_display = batch['hr'][0]
            
            # Stack them side-by-side
            comparison = torch.cat((lr_display, hr_display), dim=2)
            
            save_image(comparison, "dataset_preview.png", normalize=False)
            print("Saved dataset_preview.png. Check this file to see if your LR/HR pair looks correct.")
            print("The left side should be a blurry/pixelated version of the right side.")
            
        except Exception as e:
            print(f"Could not save preview image. Error: {e}")
            print("This is non-critical, but 'torchvision.utils.save_image' is useful.")

    except FileNotFoundError:
        print(f"ERROR: Test data path not found: {TEST_DATA_PATH}")
        print("Please download the DIV2K dataset and place it in that folder,")
        print("or update the TEST_DATA_PATH variable in this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

