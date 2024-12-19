import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import random
from torch.cuda.amp import autocast, GradScaler

print(torch.__version__)

# Ensure reproducibility
random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Tensor is on:", device)

# Define the dataset class for Flickr30k with Karpathy splits
class CustomFlickr30kDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform, split="train", subset_fraction=1.0):
        """
        img_dir: directory containing Flickr30k images
        ann_file: path to dataset_flickr30k.json (Karpathy splits)
        transform: image transformations
        split: "train", "val", or "test"
        subset_fraction: fraction of the split to use (1.0 = use all)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.split = split.lower()
        self.log_file = f'dataset_info_{self.split}.txt'

        # Clear the log file if it exists
        open(self.log_file, 'w').close()

        # Load Karpathy annotation file
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Collect samples for the specified split
        samples = []
        for img_data in annotations['images']:
            if img_data['split'] == self.split:
                filename = img_data['filename']
                img_path = os.path.join(self.img_dir, filename)
                
                # Check if image file exists; if missing, skip
                if not os.path.isfile(img_path):
                    continue

                # Instead of taking all sentences, just take one caption per image
                # For simplicity, take the first sentence
                if len(img_data['sentences']) > 0:
                    caption = img_data['sentences'][0]['raw']
                    samples.append((img_path, caption))

        # Log the total samples before subset selection
        self.write_log(f"Split: {self.split}\n")
        self.write_log(f"Total samples before subset: {len(samples)}\n")

        # Apply subset fraction if needed
        subset_size = int(len(samples) * subset_fraction)
        samples = random.sample(samples, subset_size)
        self.samples = samples

        self.write_log(f"Subset fraction: {subset_fraction}\n")
        self.write_log(f"Subset size: {subset_size}\n")

        # Write sample paths and captions to log
        for img_path, caption in self.samples:
            self.write_log(f"Image Path: {img_path} | Caption: {caption}\n")

    def write_log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        # Check at runtime if file is missing
        if not os.path.isfile(img_path):
            print(f"Warning: Image file {img_path} not found during getitem. Skipping sample.")
            return None, None

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, caption
        

# Set paths to Flickr30k dataset and Karpathy annotation file
img_dir = '/home/aditya/demo/data/flickr/flickr30k-images'  # Update this path
ann_file = '/home/aditya/demo/data/flickr/dataset.json'  # Update this path

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

# Create train, val, test datasets and dataloaders
train_dataset = CustomFlickr30kDataset(img_dir, ann_file, transform, split="train", subset_fraction=0.5)
val_dataset = CustomFlickr30kDataset(img_dir, ann_file, transform, split="val", subset_fraction=1.0)
test_dataset = CustomFlickr30kDataset(img_dir, ann_file, transform, split="test", subset_fraction=1.0)

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

# Load the CompVis Autoencoder
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Load CLIP tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)


# Define the projection layer for text embeddings
# class TextProjection(nn.Module):
#     def __init__(self, input_dim=768, output_dim=1280):
#         super(TextProjection, self).__init__()
#         self.proj = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         return self.proj(x)

# # Initialize the text projection layer
# text_projection = TextProjection().to(device)
# nn.init.xavier_uniform_(text_projection.proj.weight)
# nn.init.zeros_(text_projection.proj.bias)

# Define the UNet model with cross-attention
unet = UNet2DConditionModel(
    sample_size=32,  # Adjust as needed
    in_channels=4,   # Latent channels
    out_channels=4,  # Latent channels
    layers_per_block=2,
    block_out_channels=(320, 640, 1280, 1280),  # Channel multiplier: 1, 2, 4, 4
    down_block_types=(
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D", "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
    ),
    cross_attention_dim=768,  # CLIP embedding dimension
    attention_head_dim = 8, # No. of heads as per architecture
    # act_fn = "silu",
    # flip_sin_to_cos = True,
    # norm_eps = 1e-05,
    # norm_num_groups=32,
    # downsample_padding=1,
    # freq_shift=0,
    # mid_block_scale_factor=1,
    # center_input_sample=False
).to(device)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# Initialize the scheduler
num_timesteps = 1000
scheduler = DDIMScheduler(
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    clip_sample=False,
    num_train_timesteps=num_timesteps,
)


# Check for existing checkpoint
checkpoint_dir = 'outputs'  # Directory where checkpoints are saved
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'ldm_flickr30k_checkpoint.pth')
# load_dir = "outputs"
# load_path = os.path.join(load_dir, 'ldm_flickr30k_checkpoint.pth')
start_epoch = 0

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

# Perform inference with text prompt
def generate_image(prompt, num_inference_steps=50):
    # Get text embeddings
    inputs = tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state

    # Scheduler
    inference_scheduler = DDIMScheduler(
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        num_train_timesteps=num_timesteps,
    )
    inference_scheduler.set_timesteps(num_inference_steps)
    inference_scheduler.timesteps = inference_scheduler.timesteps.to(device)

    # Start from random noise
    latents = torch.randn((1, unet.in_channels, 32, 32), device=device)
    latents = latents * inference_scheduler.init_noise_sigma

    for t in tqdm(inference_scheduler.timesteps, desc='Denoising'):
        latent_model_input = inference_scheduler.scale_model_input(latents, t)

        with torch.no_grad():
            if not torch.is_tensor(t):
                t = torch.tensor([t], dtype=torch.long, device=device)
            elif t.device != device:
                t = t.to(device)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to images
    with torch.no_grad():
        images = vae.decode(latents / 0.18215).sample

    # Post-process images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    return images


# Training loop
epochs = 400  # Total number of epochs

for epoch in range(start_epoch, epochs):
    total_loss = 0.0
    num_batches = 0
    unet.train()
    print(f"Epoch {epoch+1}/{epochs}")
    for images, captions in tqdm(train_loader):
        images = images.to(device)

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215  # Scaling factor

        # Get text embeddings
        inputs = tokenizer(list(captions), padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            # text_embeddings = text_projection(text_embeddings)  # Project to 1280 dimensions

        # Sample random timesteps and add noise using scheduler
        batch_size = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise using UNet
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

    # Compute average loss
    average_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} Average Loss: {average_loss:.6f}")

    # Save the model checkpoint every epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

    # Save a separate checkpoint every 5 epochs
#     if (epoch + 1) % 5 == 0:
#         checkpoint_path_epoch = os.path.join(checkpoint_dir, f'unet_epoch_{epoch+1}.pth')
#         torch.save(checkpoint, checkpoint_path_epoch)
#         print(f"Checkpoint saved for epoch {epoch+1}")

    # Generate images for evaluation
    unet.eval()
    test_prompts = [
        "Young Los Angeles Lakers fans stand on top of a car waving Lakers banners in the middle of a crowd of fans."
        # Add more prompts if desired
    ]
    with torch.no_grad():
        for prompt in test_prompts:
            generated_images = generate_image(prompt, num_inference_steps=10)
            # Display or save the image
            plt.imshow(generated_images[0])
            plt.title(f'Epoch {epoch+1} - {prompt}')
            plt.axis('off')
            # Optionally save the image
            plt.savefig(f"outputs/epoch_{epoch+1}.png")
            plt.show()
    # torch.cuda.empty_cache()

