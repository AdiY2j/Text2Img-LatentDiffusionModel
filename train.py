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

print(torch.__version__)


# Ensure reproducibility
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Tensor is on:", device)

# Define the dataset class
class CustomCocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform):
        self.img_dir = img_dir
        self.transform = transform

        # Load annotations
        with open(ann_file, 'r') as f:
            annotations = json.load(f)

        # Create a mapping from image ID to file name
        self.id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

        # Create a list of image file paths and their corresponding captions
        self.samples = []
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            filename = self.id_to_filename[img_id]
            img_path = os.path.join(self.img_dir, filename)
            self.samples.append((img_path, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, caption

# Define paths to your local MS COCO dataset


img_dir = 'coco17/train2017'  # Replace with your path
ann_file = 'coco17/annotations/captions_train2017.json'  # Replace with your path

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

# Create the dataset and DataLoader
train_dataset = CustomCocoDataset(img_dir, ann_file, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# Test the DataLoader (optional, can be commented out after testing)
# for images, captions in train_loader:
#     print("Batch loaded")
#     for i in range(len(images)):
#         img = images[i].permute(1, 2, 0).numpy()
#         img = (img * 0.5) + 0.5  # Denormalize for visualization
#         plt.imshow(img)
#         plt.title(captions[i])
#         plt.axis('off')
#         plt.show()
#     break  # Remove this after testing

# Load the CompVis Autoencoder
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Load CLIP tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

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
).to(device)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

# Initialize the scheduler
num_timesteps = 1000
scheduler = DDIMScheduler(
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    num_train_timesteps=num_timesteps,
)

# Check for existing checkpoint
checkpoint_dir = 'outputs_train_demo'  # Directory where checkpoints are saved
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint_train.pth')

start_epoch = 0  # Default start epoch

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
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
epochs = 300  # Total number of epochs

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
        "Zebra eating grass",
        "A pizza"
        # Add more prompts if desired
    ]
    with torch.no_grad():
        for prompt in test_prompts:
            generated_images = generate_image(prompt, num_inference_steps=15)
            # Display or save the image
            plt.imshow(generated_images[0])
            plt.title(f'Epoch {epoch+1} - {prompt}')
            plt.axis('off')
            plt.show()
            # Optionally save the image
            image_save_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_{prompt.replace(" ", "_")}.png')
            plt.savefig(image_save_path)
            plt.close()
    torch.cuda.empty_cache()

