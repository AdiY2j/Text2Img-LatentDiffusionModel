import torch
import streamlit as st
import os
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = 'outputs_v1'  # Directory where checkpoints are saved
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint_attn_8_ep_250.pth')


@st.cache_resource
def load_trained_model(checkpoint_path=checkpoint_path):
    # Load the CompVis Autoencoder
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Load CLIP tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)


    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

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
    )

    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device)
    unet.eval()
    return unet, vae, tokenizer, text_encoder

# Load the model
unet, vae, tokenizer, text_encoder = load_trained_model()

def generate_image(prompt, num_inference_steps=7):
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
        num_train_timesteps=1000,
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
    image = (images[0] * 255).astype(np.uint8)
    plt.imshow(images[0])
    plt.title(f'{prompt}')
    plt.axis('off')
    # Optionally save the image
    plt.savefig(f"tmp.png")
    plt.show()
    return Image.fromarray(image)

# Streamlit UI
st.title("Latent Diffusion Model - Text-to-Image Generator")
prompt = st.text_input("Enter a text prompt:", placeholder="A futuristic city at night")

if st.button("Generate Image") and prompt:
    st.write("Generating Image... Please wait.")
    img = generate_image(prompt)
    st.image(img, caption="Generated Image")
