import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionPipeline
import gradio as gr

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load lightweight GPT-2 model for script generation
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

# Load Stable Diffusion with optimizations for 6GB VRAM
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.enable_attention_slicing()  # Reduce memory usage by slicing attention computation

# Function to generate a movie script from user prompt
def generate_script(prompt):
    # Tokenize input prompt and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate script with sampling for creativity
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, do_sample=True)
    # Decode generated tokens to readable text
    script = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return script

# Function to generate a key frame image from the script
def generate_frame(script):
    # Generate image with lower resolution (384x384) to fit 6GB VRAM
    image = pipe(script, height=384, width=384, num_inference_steps=30).images[0]
    return image

# Main function to create script and frame for Gradio
def mini_movie(prompt):
    # Generate script from user input
    script = generate_script(prompt)
    # Generate corresponding key frame
    frame = generate_frame(script)
    # Clear GPU memory to avoid overflow
    torch.cuda.empty_cache()
    return script, frame

# Set up Gradio interface
interface = gr.Interface(
    fn=mini_movie,
    inputs=gr.Textbox(label="Enter your movie idea"),
    outputs=[gr.Textbox(label="Script"), gr.Image(label="Key Frame")],
    title="Mini Movie Director",
    description="Generate a short movie script and key frame from your idea using GPT-2 and Stable Diffusion."
)

# Launch the interface
interface.launch(inbrowser=True)