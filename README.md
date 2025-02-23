# Mini Movie Director

## Introduction
Mini Movie Director is a creative AI tool that transforms user-provided story ideas into short movie scripts and visual key frames. It leverages GPT-2 for script generation and Stable Diffusion for image creation, showcasing the power of Large Language Models (LLMs) and generative AI. Built with PyTorch and Gradio in an Anaconda environment, it’s optimized to run on a 6GB GPU, making it accessible yet powerful. This project demonstrates my expertise in LLMs, multimodal AI, and resource-efficient implementation.

## Features
- **Script Generation**: Creates a short movie script from a user prompt using GPT-2.
- **Key Frame Creation**: Generates a visual key frame from the script with Stable Diffusion.
- **GPU Optimization**: Runs on a 6GB GPU with memory-efficient settings (FP16, attention slicing).
- **Interactive UI**: Powered by Gradio for an easy-to-use web interface.
- **Creative Output**: Combines text and image generation for a mini movie experience.

## Technical Highlights
This project showcases my understanding of LLMs and generative AI:

1. **LLM for Text Generation**:
   - Uses DistilGPT-2, a lightweight LLM, to generate coherent scripts, demonstrating efficient model selection for limited hardware.

2. **Multimodal AI**:
   - Integrates GPT-2 (text) with Stable Diffusion (image), highlighting the synergy between LLMs and vision models, akin to CLIP’s contrastive learning.

3. **Generative Optimization**:
   - Employs FP16 precision and attention slicing in Stable Diffusion to fit a 6GB GPU, showing practical knowledge of memory management in AI.

4. **Prompt-to-Image Pipeline**:
   - Converts natural language prompts into visual outputs, leveraging Stable Diffusion’s latent diffusion process guided by text embeddings.

5. **Scalable Design**:
   - Built with modularity, allowing future expansion (e.g., TTS or video synthesis), reflecting forward-thinking in AI application design.

## Installation

### Prerequisites
- Anaconda (latest version recommended)
- Python 3.9+
- NVIDIA GPU with 6GB VRAM (e.g., GTX 1660, RTX 2060)
- CUDA Toolkit 11.8 (or compatible with your GPU)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AzuCai/MiniMovieDirector.git
   cd MiniMovieDirector
   ```
2. **Create Anaconda Environment**:
   ```bash
   conda create -n minimovie python=3.9 -y
   conda activate minimovie
   ```
3. **Install Dependencies**:
   ```bash
   conda install pytorch==2.6.0 torchvision==0.21.0 cudatoolkit=11.8 -c pytorch -c conda-forge -y
   pip install transformers diffusers accelerate gradio numpy
   ```
4. **Verify Installation**:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
## Usage
1. **Clone the Repository**:
   ```bash
   conda activate minimovie
   python mini_movie.py
   ```

2. **Open the browser URL (e.g., http://127.0.0.1:7860).**
3. **Enter a movie idea (e.g., "A pirate ship in a storm").**
4. **View the generated script and key frame.**

## Future Enhancements
  Add Text-to-Speech (TTS) for narrated scripts using lightweight models.

  Extend to video generation by interpolating key frames with tools like FFmpeg.

  Integrate a larger LLM (e.g., GPT-3) for richer scripts if VRAM permits.

## Contributing
  Contributions are welcome! Submit issues or pull requests to enhance functionality or optimize performance.

## License
  This project is licensed under the MIT License.

## Acknowledgments
  Built with PyTorch, Transformers, Diffusers, and Gradio. 

  Powered by DistilGPT-2 and Stable Diffusion, showcasing cutting-edge generative AI.
