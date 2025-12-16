# FLUX Image Generation on Runpod

This guide explains how to run the FLUX image generation script on Runpod using an RTX 3090 GPU.

## Recommended Runpod Configuration

- **GPU**: RTX 3090 (24GB VRAM) - sufficient for 768×768 generation with fp16
- **vCPUs**: 4-8 cores recommended
- **RAM**: 32GB minimum
- **Storage**: 50GB+ (models can be large)
- **Container Image**: Use a PyTorch CUDA image (e.g., `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`)

## Setup Instructions

### 1. Environment Setup

First, install the dependencies:

```bash
pip install -r requirements.txt
```

**Important**: FLUX requires additional tokenization libraries. If you encounter tokenizer errors, install them with:

```bash
pip install protobuf sentencepiece
```

The script automatically handles dtype parameters and sets attention backend for compatibility.

**Important**: Make sure the PyTorch wheel matches your Runpod instance's CUDA version. Check with:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 2. Hugging Face Token Setup

If using a private model, set your Hugging Face token:

```bash
export HF_TOKEN=your_huggingface_token_here
```

The script automatically detects and uses the HF_TOKEN environment variable.

Get your token from: https://huggingface.co/settings/tokens

### 3. Run the Script

Basic usage:

```bash
python src/generate.py --prompt "your prompt here"
```

Full command with all options (showing defaults):

```bash
python src/generate.py \
  --model_id "black-forest-labs/FLUX.1-schnell" \
  --prompt "cinematic portrait photo, soft natural light, 85mm lens, shallow depth of field, ultra realistic" \
  --out_dir "src/outputs" \
  --height 768 \
  --width 768 \
  --guidance_scale 3.5 \
  --num_inference_steps 20
```

To use a different output directory:

```bash
python src/generate.py --prompt "your prompt" --out_dir "./my_outputs"
```

## Memory Usage Notes

- **RTX 3090/A6000**: Good for 768×768 images with fp16 precision
- The script uses CPU loading with `enable_model_cpu_offload()` for efficient VRAM management
- Model loads to CPU first (~34GB total), then components move to GPU during inference as needed
- Peak GPU memory usage during inference: ~8-12GB for 768×768 generation
- **RTX A6000 (48GB)**: Can handle larger images (1024×1024) or higher quality settings
- **Memory saving tip**: If you get OOM, try `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Troubleshooting Memory Issues

If you encounter OOM errors:
1. Reduce image size: `--height 512 --width 512`
2. Increase inference steps gradually: `--num_inference_steps 15` (fewer steps = faster but lower quality)
3. Check available VRAM: `nvidia-smi`

## Smoke Test

To verify everything works:

```bash
# Quick test with minimal settings
python src/generate.py \
  --prompt "simple test image" \
  --height 512 \
  --width 512 \
  --num_inference_steps 10 \
  --out_dir "./test_output"

# Check if image was created
ls -la ./test_output/flux_schnell.png
```

## Expected Output

The script will:
1. Detect and display GPU information
2. Load the FLUX model (may take 1-2 minutes)
3. Generate the image (20-60 seconds depending on settings)
4. Save to `flux_schnell.png` in the output directory
5. Display the save path

## Performance Tips

- **Faster generation**: Reduce `--num_inference_steps` to 15-20
- **Higher quality**: Increase `--guidance_scale` to 4.0-5.0
- **Batch generation**: The current script generates one image at a time
- **Model caching**: Models are cached locally, subsequent runs will be faster

## Common Issues

1. **"CUDA GPU not found"**: You're running on CPU-only instance
2. **"401 Unauthorized"**: Set `HF_TOKEN` for private models
3. **Device mapping errors**: The script uses `device_map="cuda"` which loads the model directly to GPU. FluxPipeline supports only "balanced" and "cuda" device maps. If you encounter issues, try updating `diffusers` and `accelerate` to the latest versions
4. **OOM Error**: Reduce image size (`--height 512 --width 512`) or decrease inference steps (`--num_inference_steps 15`)
5. **CUDA OOM during loading**: Try `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or restart the instance
6. **Protobuf/tokenizer errors**: Install `pip install protobuf sentencepiece` - required for FLUX tokenizers
7. **Attention/GQA errors**: The script sets `DIFFUSERS_FORCE_ATTENTION_BACKEND=math` for compatibility - if issues persist, try updating PyTorch to 2.5+
8. **Deprecation warnings**: The script uses compatible parameters - warnings can be ignored as they don't affect functionality
9. **Slow loading**: First run downloads model (~10GB), subsequent runs are faster

## Cost Estimation

- RTX 3090 on Runpod: ~$0.69/hour
- Typical generation time: 2-5 minutes per image
- Model download: ~5-10 minutes first time
