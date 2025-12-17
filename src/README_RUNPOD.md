# FLUX Image Generation on Runpod

This guide explains how to run the FLUX image generation script on Runpod using an RTX 3090 GPU.

## Model Configuration

The project now uses `FLUX.1 [dev]` by default, but you can override this via environment variables.

### Setting Model ID

To use a different model, set the `MODEL_ID` environment variable:

```bash
export MODEL_ID="black-forest-labs/FLUX.1-dev"  # Default
# or
export MODEL_ID="black-forest-labs/FLUX.1-schnell"  # For faster generation
```

**Note**: The script will automatically use the model specified in `MODEL_ID` environment variable, defaulting to `FLUX.1-dev`.

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
  --model_id "black-forest-labs/FLUX.1-dev" \
  --prompt "cinematic portrait photo, soft natural light, 85mm lens, shallow depth of field, ultra realistic" \
  --out_dir "src/outputs" \
  --height 768 \
  --width 768 \
  --guidance_scale 3.5 \
  --num_inference_steps 20
```

**Note**: The `--model_id` parameter will override the `MODEL_ID` environment variable if both are set.

### Using LoRA (Low-Rank Adaptation)

**Prerequisites:** Install PEFT library for LoRA support:
```bash
pip install peft>=0.7.0
```

To use custom LoRA weights with FLUX:

```bash
python src/generate.py \
  --prompt "your prompt here" \
  --lora_path "path/to/your_lora_weights.safetensors" \
  --lora_config_path "path/to/your_lora_config.json" \
  --lora_scale 1.0
```

LoRA parameters:
- `--lora_path`: Path to LoRA weights file (.safetensors format)
- `--lora_config_path`: Path to LoRA configuration file (.json format), optional
- `--lora_scale`: Scale factor for LoRA application (default: 1.0, recommended: 0.5-1.5)
- `--lora_trigger_word`: Trigger word for LoRA (automatically added to prompt start), optional

**Important:** If your LoRA was trained with a trigger word (like "character-name"), you MUST include it in the prompt for the LoRA to work properly. Use either:
- Manual: `--prompt "trigger-word, your prompt here"`
- Automatic: `--lora_trigger_word "trigger-word" --prompt "your prompt here"`

**Note:** If PEFT is not installed, the script will show a warning and continue without LoRA.

To use a different output directory:

```bash
python src/generate.py --prompt "your prompt" --out_dir "./my_outputs"
```

## Performance & Resource Usage

### CPU Usage Optimization

The script automatically optimizes CPU thread usage based on your system:
- **Automatic CPU thread management**: Uses 75% of available cores (max 8 threads)
- **PyTorch thread optimization**: Limits intra/inter-op threads for better performance
- **Expected CPU usage**: 60-80% during model loading and generation (normal for FLUX)

### Memory Usage Notes

- **RTX 3090/A6000**: Good for 768×768 images with fp16 precision
- The script uses sequential CPU offload for efficient VRAM management
- Model loads to CPU first (~34GB total), then components move to GPU during inference as needed
- Peak GPU memory usage during inference: ~8-12GB for 768×768 generation
- **RTX A6000 (48GB)**: Can handle larger images (1024×1024) or higher quality settings
- **Memory saving tip**: If you get OOM, try `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### High CPU Usage? This is normal!

If you see 80-95% CPU usage:
- ✅ **This is expected** during FLUX model loading and generation
- ✅ The model uses multiple CPU threads for data preprocessing
- ✅ CPU coordinates between GPU components during inference
- ✅ First run downloads ~10GB model, subsequent runs are faster

### Manual CPU Thread Control

If needed, you can manually set CPU threads:

```bash
# Limit to 4 threads (good for most systems)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Or let the script auto-detect (recommended)
# The script automatically optimizes based on your CPU cores
```

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
ls -la ./test_output/flux_dev.png
```

## Expected Output

The script will:
1. Detect and display GPU information
2. Load the FLUX model (may take 1-2 minutes)
3. Generate the image (20-60 seconds depending on settings)
4. Save to `flux_dev.png` in the output directory (filename based on model)
5. Display the save path

## Memory Management & Model Unloading

### Unloading Models to Free GPU Memory

When switching between models or to free up GPU memory, use the provided unload script:

```bash
# Unload model and clear GPU cache
python scripts/unload_model.py --verbose

# Force unload (useful if script can't detect loaded models)
python scripts/unload_model.py --force --verbose
```

**What this does:**
- Deletes model objects from memory
- Runs garbage collection
- Clears CUDA cache (`torch.cuda.empty_cache()`)
- Shows memory usage before/after (with `--verbose`)

### Runpod Instance Management

**To completely free GPU memory:**
1. **Stop/Terminate the Runpod instance** via Runpod Dashboard - this guarantees all GPU memory is freed
2. **Restart the instance** with new environment variables if needed

**Programmatic instance management:**
- Use Runpod API or CLI tools to terminate/stop instances
- Example: `runpodctl terminate <instance-id>` (if available)

### Memory Monitoring

Check GPU memory usage:
```bash
# NVIDIA GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# PyTorch memory info
python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB')"
```

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
10. **Model switching**: Use `MODEL_ID` environment variable to switch models without code changes
11. **Memory not freed**: Use `python scripts/unload_model.py` or restart instance to free GPU memory

## Cost Estimation

- RTX 3090 on Runpod: ~$0.69/hour
- Typical generation time: 2-5 minutes per image
- Model download: ~5-10 minutes first time
