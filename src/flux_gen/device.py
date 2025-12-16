"""Device detection and reporting for FLUX generation."""


def detect_and_report_device(runtime_config):
    """Detect device capabilities and print system information."""
    import torch

    if runtime_config.has_cuda:
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        return device
    else:
        print("Warning: CUDA GPU not available, falling back to CPU")
        print("Note: CPU inference will be very slow for FLUX models")
        device = torch.device("cpu")
        return device


def report_hf_token_status(runtime_config):
    """Report Hugging Face token status."""
    if not runtime_config.hf_token:
        print("Warning: HF_TOKEN environment variable not set.")
        print("If the model is private, set HF_TOKEN before running:")
        print("export HF_TOKEN=your_huggingface_token_here")
