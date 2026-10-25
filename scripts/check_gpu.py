"""Require CUDA and execute a small forward/backward check; no model is trained."""

import json
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is missing. Install with: uv sync --frozen --extra data --extra gpu")
        return 1
    if not torch.cuda.is_available():
        print("CUDA is required but unavailable. Check the NVIDIA driver and the gpu extra.")
        return 1

    device = torch.device("cuda:0")
    torch.manual_seed(3415)
    # Small synthetic tensors exercise CUDA, autograd and mixed precision only.
    # This is a runtime check, not training or evaluation of a sentiment model.
    layer = torch.nn.Linear(128, 32).to(device)
    inputs = torch.randn(16, 128, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = layer(inputs)
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not output.is_cuda or layer.weight.grad is None:
        raise RuntimeError("The verification computation did not execute on CUDA")
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(layer.weight.grad).all()):
        raise RuntimeError("CUDA verification produced nonfinite output or gradients")
    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "status": "passed",
                "python": sys.version.split()[0],
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "vram_gib": round(properties.total_memory / 2**30, 2),
                "device": str(output.device),
                "mixed_precision_forward_and_backward": "passed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
