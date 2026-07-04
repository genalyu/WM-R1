#!/usr/bin/env python3
"""Check all dependencies for WM-R1 training.

Usage:
    python scripts/check_deps.py
"""
import importlib
import sys
import subprocess

CHECKS = [
    # (import_name, package_name, required, notes)
    ("torch", "torch", True, "PyTorch >= 2.1, CUDA support required"),
    ("vllm", "vllm", True, "vLLM >= 0.6 for rollout"),
    ("flash_attn", "flash-attn", True, "Flash Attention 2"),
    ("ray", "ray", True, "Ray >= 2.10 for distributed training"),
    ("transformers", "transformers", True, "HuggingFace Transformers"),
    ("accelerate", "accelerate", True, "HuggingFace Accelerate"),
    ("datasets", "datasets", True, "HuggingFace Datasets"),
    ("qwen_vl_utils", "qwen-vl-utils", True, "Qwen VL image processing"),
    ("omegaconf", "omegaconf", True, "Config management"),
    ("psutil", "psutil", True, "System utilities"),
    ("PIL", "Pillow", True, "Image processing"),
    ("numpy", "numpy", True, "Numerical computing"),
    ("pandas", "pandas", True, "Data processing (parquet)"),
    ("wandb", "wandb", False, "Logging (optional)"),
    ("mlflow", "mlflow", False, "Logging (optional)"),
    ("swanlab", "swanlab", False, "Logging (optional)"),
    ("tensordict", "tensordict", True, "TensorDict for data"),
    ("torchdata", "torchdata", True, "Torch data utilities"),
    ("liger_kernel", "liger-kernel", False, "Fused kernels (optional)"),
    ("mathruler", "mathruler", False, "Math evaluation (optional)"),
    ("codetiming", "codetiming", False, "Timing utilities (optional)"),
    ("protobuf", "protobuf", True, "Serialization >= 4.21"),
]

def check_cuda():
    """Check CUDA availability and version."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        return True, f"{gpu_name}, {mem:.0f}GB, CUDA {torch.version.cuda}"
    except Exception as e:
        return False, str(e)

def check_flash_attn():
    """Check flash attention version."""
    try:
        import flash_attn
        return True, f"version {flash_attn.__version__}"
    except:
        return False, "not installed"

def check_vllm():
    """Check vLLM version."""
    try:
        import vllm
        return True, f"version {vllm.__version__}"
    except:
        return False, "not installed"

def check_playwright():
    """Check Playwright for WM rendering."""
    try:
        import playwright
        return True, f"version {playwright.__version__}"
    except:
        return False, "not installed (needed for WM HTML rendering)"

def main():
    print("=" * 60)
    print("WM-R1 Dependency Check")
    print("=" * 60)

    # CUDA check
    ok, msg = check_cuda()
    status = "✓" if ok else "✗"
    print(f"\n  {status} CUDA: {msg}")

    # Package checks
    print(f"\n{'Package':<25} {'Status':<6} {'Details'}")
    print("-" * 60)

    missing_required = []
    missing_optional = []

    for import_name, pkg_name, required, notes in CHECKS:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  ✓ {pkg_name:<23} {version}")
        except ImportError:
            if required:
                print(f"  ✗ {pkg_name:<23} MISSING (required)")
                missing_required.append(pkg_name)
            else:
                print(f"  - {pkg_name:<23} not installed (optional)")
                missing_optional.append(pkg_name)

    # Special checks
    print(f"\nSpecial checks:")
    ok, msg = check_flash_attn()
    print(f"  {'✓' if ok else '✗'} Flash Attention: {msg}")

    ok, msg = check_vllm()
    print(f"  {'✓' if ok else '✗'} vLLM: {msg}")

    ok, msg = check_playwright()
    print(f"  {'✓' if ok else '✗'} Playwright: {msg}")

    # Check Playwright chromium
    try:
        result = subprocess.run(
            ["playwright", "install", "--dry-run", "chromium"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  ✓ Playwright Chromium: installed")
        else:
            print(f"  ✗ Playwright Chromium: run 'playwright install chromium'")
    except:
        print(f"  ? Playwright Chromium: could not check")

    # Summary
    print("\n" + "=" * 60)
    if missing_required:
        print(f"MISSING REQUIRED ({len(missing_required)}):")
        print(f"  pip install {' '.join(missing_required)}")
    else:
        print("All required packages installed ✓")

    if missing_optional:
        print(f"\nOptional packages not installed ({len(missing_optional)}):")
        print(f"  {', '.join(missing_optional)}")

    print("=" * 60)

    # Quick install command
    if missing_required:
        print("\nQuick fix:")
        print(f"  pip install {' '.join(missing_required)}")

    return len(missing_required) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
