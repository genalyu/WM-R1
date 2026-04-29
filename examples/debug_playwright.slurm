#!/bin/bash
#SBATCH --job-name=debug_playwright
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/debug_playwright_%j.out
#SBATCH --error=logs/debug_playwright_%j.err

echo "=== Playwright Debug Script ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# === Step 1: 激活 wm conda 环境 ===
source ~/jyy/anaconda/etc/profile.d/conda.sh
echo "[1] Activating 'wm' conda env..."
conda activate wm
echo "    Python: $(which python)"
echo "    Python version: $(python --version)"
echo ""

# === Step 2: 检查 playwright 包 ===
echo "[2] Checking playwright package..."
pip show playwright 2>&1 || echo "    ERROR: playwright package not installed!"
echo ""

# === Step 3: 检查 playwright 浏览器二进制文件 ===
echo "[3] Checking browser binaries..."
echo "    ~/.cache/ms-playwright contents:"
ls -la ~/.cache/ms-playwright/ 2>&1 || echo "    WARNING: ~/.cache/ms-playwright/ does not exist!"
echo ""

# === Step 4: 检查 Playwright 是否依赖的 Node.js ===
echo "[4] Checking playwright-bundled Node.js..."
# playwright bundles its own node binary
PLAYWRIGHT_NODE=$(python -c "
import playwright
import os
pw_dir = os.path.dirname(playwright.__file__)
# Try to find the node binary in playwright's driver
import glob
node_candidates = glob.glob(os.path.join(pw_dir, 'driver', '**', 'node'), recursive=True)
if node_candidates:
    print(node_candidates[0])
else:
    print('NOT_FOUND')
")
echo "    Playwright node path: $PLAYWRIGHT_NODE"
if [ "$PLAYWRIGHT_NODE" != "NOT_FOUND" ] && [ -f "$PLAYWRIGHT_NODE" ]; then
    echo "    Node version: $($PLAYWRIGHT_NODE --version 2>&1)"
    echo "    GLIBC deps:"
    ldd "$PLAYWRIGHT_NODE" 2>&1 | grep -i "not found" || echo "    All GLIBC deps OK"
else
    echo "    WARNING: Could not find bundled node binary"
fi
echo ""

# === Step 5: 尝试 import playwright ===
echo "[5] Testing 'from playwright.sync_api import sync_playwright'..."
python -c "from playwright.sync_api import sync_playwright; print('    Import OK')" 2>&1
echo ""

# === Step 6: 尝试启动 sync_playwright ===
echo "[6] Testing sync_playwright() context manager..."
python -c "
from playwright.sync_api import sync_playwright
p = sync_playwright()
print('    sync_playwright() returned:', type(p))
print('    dir(p):', dir(p))
try:
    pw = p.__enter__()
    print('    __enter__ OK')
    p.__exit__(None, None, None)
    print('    __exit__ OK')
except Exception as e:
    print('    __enter__ FAILED:', type(e).__name__, str(e))
" 2>&1
echo ""

# === Step 7: 尝试完整流程 (launch chromium, screenshot) ===
echo "[7] Full end-to-end test: launch chromium and take a screenshot..."
python << 'PYEOF'
import sys
try:
    from playwright.sync_api import sync_playwright
    print("    Import OK")
    with sync_playwright() as p:
        print("    Context manager entered")
        browser = p.chromium.launch(headless=True)
        print("    Chromium launched")
        page = browser.new_page()
        page.goto("data:text/html,<h1>Hello</h1>")
        page.screenshot(path="/tmp/playwright_test.png")
        print("    Screenshot saved to /tmp/playwright_test.png")
        browser.close()
        print("    Browser closed - SUCCESS!")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
PYEOF
echo ""

echo "=== Debug complete ==="
echo "Check logs/debug_playwright_*.out for full output"
echo "Check logs/debug_playwright_*.err for stderr"
