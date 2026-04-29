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

echo "=== Playwright Debug & Fix Script ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "System GLIBC: $(ldd --version 2>&1 | head -1)"
echo ""

# ============================================================
# Step 1: 激活 wm conda 环境
# ============================================================
source ~/jyy/anaconda/etc/profile.d/conda.sh
echo "[1] Activating 'wm' conda env..."
conda activate wm
echo "    Python: $(which python)"
echo "    Python version: $(python --version)"
echo "    LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

# ============================================================
# Step 2: 检查 playwright 包版本
# ============================================================
echo "[2] Checking playwright package..."
pip show playwright 2>&1
PLAYWRIGHT_VERSION=$(pip show playwright 2>/dev/null | grep Version | awk '{print $2}')
echo ""

# ============================================================
# Step 3: 检查浏览器二进制文件 (安装前)
# ============================================================
echo "[3] Checking browser binaries BEFORE install..."
if [ -d ~/.cache/ms-playwright ]; then
    ls -la ~/.cache/ms-playwright/
else
    echo "    ~/.cache/ms-playwright/ does not exist — will install now"
fi
echo ""

# ============================================================
# Step 4: 安装 chromium 浏览器
# ============================================================
echo "[4] Running 'python -m playwright install chromium'..."
echo "    This may take a few minutes to download ~150MB..."
python -m playwright install chromium 2>&1
echo "    Install exit code: $?"
echo ""

# ============================================================
# Step 5: 检查浏览器二进制文件 (安装后)
# ============================================================
echo "[5] Checking browser binaries AFTER install..."
ls -la ~/.cache/ms-playwright/ 2>&1
echo ""

# 检查 chromium 主程序是否存在
CHROME_BIN=$(find ~/.cache/ms-playwright -name "chrome" -path "*/chrome-linux/*" 2>/dev/null | head -1)
if [ -n "$CHROME_BIN" ]; then
    echo "    Chrome binary found: $CHROME_BIN"
    echo "    File info: $(file "$CHROME_BIN")"
    echo "    Chrome GLIBC deps:"
    ldd "$CHROME_BIN" 2>&1 | grep -i "not found" && echo "    ^^^ MISSING GLIBC DEPS ABOVE" || echo "    All GLIBC deps OK"
    echo ""

    echo "    Chrome bundled libraries GLIBC check:"
    CHROME_DIR=$(dirname "$CHROME_BIN")
    for lib in "$CHROME_DIR/lib"*".so" "$CHROME_DIR"/*".so"; do
        if [ -f "$lib" ]; then
            MISSING=$(ldd "$lib" 2>&1 | grep -i "not found" || true)
            if [ -n "$MISSING" ]; then
                echo "    MISSING in $lib: $MISSING"
            fi
        fi
    done
    echo "    (empty = all bundled libs OK)"
else
    echo "    WARNING: Chrome binary not found after install!"
fi
echo ""

# ============================================================
# Step 6: 检查 playwright-bundled Node.js
# ============================================================
echo "[6] Checking playwright-bundled Node.js..."
PLAYWRIGHT_NODE=$(python -c "
import playwright, os, glob
pw_dir = os.path.dirname(playwright.__file__)
nodes = glob.glob(os.path.join(pw_dir, 'driver', '**', 'node'), recursive=True)
print(nodes[0] if nodes else 'NOT_FOUND')
")
echo "    Node path: $PLAYWRIGHT_NODE"
if [ "$PLAYWRIGHT_NODE" != "NOT_FOUND" ] && [ -f "$PLAYWRIGHT_NODE" ]; then
    echo "    Node version: $($PLAYWRIGHT_NODE --version 2>&1)"
    echo "    Node GLIBC deps:"
    ldd "$PLAYWRIGHT_NODE" 2>&1 | grep -i "not found" && echo "    ^^^ MISSING DEPS ABOVE" || echo "    All GLIBC deps OK"
else
    echo "    WARNING: bundled node not found"
fi
echo ""

# ============================================================
# Step 7: 安装系统依赖 (如果有权限)
# ============================================================
echo "[7] Attempting to install system dependencies..."
# playwright 需要一些系统级库，尝试通过 playwright 自带的脚本安装
python -m playwright install-deps chromium 2>&1
DEPS_EXIT=$?
if [ $DEPS_EXIT -ne 0 ]; then
    echo "    install-deps failed (exit $DEPS_EXIT) — may need root. Trying conda fallback..."
    # 尝试通过 conda 安装常见的缺失库
    conda install -y -c conda-forge libxcb libx11 nss nspr cups libatk libxcomposite \
        libxrandr libxdamage libxfixes libxshmfence alsa-lib pango atk gtk3 2>&1 || \
        echo "    conda install also failed or packages not available"
fi
echo ""

# ============================================================
# Step 8: 检查是否缺少系统级 .so 文件
# ============================================================
echo "[8] Checking for missing system libraries..."
# 列出 chromium 二进制依赖的所有 not found 库
if [ -n "$CHROME_BIN" ] && [ -f "$CHROME_BIN" ]; then
    ldd "$CHROME_BIN" 2>&1 | grep "not found" > /tmp/pw_missing_libs.txt
    if [ -s /tmp/pw_missing_libs.txt ]; then
        echo "    Missing libraries:"
        cat /tmp/pw_missing_libs.txt
        echo ""
        echo "    If you have root, run: sudo apt-get install <package-for-lib>"
        echo "    Without root, try conda: conda install -c conda-forge <library-name>"
    else
        echo "    No missing libraries — chromium deps look good"
    fi
fi
echo ""

# ============================================================
# Step 9: 测试 import
# ============================================================
echo "[9] Testing 'from playwright.sync_api import sync_playwright'..."
python -c "from playwright.sync_api import sync_playwright; print('    Import OK')" 2>&1
echo ""

# ============================================================
# Step 10: 测试 context manager
# ============================================================
echo "[10] Testing sync_playwright() context manager..."
python -c "
from playwright.sync_api import sync_playwright
p = sync_playwright()
print('    sync_playwright() returned:', type(p))
try:
    pw = p.__enter__()
    print('    __enter__ OK')
    p.__exit__(None, None, None)
    print('    __exit__ OK')
except Exception as e:
    print('    __enter__ FAILED:', type(e).__name__, str(e))
" 2>&1
echo ""

# ============================================================
# Step 11: 完整流程测试
# ============================================================
echo "[11] Full end-to-end test: launch chromium, navigate, screenshot..."
python << 'PYEOF'
import sys
try:
    from playwright.sync_api import sync_playwright
    print("    Import OK")
    with sync_playwright() as p:
        print("    Context manager entered OK")
        browser = p.chromium.launch(headless=True)
        print("    Chromium launched OK")
        page = browser.new_page()
        print("    New page created OK")
        page.goto("data:text/html,<h1 style='font-size:48px'>Hello Playwright!</h1>")
        print("    Page navigated OK")
        page.screenshot(path="/tmp/playwright_test.png")
        print("    Screenshot saved to /tmp/playwright_test.png")
        browser.close()
        print("    === ALL STEPS PASSED ===")
except Exception as e:
    print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
PYEOF
echo ""

echo "=== Debug complete ==="
echo "Full output: logs/debug_playwright_*.out"
echo "Errors:      logs/debug_playwright_*.err"
