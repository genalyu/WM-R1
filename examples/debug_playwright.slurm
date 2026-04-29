#!/bin/bash
#SBATCH --job-name=debug_playwright
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/debug_playwright_%j.out
#SBATCH --error=logs/debug_playwright_%j.err

echo "=== Playwright Version & Compatibility Check ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

source ~/jyy/anaconda/etc/profile.d/conda.sh
conda activate wm

echo "--- System Info ---"
echo "System GLIBC: $(ldd --version 2>&1 | head -1)"
echo "Kernel: $(uname -r)"
echo "Home dir: $HOME"
echo ""

echo "--- Playwright Python Package ---"
python -c "
import playwright
print(f'Version: {playwright.__version__}')
print(f'Location: {playwright.__file__}')
"
echo ""

echo "--- Bundled Driver (Node.js) ---"
python -c "
import playwright, os, glob
pw_dir = os.path.dirname(playwright.__file__)
nodes = glob.glob(os.path.join(pw_dir, 'driver', '**', 'node'), recursive=True)
driver_dir = os.path.join(pw_dir, 'driver')
print(f'Driver dir: {driver_dir}')
print(f'Files: {os.listdir(driver_dir)}')
print(f'Node: {nodes[0] if nodes else \"NOT_FOUND\"}')
"
echo ""

PLAYWRIGHT_NODE=$(python -c "
import playwright, os, glob
pw_dir = os.path.dirname(playwright.__file__)
nodes = glob.glob(os.path.join(pw_dir, 'driver', '**', 'node'), recursive=True)
print(nodes[0] if nodes else 'NOT_FOUND')
")
if [ -f "$PLAYWRIGHT_NODE" ]; then
    echo "Node version: $($PLAYWRIGHT_NODE --version)"
    echo "Node GLIBC check:"
    ldd "$PLAYWRIGHT_NODE" 2>&1 | grep "not found" && echo ">>> MISSING" || echo ">>> All OK"
fi
echo ""

echo "--- Browser Binaries ---"
if [ -d ~/.cache/ms-playwright ]; then
    echo "~/.cache/ms-playwright contents:"
    ls -la ~/.cache/ms-playwright/

    for dir in ~/.cache/ms-playwright/chromium-*; do
        if [ -d "$dir" ]; then
            echo ""
            echo "Found: $dir"
            CHROME="$dir/chrome-linux/chrome"
            if [ -f "$CHROME" ]; then
                echo "  chrome binary: EXISTS"
                echo "  GLIBC check:"
                ldd "$CHROME" 2>&1 | grep "not found" && echo "  >>> MISSING" || echo "  >>> All OK"
            else
                echo "  chrome binary: MISSING at $CHROME"
                echo "  Contents:"
                ls -la "$dir/" 2>/dev/null
            fi
        fi
    done
else
    echo "~/.cache/ms-playwright/ does NOT exist"
    echo ">>> Need to run: playwright install chromium"
fi
echo ""

echo "--- Version Compatibility Summary ---"
PW_VER=$(python -c "import playwright; print(playwright.__version__)")
echo "Playwright Python: $PW_VER"
echo "Expected chromium revision: 878941"
echo "System GLIBC: $(ldd --version 2>&1 | head -1 | awk '{print $NF}')"
echo ""
echo "Known: Playwright 1.11.1 bundles chromium r878941"
echo "  chromium r878941 requires GLIBC >= 2.17 (RHEL/CentOS 7 baseline)"
echo "  Your system GLIBC 2.17 => should be COMPATIBLE"
echo ""

# Final test
echo "--- End-to-End Test ---"
CHROME=$(find ~/.cache/ms-playwright -name "chrome" -path "*/chrome-linux/*" 2>/dev/null | head -1)
if [ -f "$CHROME" ]; then
    python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page()
    pg.goto('data:text/html,<h1>OK</h1>')
    pg.screenshot(path='/tmp/pw_test.png')
    b.close()
    print('SUCCESS: Playwright is working!')
" 2>&1
else
    echo "SKIP: Chromium not installed yet"
    echo ""
    echo "Run this on login node (has internet):"
    echo "  conda activate wm"
    echo "  playwright install chromium"
fi
