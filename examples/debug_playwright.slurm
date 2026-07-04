#!/bin/bash
#SBATCH --job-name=debug_playwright
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/debug_playwright_%j.out
#SBATCH --error=logs/debug_playwright_%j.err

echo "=== Playwright Version & Compatibility Check ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate wmr1  # 或你用来装 Playwright 的环境

echo "--- System Info ---"
echo "System GLIBC: $(ldd --version 2>&1 | head -1)"
echo "Kernel: $(uname -r)"
echo ""

echo "--- Playwright Python Package ---"
python -c "
import playwright
print(f'Version: {playwright.__version__}')
print(f'Location: {playwright.__file__}')
"
echo ""

echo "--- Browser Binaries ---"
if [ -d ~/.cache/ms-playwright ]; then
    ls -la ~/.cache/ms-playwright/
    CHROME=$(find ~/.cache/ms-playwright -name "chrome" -path "*/chrome-linux/*" 2>/dev/null | head -1)
    if [ -f "$CHROME" ]; then
        echo "chrome binary: $CHROME"
        ldd "$CHROME" 2>&1 | grep "not found" && echo ">>> MISSING libs" || echo ">>> All libs OK"
    fi
else
    echo "~/.cache/ms-playwright/ does NOT exist"
    echo ">>> Run: playwright install chromium"
fi
echo ""

# End-to-End Test
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
    echo "SKIP: Chromium not installed"
    echo "Run: conda activate wmr1 && playwright install chromium"
fi
