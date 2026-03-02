#!/bin/bash
set -e

WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
WHEEL_NAME="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
WHEEL_PATH="/tmp/${WHEEL_NAME}"

echo "Downloading flash-attn 2.8.3 wheel..."
python -c "
import urllib.request
urllib.request.urlretrieve('${WHEEL_URL}', '${WHEEL_PATH}')
"

echo "Installing flash-attn 2.8.3..."
pip install "${WHEEL_PATH}"

rm -f "${WHEEL_PATH}"

echo "Verifying installation..."
python -c "
from flash_attn import __version__
assert __version__ >= '2.6.0', f'Expected >= 2.6.0, got {__version__}'
print(f'flash-attn {__version__} installed successfully')
"
