## 适配working environment
export HF_HOME=/home/wilsxue.adm/.cache
export UV_CACHE_DIR=/data_fast/Alpamayo/uv-cache
export MPLCONFIGDIR=/data_fast/alpamayo2_super_mpl

## usage实验：双卡 Navigation CFG

此 demo 将完整 VLM 放到 cuda:0，专家及 guided/unguided KV cache 放到 cuda:1，
不是将 VLM 分片到两张卡。官方验证配置为两张 80GB H100；当前每张 32GB 的
RTX 5000 Ada 无法容纳第一张卡上的 BF16 VLM，下面命令修正了入口和路径，
但不代表当前硬件能完成推理。第三张卡不会被此 demo 使用。
此入口执行导航 CFG，与普通 inference_smoke 的单提示推理不同。

```bash
cd /home/wilsxue.adm/nvalpamayo2/alpamayo2
source .venv/bin/activate

export ALPAMAYO2_SUPER_MODEL_ID="/home/wilsxue.adm/nvalpamayo2/alpamayo2/nv-community--Alpamayo2-Super/snapshots/master"
export ALPAMAYO2_SUPER_OUTPUT_DIR="$PWD/outputs"
mkdir -p "$ALPAMAYO2_SUPER_OUTPUT_DIR"

set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/two_gpu_nav_cfg_demo.py \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --manifest examples/nav_cfg_validation_samples.json \
  --sample-index 0 \
  --vlm-device cuda:0 \
  --expert-device cuda:1 \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.json" \
  2>&1 | tee "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.log"
```