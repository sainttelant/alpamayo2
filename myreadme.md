## 适配working environment
export HF_HOME=/home/wilsxue.adm/.cache
export UV_CACHE_DIR=/data_fast/Alpamayo/uv-cache
export MPLCONFIGDIR=/data_fast/alpamayo2_super_mpl

## usage实验：三卡 Navigation CFG

使用新增的 --three-gpu 模式，将 VLM 的 64 层按 24/26/14 分到三张卡。
视觉编码器、输入嵌入和输出头在 cuda:0，专家及两份 CFG KV cache 在 cuda:2。
保持 BF16，不量化、不做 CPU offload。默认 sample 0 已在本机三张 32GB
RTX 5000 Ada 上完成真实推理，PyTorch 峰值显存约 27.82/25.32/20.71 GiB。
这些是该样本的实测值，其他输入或 GPU 上的其他任务可能增加显存需求。
此入口执行导航 CFG，与普通 inference_smoke 的单提示推理不同。

```bash
cd /home/wilsxue.adm/nvalpamayo2/alpamayo2
source .venv/bin/activate

export ALPAMAYO2_SUPER_MODEL_ID="/home/wilsxue.adm/nvalpamayo2/alpamayo2/nv-community--Alpamayo2-Super/snapshots/master"
export ALPAMAYO2_SUPER_OUTPUT_DIR="$PWD/outputs"
mkdir -p "$ALPAMAYO2_SUPER_OUTPUT_DIR"

set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u examples/two_gpu_nav_cfg_demo.py \
  --three-gpu \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --manifest examples/nav_cfg_validation_samples.json \
  --sample-index 0 \
  --vlm-device cuda:0 \
  --expert-device cuda:2 \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.json" \
  2>&1 | tee "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.log"
```