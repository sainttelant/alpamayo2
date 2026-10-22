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
  --manifest examples/validation_samples.json \
  --sample-index 0 \
  --vlm-device cuda:0 \
  --expert-device cuda:2 \
  --save-viz "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.png" \
  --save-json "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.json" \
  2>&1 | tee "$ALPAMAYO2_SUPER_OUTPUT_DIR/nav_cfg_sample0.log"
```

## 下载示例数据

hf download dgural/PhysicalAI-Autonomous-Vehicles-Sample \
  --repo-type dataset \
  --revision main \
  --local-dir /home/wilsxue.adm/nvalpamayo2/alpamayo2/datasets \
  --max-workers 4 \
  2>&1 | tee /home/wilsxue.adm/nvalpamayo2/alpamayo2/datasets/download.log

## 对本地 Sample MP4 做整段连续推理

相机视频从 `datasets/data` 本地读取；对应 UUID 的真实 ego 历史和评估真值从官方
PhysicalAI-AV 数据读取。默认每 1 秒跑一个窗口，每个窗口包含截止当前时刻的最近
4 帧；模型只加载一次。需要官方数据集访问权限及网络。输出 JSONL 每行对应一个
时刻；同时会在输出目录下自动创建 `<clip_id>/`，每个窗口分别保存 PNG 和 JSON。
视频末尾没有足够未来真值的窗口仍会推理，但没有 ADE。

```bash
cd /home/wilsxue.adm/nvalpamayo2/alpamayo2
source .venv/bin/activate
export HF_HOME=/home/wilsxue.adm/.cache
export ALPAMAYO2_SUPER_MODEL_ID="$PWD/nv-community--Alpamayo2-Super/snapshots/master"
export ALPAMAYO2_SUPER_OUTPUT_DIR="$PWD/outputs"
mkdir -p "$ALPAMAYO2_SUPER_OUTPUT_DIR"

set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u examples/two_gpu_nav_cfg_demo.py \
  --three-gpu \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --local-video-root "$PWD/datasets" \
  --clip-id 808039e3-5efb-4cc2-b845-93c65b99ddf8 \
  --t0-us 2000000 \
  --stride-us 1000000 \
  --output-dir "$ALPAMAYO2_SUPER_OUTPUT_DIR" \
  2>&1 | tee "$ALPAMAYO2_SUPER_OUTPUT_DIR/local_clip_predictions.log"
```

替换 `--clip-id` 为 `datasets/samples.json` 中任一 UUID 可以选择其他 clip。
PNG、JSON 和汇总 JSONL 会保存到
`$ALPAMAYO2_SUPER_OUTPUT_DIR/<clip_id>/`。将 `--stride-us` 改为 `100000`
可每 0.1 秒推理一次；按 30fps 每帧推理会非常耗时。