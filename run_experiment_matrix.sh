#!/usr/bin/env bash

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="SeqSNN"
WANDB_PROJECT="HybridSNN"
WANDB_ENABLED="true"
WANDB_MODE="online"
SEEDS=(40 41 42)
HORIZONS=(6 24)
DATASETS=(metr-la electricity shd)
ENCODERS=(delta_conv gaf merged)
GPUS=(0 1 2 3)
SLOTS_PER_GPU=1
MAX_RETRIES=2
NUM_WORKERS=0
SKIP_EXISTING="true"
SHD_ONCE="true"
RUN_PREFIX=""
RUN_ROOT="/projects/SNN/leo_workspace/saved_stuff/HybridSNN_saved"
LOG_DIR="/projects/SNN/leo_workspace/saved_stuff/HybridSNN_saved"
WAIT_FOR_FREE_GPU="true"
GPU_FREE_MEM_MIN_MB_PER_SLOT=12000
CONFIG_OVERRIDE=""
OVERRIDES=()

parse_csv() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-enabled) WANDB_ENABLED="$2"; shift 2 ;;
        --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
        --seeds) parse_csv "$2" SEEDS; shift 2 ;;
        --horizons) parse_csv "$2" HORIZONS; shift 2 ;;
        --datasets) parse_csv "$2" DATASETS; shift 2 ;;
        --encoders) parse_csv "$2" ENCODERS; shift 2 ;;
        --gpus) parse_csv "$2" GPUS; shift 2 ;;
        --slots-per-gpu) SLOTS_PER_GPU="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="$2"; shift 2 ;;
        --shd-once) SHD_ONCE="$2"; shift 2 ;;
        --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --wait-for-free-gpu) WAIT_FOR_FREE_GPU="$2"; shift 2 ;;
        --gpu-free-mem-min-mb-per-slot) GPU_FREE_MEM_MIN_MB_PER_SLOT="$2"; shift 2 ;;
        --config-override) CONFIG_OVERRIDE="$2"; shift 2 ;;
        --override)
            OVERRIDES+=("--$2" "$3")
            shift 3
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$ROOT/.cache/matplotlib"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

jobs=()
first_horizon="${HORIZONS[0]}"
for encoder in "${ENCODERS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            if [[ "$dataset" == "shd" && "$SHD_ONCE" == "true" && "$horizon" != "$first_horizon" ]]; then
                continue
            fi
            for seed in "${SEEDS[@]}"; do
                jobs+=("${dataset}|${seed}|${horizon}|${encoder}")
            done
        done
    done
done

if [[ "${#jobs[@]}" -eq 0 ]]; then
    echo "No jobs to run."
    exit 0
fi

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

slot_refs=()
for gpu in "${GPUS[@]}"; do
    for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
        slot_refs+=("${gpu}:${slot}")
    done
done

for idx in "${!jobs[@]}"; do
    slot_ref="${slot_refs[$((idx % ${#slot_refs[@]}))]}"
    gpu="${slot_ref%%:*}"
    slot="${slot_ref##*:}"
    printf '%s\n' "${jobs[$idx]}" >> "$tmp_dir/gpu_${gpu}_slot_${slot}.jobs"
done

get_gpu_free_mem_mb() {
    local gpu="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return 0
    fi
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d ' '
}

wait_for_gpu_ready() {
    local gpu="$1"
    if [[ "$WAIT_FOR_FREE_GPU" != "true" ]]; then
        return 0
    fi

    local free_mem
    free_mem="$(get_gpu_free_mem_mb "$gpu")"
    if [[ -z "$free_mem" ]]; then
        return 0
    fi

    while [[ "$free_mem" =~ ^[0-9]+$ ]] && (( free_mem < GPU_FREE_MEM_MIN_MB_PER_SLOT )); do
        echo "[GPU $gpu] WAIT  free_mem=${free_mem}MB < ${GPU_FREE_MEM_MIN_MB_PER_SLOT}MB"
        sleep 20
        free_mem="$(get_gpu_free_mem_mb "$gpu")"
        [[ -z "$free_mem" ]] && return 0
    done
}

run_job() {
    local gpu="$1"
    local slot="$2"
    local dataset="$3"
    local seed="$4"
    local horizon="$5"
    local encoder="$6"

    local config
    if [[ -n "$CONFIG_OVERRIDE" ]]; then
        config="$CONFIG_OVERRIDE"
    else
        case "$dataset" in
            metr-la) config="$ROOT/exp/forecast/hybrid_snn/hybrid_snn_metr-la_run.yml" ;;
            electricity) config="$ROOT/exp/forecast/hybrid_snn/hybrid_snn_electricity_run.yml" ;;
            shd) config="$ROOT/exp/classify/hybrid_snn/hybrid_snn_shd_run.yml" ;;
            *)
                echo "[GPU $gpu] Unknown dataset: $dataset" >&2
                return 1
                ;;
        esac
    fi

    local label="${dataset}"
    if [[ "$dataset" != "shd" || "$SHD_ONCE" != "true" ]]; then
        label+="_h${horizon}"
    fi
    label+="_${encoder}_seed${seed}"
    if [[ -n "$RUN_PREFIX" ]]; then
        label="${RUN_PREFIX}_${label}"
    fi

    local out_dir="$RUN_ROOT/$label"
    local checkpoints_dir="$out_dir/checkpoints"
    local log_file="$LOG_DIR/${label}.log"

    if [[ "$SKIP_EXISTING" == "true" && -f "$checkpoints_dir/res.json" ]]; then
        echo "[GPU $gpu/$slot] SKIP  $label"
        return 0
    fi

    echo "[GPU $gpu/$slot] START $label"
    mkdir -p "$out_dir"

    local status=1
    local attempt=0
    while (( attempt <= MAX_RETRIES )); do
        local subsample_override=""
        local gaf_channel_chunk_override=""
        local gaf_backbone_chunk_override=""
        local extra_attempt_note=""

        case "${dataset}:${encoder}:${attempt}" in
            metr-la:gaf:0)
                subsample_override="10"
                gaf_channel_chunk_override="8"
                gaf_backbone_chunk_override="256"
                ;;
            metr-la:gaf:1|metr-la:merged:0)
                subsample_override="12"
                gaf_channel_chunk_override="4"
                gaf_backbone_chunk_override="128"
                extra_attempt_note=" low-mem-profile=1"
                ;;
            metr-la:gaf:*|metr-la:merged:*)
                subsample_override="14"
                gaf_channel_chunk_override="2"
                gaf_backbone_chunk_override="64"
                extra_attempt_note=" low-mem-profile=2"
                ;;
            electricity:gaf:0|electricity:merged:0)
                subsample_override="12"
                gaf_channel_chunk_override="4"
                gaf_backbone_chunk_override="128"
                ;;
            electricity:gaf:1|electricity:merged:1)
                subsample_override="14"
                gaf_channel_chunk_override="2"
                gaf_backbone_chunk_override="64"
                extra_attempt_note=" low-mem-profile=1"
                ;;
            electricity:gaf:*|electricity:merged:*)
                subsample_override="18"
                gaf_channel_chunk_override="1"
                gaf_backbone_chunk_override="32"
                extra_attempt_note=" low-mem-profile=2"
                ;;
            shd:gaf:0|shd:merged:0)
                subsample_override="14"
                gaf_channel_chunk_override="2"
                gaf_backbone_chunk_override="64"
                ;;
            shd:gaf:1|shd:merged:1)
                subsample_override="18"
                gaf_channel_chunk_override="1"
                gaf_backbone_chunk_override="32"
                extra_attempt_note=" low-mem-profile=1"
                ;;
            shd:gaf:*|shd:merged:*)
                subsample_override="25"
                gaf_channel_chunk_override="1"
                gaf_backbone_chunk_override="16"
                extra_attempt_note=" low-mem-profile=2"
                ;;
        esac

        local -a cmd=(
            conda run -n "$CONDA_ENV"
            env
            MPLCONFIGDIR="$ROOT/.cache/matplotlib"
            PYTHONPATH="$PYTHONPATH"
            WANDB_MODE="$WANDB_MODE"
            PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
            python -u -m HybridSNN.entry.run
            "$config"
            --runtime.use_cuda true
            --runtime.seed "$seed"
            --runtime.output_dir "$out_dir"
            --network.encoder_type "$encoder"
            --network.neuron_type tslif
            --network.gaf_checkpoint_backbone true
            --runner.num_workers "$NUM_WORKERS"
            --runner.persistent_workers false
            --runner.wandb_enabled "$WANDB_ENABLED"
            --runner.wandb_project "$WANDB_PROJECT"
            --runner.wandb_run_name "$label"
        )

        if [[ "$dataset" != "shd" ]]; then
            cmd+=(--data.horizon "$horizon")
        fi
        if [[ -n "$subsample_override" ]]; then
            cmd+=(--network.subsample_rate "$subsample_override")
        fi
        if [[ -n "$gaf_channel_chunk_override" ]]; then
            cmd+=(--network.gaf_channel_chunk_size "$gaf_channel_chunk_override")
        fi
        if [[ -n "$gaf_backbone_chunk_override" ]]; then
            cmd+=(--network.gaf_backbone_chunk_size "$gaf_backbone_chunk_override")
        fi
        if [[ "${#OVERRIDES[@]}" -gt 0 ]]; then
            cmd+=("${OVERRIDES[@]}")
        fi

        wait_for_gpu_ready "$gpu"
        if [[ -n "$extra_attempt_note" ]]; then
            echo "[GPU $gpu/$slot] RETRY-PROFILE $label attempt=$attempt${extra_attempt_note} subsample=${subsample_override:-default} chunk=${gaf_channel_chunk_override:-default} backbone=${gaf_backbone_chunk_override:-default}"
        fi

        if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_file" 2>&1; then
            status=0
            break
        fi
        if grep -q "OutOfMemoryError" "$log_file" 2>/dev/null; then
            echo "[GPU $gpu/$slot] OOM   $label attempt=$attempt"
        else
            echo "[GPU $gpu/$slot] RETRY $label attempt=$attempt"
            break
        fi
        attempt=$((attempt + 1))
        sleep 10
    done

    if (( status == 0 )); then
        echo "[GPU $gpu/$slot] DONE  $label"
    else
        echo "[GPU $gpu/$slot] FAIL  $label (see $log_file)"
    fi
    return "$status"
}

worker() {
    local gpu="$1"
    local slot="$2"
    local job_file="$3"
    local failed=0
    while IFS='|' read -r dataset seed horizon encoder; do
        [[ -z "$dataset" ]] && continue
        if ! run_job "$gpu" "$slot" "$dataset" "$seed" "$horizon" "$encoder"; then
            failed=$((failed + 1))
        fi
    done < "$job_file"
    return "$failed"
}

echo "============================================================"
echo " HybridSNN experiment matrix"
echo " datasets=${DATASETS[*]}"
echo " horizons=${HORIZONS[*]}"
echo " seeds=${SEEDS[*]}"
echo " encoders=${ENCODERS[*]}"
echo " gpus=${GPUS[*]}"
echo " slots_per_gpu=$SLOTS_PER_GPU"
echo " logs=$LOG_DIR"
echo " runs=$RUN_ROOT"
echo "============================================================"

pids=()
for gpu in "${GPUS[@]}"; do
    for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
        job_file="$tmp_dir/gpu_${gpu}_slot_${slot}.jobs"
        [[ -f "$job_file" ]] || continue
        worker "$gpu" "$slot" "$job_file" &
        pids+=("$!")
    done
done

failed_workers=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed_workers=$((failed_workers + 1))
    fi
done

echo "============================================================"
echo " Matrix complete. Worker failures: $failed_workers"
echo "============================================================"

if (( failed_workers > 0 )); then
    exit 1
fi
