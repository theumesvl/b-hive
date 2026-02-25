#!/bin/bash
#
# Complete MVA training workflow for HPlusCHToWW
#
# Usage:
#   ./train_MVA.sh                     # Run with defaults (multi-class)
#   ./train_MVA.sh --debug             # Quick test run
#   ./train_MVA.sh --epochs 100        # Custom epochs
#

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG="HPlusCHToWW_multiclass"
#FILELIST="/eos/user/c/cgupta/higgscharm/combined/hww/2022postEE/filelists/base.txt"
FILELIST="/eos/user/c/cgupta/higgscharm/outputs/hww_MVA/2022postEE/filelists/base.txt"
VERSION="hww_multiclass_v1"
MODEL="SimpleMLP_MultiClass"
EPOCHS=50
BATCH_SIZE=1024
LEARNING_RATE=1e-3
COFFEA_WORKERS=4
CHUNK_SIZE=5000000
LOSS_WEIGHTING=false
DEBUG=false

# Paths
#COMBINE_SCRIPT="/afs/cern.ch/user/c/cgupta/higgscharm_thomas/higgscharm/combine_parquet.sh"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --filelist)
            FILELIST="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --workers)
            COFFEA_WORKERS="$2"
            shift 2
            ;;
        --no-loss-weighting)
            LOSS_WEIGHTING=false
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG       Config name (default: HPlusCHToWW_multiclass)"
            echo "  --filelist FILE       Path to filelist"
            echo "  --version VERSION     Dataset/training version tag (default: hww_multiclass_v1)"
            echo "  --model MODEL         Model name (default: SimpleMLP_MultiClass)"
            echo "  --epochs N            Number of epochs (default: 50)"
            echo "  --batch-size N        Batch size (default: 1024)"
            echo "  --lr RATE             Learning rate (default: 1e-3)"
            echo "  --workers N           Coffea workers (default: 4)"
            echo "  --no-loss-weighting   Use weighted sampling instead of loss weighting"
            echo "  --skip-combine        Skip combine_parquet step (use existing files)"
            echo "  --debug               Run on small subset for testing"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Available models:"
            echo "  SimpleMLP_MultiClass       - Multi-class (higgs vs tt/st/diboson/ewk/dy)"
            echo "  SimpleMLP_MultiClass_Large - Larger multi-class variant"
            echo "  SimpleMLP_HToWW            - Binary (signal vs background)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# SETUP ENVIRONMENT
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "HPlusCHToWW MVA Training (Multi-class)"
echo "=============================================="
echo "Config:         $CONFIG"
echo "Filelist:       $FILELIST"
echo "Version:        $VERSION"
echo "Model:          $MODEL"
echo "Epochs:         $EPOCHS"
echo "Batch size:     $BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Loss weighting: $LOSS_WEIGHTING"
echo "Debug mode:     $DEBUG"
echo "=============================================="
echo ""

# =============================================================================
# STEP 0: COMBINE PARQUET FILES
# =============================================================================

# Check if filelist exists
if [ ! -f "$FILELIST" ]; then
    echo "ERROR: Filelist not found: $FILELIST"
    exit 1
fi

# =============================================================================
# STEP 1: DATASET CONSTRUCTION
# =============================================================================
echo ""
echo "=============================================="
echo "Step 1: Dataset Construction"
echo "=============================================="

DATASET_CMD="law run DatasetConstructorTask"
DATASET_CMD="$DATASET_CMD --config $CONFIG"
DATASET_CMD="$DATASET_CMD --filelist $FILELIST"
DATASET_CMD="$DATASET_CMD --dataset-version $VERSION"
DATASET_CMD="$DATASET_CMD --coffea-worker $COFFEA_WORKERS"
DATASET_CMD="$DATASET_CMD --chunk-size $CHUNK_SIZE"

if [ "$DEBUG" = true ]; then
    DATASET_CMD="$DATASET_CMD --debug"
fi

echo "Running: $DATASET_CMD"
echo ""
$DATASET_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset construction failed!"
    exit 1
fi

# =============================================================================
# STEP 2: TRAINING
# =============================================================================
echo ""
echo "=============================================="
echo "Step 2: Training"
echo "=============================================="

TRAIN_CMD="law run TrainingTask"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
TRAIN_CMD="$TRAIN_CMD --filelist $FILELIST"
TRAIN_CMD="$TRAIN_CMD --dataset-version $VERSION"
TRAIN_CMD="$TRAIN_CMD --training-version $VERSION"
TRAIN_CMD="$TRAIN_CMD --model-name $MODEL"
TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --learning-rate $LEARNING_RATE"

if [ "$LOSS_WEIGHTING" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --loss-weighting"
fi

if [ "$DEBUG" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --debug"
fi

echo "Running: $TRAIN_CMD"
echo ""
$TRAIN_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

# =============================================================================
# STEP 3: INFERENCE
# =============================================================================
echo ""
echo "=============================================="
echo "Step 3: Inference"
echo "=============================================="

INFERENCE_CMD="law run InferenceTask"
INFERENCE_CMD="$INFERENCE_CMD --config $CONFIG"
INFERENCE_CMD="$INFERENCE_CMD --filelist $FILELIST"
INFERENCE_CMD="$INFERENCE_CMD --dataset-version $VERSION"
INFERENCE_CMD="$INFERENCE_CMD --training-version $VERSION"
INFERENCE_CMD="$INFERENCE_CMD --model-name $MODEL"
INFERENCE_CMD="$INFERENCE_CMD --epochs $EPOCHS"
INFERENCE_CMD="$INFERENCE_CMD --batch-size $BATCH_SIZE"
INFERENCE_CMD="$INFERENCE_CMD --learning-rate $LEARNING_RATE"
INFERENCE_CMD="$INFERENCE_CMD --test-dataset-version $VERSION"
INFERENCE_CMD="$INFERENCE_CMD --test-filelist $FILELIST"

if [ "$DEBUG" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --debug"
fi

echo "Running: $INFERENCE_CMD"
echo ""
$INFERENCE_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

# =============================================================================
# STEP 4: ROC CURVE
# =============================================================================
echo ""
echo "=============================================="
echo "Step 4: ROC Curve"
echo "=============================================="

ROC_CMD="law run ROCCurveTask"
ROC_CMD="$ROC_CMD --config $CONFIG"
ROC_CMD="$ROC_CMD --filelist $FILELIST"
ROC_CMD="$ROC_CMD --dataset-version $VERSION"
ROC_CMD="$ROC_CMD --training-version $VERSION"
ROC_CMD="$ROC_CMD --model-name $MODEL"
ROC_CMD="$ROC_CMD --epochs $EPOCHS"
ROC_CMD="$ROC_CMD --batch-size $BATCH_SIZE"
ROC_CMD="$ROC_CMD --learning-rate $LEARNING_RATE"
ROC_CMD="$ROC_CMD --test-dataset-version $VERSION"
ROC_CMD="$ROC_CMD --test-filelist $FILELIST"

if [ "$DEBUG" = true ]; then
    ROC_CMD="$ROC_CMD --debug"
fi

echo "Running: $ROC_CMD"
echo ""
$ROC_CMD

if [ $? -ne 0 ]; then
    echo "ERROR: ROC curve plotting failed!"
    exit 1
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Output location: $SCRIPT_DIR/output/TrainingTask/$CONFIG/$VERSION/"
echo "ROC curves: $SCRIPT_DIR/output/ROCCurveTask/$CONFIG/$VERSION/"
echo ""
ls -la "$SCRIPT_DIR/output/TrainingTask/$CONFIG/$VERSION/" 2>/dev/null || echo "Check output directory for results"
