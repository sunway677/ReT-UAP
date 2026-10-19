#!/bin/bash
#SBATCH -J retuap_train
# #SBATCH -p <partition>   # uncomment and set your cluster GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#
# Train ReT-UAP with one primary surrogate; the other five models of the pool
# (ALBEF, TCL, BLIP, XVLM, ViT-B/16, ViT-B/32) are the auxiliaries.
#
# Usage (from the repository root):
#   bash   scripts/train.sh <PRIMARY> <DATASET> [extra retuap_train.py flags]
#   sbatch scripts/train.sh <PRIMARY> <DATASET> [extra retuap_train.py flags]
#     PRIMARY: ALBEF | TCL | BLIP | XVLM | ViT-B/16 | ViT-B/32
#     DATASET: flickr30k | mscoco
# Example: bash scripts/train.sh "ViT-B/16" flickr30k
#
# All hyperparameters use the retuap_train.py defaults (manuscript Sec. 5.1);
# extra flags after DATASET are passed through, e.g. --eps 8.
# Prerequisites: ./checkpoint/<MODEL>/<DATASET>.pth for ALBEF/TCL/BLIP/XVLM and
# word_embedding_<TAG>_<DATASET>.pth in the repository root
# (python scripts/generate_word_embeddings.py --all).
# Output: output_retuap/<TAG>/<DATASET>/ with TAG = PRIMARY with '/' -> '_'.

usage="usage: $0 <ALBEF|TCL|BLIP|XVLM|ViT-B/16|ViT-B/32> <flickr30k|mscoco> [extra flags]"
PRIMARY="${1:?$usage}"
DATASET="${2:?$usage}"
shift 2

case "$PRIMARY" in
  ALBEF|TCL|BLIP|XVLM|ViT-B/16|ViT-B/32) ;;
  *) echo "unknown primary surrogate '$PRIMARY'"; echo "$usage"; exit 1 ;;
esac
case "$DATASET" in
  flickr30k) CFG=flickr ;;
  mscoco)    CFG=coco ;;
  *) echo "unknown dataset '$DATASET'"; echo "$usage"; exit 1 ;;
esac

source ~/.bashrc
conda activate retuap
set -e
# repo root: script location under bash, submission directory under sbatch
# (sbatch runs a spooled copy of this script, so $0 does not point into the repo)
if [ -f "$(dirname "$0")/../retuap_train.py" ]; then cd "$(dirname "$0")/.."; else cd "$SLURM_SUBMIT_DIR"; fi
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python retuap_train.py \
  --config "./configs/Retrieval_${CFG}_train.yaml" \
  --test_config "./configs/Retrieval_${CFG}_test.yaml" \
  --primary_encoder "$PRIMARY" \
  --checkpoint_dir ./checkpoint \
  --save_dir output_retuap \
  "$@"
