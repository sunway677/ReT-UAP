#!/bin/bash
#SBATCH -J retuap_eval
# #SBATCH -p <partition>   # uncomment and set your cluster GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#
# Evaluate the UAP (delta_x*, delta_t*) trained with scripts/train.sh on all
# six target models.
#
# Usage (from the repository root):
#   bash   scripts/eval.sh <PRIMARY> <DATASET> [extra eval.py flags]
#   sbatch scripts/eval.sh <PRIMARY> <DATASET> [extra eval.py flags]
#     PRIMARY: ALBEF | TCL | BLIP | XVLM | ViT-B/16 | ViT-B/32  (the source model)
#     DATASET: flickr30k | mscoco
# Example: bash scripts/eval.sh "ViT-B/16" flickr30k
#
# Reads output_retuap/<TAG>/<DATASET>/uap_{noise,embedding}-99.pth
# (TAG = PRIMARY with '/' -> '_') and word_embedding_<TAG>_<DATASET>.pth plus
# word_idx_{BERT,CLIP}.npy from the repository root. Results are written to
# outputs/result_retuap_<TAG>_<DATASET>/.

usage="usage: $0 <ALBEF|TCL|BLIP|XVLM|ViT-B/16|ViT-B/32> <flickr30k|mscoco> [extra flags]"
PRIMARY="${1:?$usage}"
DATASET="${2:?$usage}"
shift 2

case "$PRIMARY" in
  ALBEF|TCL|BLIP|XVLM|ViT-B/16|ViT-B/32) ;;
  *) echo "unknown source model '$PRIMARY'"; echo "$usage"; exit 1 ;;
esac
case "$DATASET" in
  flickr30k) CFG=flickr ;;
  mscoco)    CFG=coco ;;
  *) echo "unknown dataset '$DATASET'"; echo "$usage"; exit 1 ;;
esac
TAG="${PRIMARY//\//_}"

source ~/.bashrc
conda activate retuap
set -e
# repo root: script location under bash, submission directory under sbatch
# (sbatch runs a spooled copy of this script, so $0 does not point into the repo)
if [ -f "$(dirname "$0")/../retuap_train.py" ]; then cd "$(dirname "$0")/.."; else cd "$SLURM_SUBMIT_DIR"; fi
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval.py \
  --config "./configs/Retrieval_${CFG}_test.yaml" \
  --source_model "$PRIMARY" \
  --checkpoint_dir ./checkpoint \
  --load_dir ./output_retuap \
  --epoch 99 \
  --output "result_retuap_${TAG}_${DATASET}" \
  "$@"
