#!/bin/bash
## run this eg with:
## sbatch training/train_lumi.sh --teacher_model facebook/nllb-200-distilled-600M --src_lang ces --tgt_langs epo,kab --batch_size 2 --eval_steps 1000
#SBATCH --job-name=distil_nllb
#SBATCH --account=project_462000764
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=small-g
##SBATCH --mail-type=BEGIN #uncomment to enable mail
#SBATCH --gpus-per-node=1

module use /appl/local/csc/modulefiles/
module load pytorch

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/mstefani/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    echo "Initializing conda"
    eval "$__conda_setup"
else
    if [ -f "/users/mstefani/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/users/mstefani/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/mstefani/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate soft-clm

export MY_PROJECT_HOME=/scratch/project_462000764/mstefani/modular-distillation/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME=/scratch/project_462000764/mstefani/hf_home

srun python3.10 training/distil_nllb_one_lang.py --checkpoint_dir ${MY_PROJECT_HOME} "$@"
