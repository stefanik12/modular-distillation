#SBATCH --job-name=train_loras_nllb_langs
#SBATCH --account=project_462000764
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=2G
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

export DATA_DIR=/scratch/project_2001194/tiedeman/Tatoeba-Challenge/data/release/v2023-09-26

export MY_PROJECT_HOME=/scratch/project_462000764/mstefani/modular-distillation/

srun python training/distil_nllb_one_lang.py --base_data_dir ${DATA_DIR} --checkpoint_dir ${MY_PROJECT_HOME} "$@"


