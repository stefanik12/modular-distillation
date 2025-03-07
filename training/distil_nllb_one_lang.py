import argparse
import itertools
import os
from copy import deepcopy
from typing import Optional, List, Type

import torch
import wandb
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM, AutoConfig, M2M100ForConditionalGeneration

from training.langs import flores200_langs, iso639_3_to_iso639_1, drop_locale

torch.manual_seed(4321)

import random
random.seed(4321)

from adaptor.adapter import Adapter
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset, concatenate_datasets, Dataset, get_dataset_config_names
from peft import LoraConfig, TaskType
from tqdm import tqdm

torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_model", help="A pre-trained model to initialize "
                                            "the training with", required=True, type=str)
parser.add_argument("--checkpoint_dir", help="A base folder where to store the training checkpoints."
                                             "Ignored in continued training.", type=str, default=".")
parser.add_argument("--reset_weights", help="Whether to reset the base model's weights",
                    type=bool, default=False)
parser.add_argument("--src_lang", help="Source and target lang for one-to-many and many-to-one distillation")
parser.add_argument("--tgt_langs", help="Coma-separated list of target languages. E.g: "
                                           "`sgn,tah`. Defaults to all the NLLB's target languages.", default="")
# parser.add_argument("--pair_evaluation_langs", help="Language pairs on which to perform pair evaluations"
#                                                     "(GradientDotProduct eval). Format: 'fur,tah;epo,est'", default="")
parser.add_argument("--eval_batches", default=20, type=int)
parser.add_argument("--eval_steps", default=500, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--train_firstn", default=None, type=int)
parser.add_argument("--save_steps", default=500, type=int)
parser.add_argument("--resume_from_checkpoint", help="Whether this is a continued training."
                                                     "Defaults to False", default="False", type=str)
parser.add_argument("--learning_rate", help="Learning rate used with all objectives. Defaults to `2e-5`.",
                    default=2e-5, type=float)
args = parser.parse_args()
args.resume_from_checkpoint = args.resume_from_checkpoint.lower() != "false"
# args.eval_run = args.eval_run.lower() != "false"

print("Running with arguments: %s" % args)
print("Training World size: %s" % int(os.environ.get("WORLD_SIZE", 1)))

# wandb.init(project="modular-distillation")
# wandb.log({"slurm_id": os.environ.get("SLURM_JOB_ID", -1)}, commit=False)
#
if args.resume_from_checkpoint:
    # remove the checkpoint-X part of path
    checkpoint_dir = args.checkpoint_dir.split("/checkpoint-")[0]
    # was: checkpoint_dir = args.base_model.split("/checkpoint-")[0]
else:
    checkpoint_dir = os.path.join(args.checkpoint_dir, "checkpoints")
#     if not args.local_run and os.environ.get("LOCAL_RANK", 0) == 0:
#         checkpoint_dir = checkpoint_dir + "-" + wandb.run.name
#
print("Checkpoint will be saved to '{}'".format(checkpoint_dir))


# 1. Initialize data: evaluation (flores for given lang), training (opus for given lang)
# 1.1 Eval Dataset
all_eval_splits = []

src_lang_fl = [lang for lang in flores200_langs if lang.startswith(args.src_lang)]
assert len(src_lang_fl) == 1, "Ambiguous src lang resolution for %s" % args.src_lang
src_lang_fl = src_lang_fl[0]

tgt_langs_fl = []

tgt_langs = args.tgt_langs.split(",") if args.tgt_langs else flores200_langs

for tgt_lang in tgt_langs:
    # resolution of tgt language in arbitrary encoding
    tgt_lang_fl = [lang for lang in flores200_langs if lang.startswith(tgt_lang)]
    assert len(tgt_lang_fl) == 1, "Ambiguous tgt lang resolution for %s" % tgt_lang_fl
    tgt_lang_fl = tgt_lang_fl[0]

    assert tgt_lang_fl in flores200_langs

    if src_lang_fl == tgt_lang_fl:
        continue
    tgt_langs_fl.append(tgt_lang_fl)

    new_dataset = load_dataset("Muennighoff/flores200", "%s-%s" % (src_lang_fl, tgt_lang_fl),
                               split="dev", trust_remote_code=True)
    new_dataset_subset = new_dataset.select(range(args.eval_batches * args.batch_size))

    new_dataset_subset = new_dataset_subset.map(lambda row: {"src_lang": src_lang_fl,
                                                             "tgt_lang": tgt_lang_fl,
                                                             "src_text": row["sentence_%s" % src_lang_fl],
                                                             "tgt_text": row["sentence_%s" % tgt_lang_fl]})
    all_eval_splits.append(new_dataset_subset)

eval_dataset = concatenate_datasets(all_eval_splits)  # contains 'src_text' and 'tgt_text' columns to use for eval

# 1.2 Train dataset
TRAIN_DATASET_IDS = "michal-stefanik/tatoeba_mt_ces-x"

all_splits = get_dataset_config_names(TRAIN_DATASET_IDS)
src_lang_subsets = [s for s in all_splits if drop_locale(src_lang_fl) in s]

all_train_datasets = []

for subset in src_lang_subsets:
    split_with_subset = "train" if args.train_firstn is None else "train[:%s]" % args.train_firstn
    try:
        new_tatoeba_dataset = load_dataset(TRAIN_DATASET_IDS, subset, split=split_with_subset)
    except ValueError:
        # ValueError: Unknown split "train".
        print("Subset %s does not contain train split; skipping." % subset)
        continue

    # TODO unify the direction of the training data into the same column

    all_train_datasets.append(new_tatoeba_dataset)

train_dataset = concatenate_datasets(all_train_datasets)  # same format as eval data
print()


# 2. Initialize student model: copy the weights of alternate transformer blocks, both encoder and decoder
def construct_student_from_teacher(teacher_model_id: str,
                                   student_model_type: Type[PreTrainedModel],
                                   reset_weights: bool,
                                   layers_reduction_ratio: int = 4) -> PreTrainedModel:
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_id)
    teacher_config = AutoConfig.from_pretrained(teacher_model_id)
    student_config = AutoConfig.from_pretrained(teacher_model_id)
    if teacher_config.is_encoder_decoder:
        assert teacher_config.encoder_layers % layers_reduction_ratio == 0, \
            "Num of student encoder layers must be divisible by layers_reduction_ratio"
        assert teacher_config.decoder_layers % layers_reduction_ratio == 0, \
            "Num of student decoder layers must be divisible by layers_reduction_ratio"
    else:
        raise ValueError("For now, we only support encoder-decoder architecture")
    student_config.encoder_layers = teacher_config.encoder_layers // layers_reduction_ratio
    student_config.decoder_layers = teacher_config.decoder_layers // layers_reduction_ratio
    student_model = student_model_type(config=student_config)

    if not reset_weights:
        for student_i, teacher_i in enumerate(range(3, teacher_config.encoder_layers, layers_reduction_ratio)):
            # TODO: this might be problematic with gradient propagation -- let's try this out and see if it works
            teacher_layer = teacher_model.base_model.encoder.layers[teacher_i]
            student_model.base_model.encoder.layers[student_i] = deepcopy(teacher_layer)

        for student_i, teacher_i in enumerate(range(3, teacher_config.decoder_layers, layers_reduction_ratio)):
            teacher_layer = teacher_model.base_model.decoder.layers[teacher_i]
            student_model.base_model.decoder.layers[student_i] = deepcopy(teacher_layer)

    return student_model


student_model = construct_student_from_teacher(args.teacher_model,
                                               M2M100ForConditionalGeneration,
                                               args.reset_weights)

# 3. Initialize two objectives: 1. forward with data: {lang}-{others}, 2. backward with data: {others}-{lang}
#   the data can be constructed in advance by concatenating all relevant subsets of HF Opus-100 dataset


# 4. Consider whether to weigh the distil loss with the standard seq2seq

