import argparse
import os
from copy import deepcopy
from typing import Type

import torch
from adaptor.lang_module import LangModule
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM, AutoConfig, M2M100ForConditionalGeneration, \
    AutoTokenizer

import wandb
from training.distilled_seq2seq import DistilledNLLB
from training.langs import flores200_langs, drop_locale, get_intersecting_target_langs
import random

torch.manual_seed(4321)
random.seed(4321)

from adaptor.adapter import Adapter
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from adaptor.evaluators.generative import BLEU
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

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
parser.add_argument("--train_firstn", default=0, type=int)
parser.add_argument("--save_steps", default=500, type=int)
parser.add_argument("--layers_reduction_ratio", default=12, type=int)
parser.add_argument("--resume_from_checkpoint", help="Whether this is a continued training."
                                                     "Defaults to False", default="False", type=str)
parser.add_argument("--learning_rate", help="Learning rate used with all objectives. Defaults to `2e-5`.",
                    default=2e-5, type=float)
args = parser.parse_args()
args.resume_from_checkpoint = args.resume_from_checkpoint.lower() != "false"
# args.eval_run = args.eval_run.lower() != "false"

print("Running with arguments: %s" % args)
print("Training World size: %s" % int(os.environ.get("WORLD_SIZE", 1)))

wandb.init(project="modular-distillation")
wandb.log({"slurm_id": os.environ.get("SLURM_JOB_ID", -1)}, commit=False)

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
# Languages resolution

all_eval_splits = []

src_lang_fl = [lang for lang in flores200_langs if lang.startswith(args.src_lang)]
assert len(src_lang_fl) == 1, "Ambiguous src lang resolution for %s" % args.src_lang
src_lang_fl = src_lang_fl[0]

src_lang_tatoeba = drop_locale(src_lang_fl)

TRAIN_DATASET_ID = "michal-stefanik/tatoeba_mt_ces-x"
all_tatoeba_splits = get_dataset_config_names(TRAIN_DATASET_ID)
srclang_tatoeba_splits = [split for split in all_tatoeba_splits if src_lang_tatoeba in split]

tgt_langs_tatoeba = [split.replace(src_lang_tatoeba, "").replace("-", "") for split in srclang_tatoeba_splits]

tgt_langs_initial = [l.split("_")[0] for l in args.tgt_langs.split(",")] if args.tgt_langs else tgt_langs_tatoeba

assert all(any(lang in split for split in all_tatoeba_splits) for lang in tgt_langs_initial), \
    "Tatoeba does not have data for some of the given target langs. Available splits: %s" % srclang_tatoeba_splits

tgt_langs_fl, tgt_langs_tatoeba = get_intersecting_target_langs(tatoeba_target_langs=tgt_langs_initial)

# 1.1 FLORES eval data resolution
for tgt_lang_fl in tgt_langs_fl:
    if src_lang_fl == tgt_lang_fl:
        # skip the training for the identical language
        continue

    # tgt_langs_fl.append(tgt_lang_fl)

    new_dataset = load_dataset("Muennighoff/flores200", "%s-%s" % (src_lang_fl, tgt_lang_fl),
                               split="dev", trust_remote_code=True)
    new_dataset_subset = new_dataset.select(range(args.eval_batches * args.batch_size))

    new_dataset_subset = new_dataset_subset.map(lambda row: {"source_lang": src_lang_fl,
                                                             "target_lang": tgt_lang_fl,
                                                             "source_text": row["sentence_%s" % src_lang_fl],
                                                             "target_text": row["sentence_%s" % tgt_lang_fl]})
    all_eval_splits.append(new_dataset_subset)

eval_dataset = concatenate_datasets(all_eval_splits)  # contains 'src_text' and 'tgt_text' columns to use for eval

# 1.2 Tatoeba Training dataset resolution

# debug:
# all_splits = ["bre-ces"]
src_lang_subsets = [s for s in all_tatoeba_splits if src_lang_tatoeba in s]

all_train_datasets = []

src_tgtlang_tatoeba_splits = [s for s in srclang_tatoeba_splits if any(tgt_l in s for tgt_l in tgt_langs_tatoeba)]
for subset in src_tgtlang_tatoeba_splits:
    split_with_subset = "train" if not args.train_firstn else "train[:%s]" % args.train_firstn
    try:
        new_tatoeba_dataset = load_dataset(TRAIN_DATASET_ID, subset, split=split_with_subset)
    except ValueError:
        # ValueError: Unknown split "train".
        print("Subset %s does not contain train split; skipping." % subset)
        continue

    # consistent ordering of languages to allow quick column-wise access:
    new_tatoeba_dataset = new_tatoeba_dataset.map(lambda row: row if row["source_lang"] == src_lang_tatoeba else
    {"source_text": row["target_text"],
     "target_text": row["source_text"],
     "source_lang": row["target_lang"],
     "target_lang": row["source_lang"]})
    # NLLB-specific adjustment: lang codes match the Flores collection
    # assuming that the whole dataset subset contains consistent lang pair
    tgt_lang_fl = [lang for lang in flores200_langs if lang.startswith(new_tatoeba_dataset["target_lang"][0])]
    assert len(tgt_lang_fl) == 1, "Ambiguous tgt lang resolution for %s" % tgt_lang_fl
    tgt_lang_fl = tgt_lang_fl[0]

    new_tatoeba_dataset = new_tatoeba_dataset.map(lambda row: {"source_lang": src_lang_fl,
                                                               "target_lang": tgt_lang_fl})

    all_train_datasets.append(new_tatoeba_dataset)

train_dataset = concatenate_datasets(all_train_datasets)  # same format as eval data
print()


# 2. Initialize student model: copy the weights of alternate transformer blocks, both encoder and decoder
def construct_student_from_teacher(teacher_model: PreTrainedModel,
                                   teacher_model_id: str,
                                   student_model_type: Type[PreTrainedModel],
                                   reset_weights: bool,
                                   layers_reduction_ratio: int = args.layers_reduction_ratio) -> PreTrainedModel:
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
        for student_i, teacher_i in enumerate(range(layers_reduction_ratio - 1,
                                                    teacher_config.encoder_layers, layers_reduction_ratio)):
            # TODO: this might be problematic with gradient propagation -- let's try this out and see if it works
            teacher_layer = teacher_model.base_model.encoder.layers[teacher_i]
            student_model.base_model.encoder.layers[student_i] = deepcopy(teacher_layer)

        for student_i, teacher_i in enumerate(range(layers_reduction_ratio - 1, teacher_config.decoder_layers,
                                                    layers_reduction_ratio)):
            teacher_layer = teacher_model.base_model.decoder.layers[teacher_i]
            student_model.base_model.decoder.layers[student_i] = deepcopy(teacher_layer)

    return student_model


teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model)
student_model = construct_student_from_teacher(teacher_model,
                                               args.teacher_model,
                                               M2M100ForConditionalGeneration,
                                               args.reset_weights)
INIT_MODEL_PATH = os.path.join(args.checkpoint_dir, "init_student_model")

student_model.save_pretrained(INIT_MODEL_PATH)
AutoTokenizer.from_pretrained(args.teacher_model).save_pretrained(INIT_MODEL_PATH)

# 3. Initialize two objectives: 1. forward with data: {lang}-{others}, 2. backward with data: {others}-{lang}
#   the data can be constructed in advance by concatenating all relevant subsets of HF Opus-100 dataset
lang_module = LangModule(INIT_MODEL_PATH)
evaluators = [BLEU()]

fwd_objective = DistilledNLLB(lang_module=lang_module,
                              teacher_model=teacher_model,
                              batch_size=args.batch_size,
                              val_evaluators=evaluators,

                              texts_or_path=train_dataset["source_text"],
                              texts_langs=train_dataset["source_lang"],
                              labels_or_path=train_dataset["target_text"],
                              labels_langs=train_dataset["target_lang"],

                              val_texts_or_path=eval_dataset["source_text"],
                              val_texts_langs=eval_dataset["source_lang"],
                              val_labels_or_path=eval_dataset["target_text"],
                              val_labels_langs=eval_dataset["target_lang"],
                              # source_lang_id="en", target_lang_id="cs"
                              objective_id="%s->X" % args.src_lang
                              )
bwd_objective = DistilledNLLB(lang_module,
                              teacher_model=teacher_model,
                              batch_size=args.batch_size,
                              val_evaluators=evaluators,

                              texts_or_path=train_dataset["target_text"],
                              texts_langs=train_dataset["target_lang"],
                              labels_or_path=train_dataset["source_text"],
                              labels_langs=train_dataset["source_lang"],

                              val_texts_or_path=eval_dataset["target_text"],
                              val_texts_langs=eval_dataset["target_lang"],
                              val_labels_or_path=eval_dataset["source_text"],
                              val_labels_langs=eval_dataset["source_lang"],
                              # source_lang_id="en", target_lang_id="cs"
                              objective_id="X->%s" % args.src_lang
                              )
train_objectives = [fwd_objective, bwd_objective]

training_arguments = AdaptationArguments(output_dir=args.checkpoint_dir,
                                         stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=8,
                                         evaluation_strategy="steps",
                                         # log_level="critical",
                                         logging_steps=200,
                                         eval_steps=args.eval_steps,
                                         num_train_epochs=10,
                                         save_steps=args.save_steps,
                                         no_cuda=True if args.train_firstn < 10e4 else False,
                                         )

schedule = ParallelSchedule(train_objectives, training_arguments)
adapter = Adapter(lang_module, schedule, training_arguments)
adapter.train()

# 4. Consider whether to weigh the distil loss with the standard seq2seq
