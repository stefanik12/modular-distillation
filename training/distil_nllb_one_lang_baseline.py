import argparse
import itertools
import os
from copy import deepcopy
from typing import Type, Tuple, Iterator, List

import torch
from adaptor.lang_module import LangModule
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM, AutoConfig, M2M100ForConditionalGeneration, \
    AutoTokenizer

import wandb
from training.distilled_seq2seq import DistilledNLLB, BaselineSeq2Seq
from training.langs import flores200_langs, drop_locale, get_intersecting_target_langs, match_flores_langs
import random

from training.teacher_utils import CachedTeacherTranslator

torch.manual_seed(4321)
random.seed(4321)

from adaptor.adapter import Adapter
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from adaptor.evaluators.generative import BLEU
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, IterableDataset, interleave_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_model", help="A pre-trained model to initialize "
                                            "the training with", required=True, type=str)
parser.add_argument("--use_teacher_targets", help="Whether to use teacher's predictions as targets", default="False")
parser.add_argument("--teacher_targets_batch_size", help="Batch size for inference with teacher", type=int, default=32)
parser.add_argument("--construct_new_student", help="Whether to reinitialize a new student model"
                                                    "based on the teacher", type=str, default="True")
parser.add_argument("--checkpoint_dir", help="A base folder where to store the training checkpoints."
                                             "Ignored in continued training.", type=str, default=".")
parser.add_argument("--reset_weights", help="Whether to reset the base model's weights", type=bool, default=False)
parser.add_argument("--add_hidden_states_loss", help="Whether to distill also based on hidden states", default="True")
parser.add_argument("--restrict_loss_to_mask", help="Whether to compute loss only from attended tokens, default", default="True")
parser.add_argument("--src_lang", help="Source and target lang for one-to-many and many-to-one distillation")
parser.add_argument("--tgt_langs", help="Coma-separated list of target languages. E.g: "
                                        "`sgn,tah`. Defaults to all the NLLB's target languages.", default="")
parser.add_argument("--alignment_scores_paths", help="Paths to text files with quality scores for translation pairs", default="None")
parser.add_argument("--alignment_scores_threshold", help="Threshold for filtering out low-quality input pairs", default=0.95)
# parser.add_argument("--pair_evaluation_langs", help="Language pairs on which to perform pair evaluations"
#                                                     "(GradientDotProduct eval). Format: 'fur,tah;epo,est'", default="")
parser.add_argument("--eval_batches", default=20, type=int)
parser.add_argument("--eval_steps", default=500, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--train_data_buffer_ratio", default=4, type=int)
parser.add_argument("--effective_batch_size", default=32, type=int)
parser.add_argument("--train_firstn", default=0, type=int)
parser.add_argument("--save_steps", default=10000, type=int)
parser.add_argument("--layers_reduction_ratio", default=12, type=int)  # TODO: check this in Distillation
parser.add_argument("--resume_from_checkpoint", help="Whether this is a continued training."
                                                     "Defaults to False", default="False", type=str)
parser.add_argument("--learning_rate", help="Learning rate used with all objectives. Defaults to `2e-5`.",
                    default=2e-5, type=float)
args = parser.parse_args()
args.use_teacher_targets = args.use_teacher_targets.lower() != "false"
args.construct_new_student = args.construct_new_student.lower() != "false"
args.resume_from_checkpoint = args.resume_from_checkpoint.lower() != "false"
args.add_hidden_states_loss = args.add_hidden_states_loss.lower() != "false"
args.restrict_loss_to_mask = args.restrict_loss_to_mask.lower() != "false"
args.alignment_scores_paths = args.alignment_scores_paths.split(",") if args.alignment_scores_paths != "None" else None
# args.eval_run = args.eval_run.lower() != "false"

USE_CUDA = False if (args.train_firstn and args.train_firstn < 10e4) else True  # No cuda if running with subset of data

wandb.init(project="modular-distillation")
wandb.log({"slurm_id": os.environ.get("SLURM_JOB_ID", -1)}, commit=False)

print("Running with arguments: %s" % args)
print("Training World size: %s" % int(os.environ.get("WORLD_SIZE", 1)))
print("CUDA.is_available(): %s" % torch.cuda.is_available())
print("USE_CUDA: %s" % USE_CUDA)

if args.resume_from_checkpoint:
    # remove the checkpoint-X part of path
    checkpoint_dir = args.checkpoint_dir.split("/checkpoint-")[0]
    # was: checkpoint_dir = args.base_model.split("/checkpoint-")[0]
else:
    checkpoint_dir = os.path.join(args.checkpoint_dir, "checkpoints")
    if os.environ.get("LOCAL_RANK", 0) == 0 and wandb.run is not None:
        checkpoint_dir = checkpoint_dir + "-" + wandb.run.name

print("Checkpoint will be saved to '{}'".format(checkpoint_dir))

# 0. Initialize teacher
teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model)
if USE_CUDA:
    teacher_model = teacher_model.to("cuda")
teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

if args.use_teacher_targets:
    teacher_wrapper = CachedTeacherTranslator(teacher_model, teacher_tokenizer, args.checkpoint_dir)
else:
    teacher_wrapper = None

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

    # TODO: maybe perform the same filtering as with train dataset?

    new_dataset = load_dataset("Muennighoff/flores200", "%s-%s" % (src_lang_fl, tgt_lang_fl),
                               split="dev", trust_remote_code=True)
    new_dataset_subset = new_dataset.select(range(args.eval_batches * args.batch_size))

    new_dataset_subset = new_dataset_subset.map(lambda row: {"source_lang": src_lang_fl,
                                                             "target_lang": tgt_lang_fl,
                                                             "source_text": row["sentence_%s" % src_lang_fl],
                                                             "target_text": row["sentence_%s" % tgt_lang_fl]})
    all_eval_splits.append(new_dataset_subset)

eval_dataset = interleave_datasets(all_eval_splits)  # contains 'src_text' and 'tgt_text' columns to use for eval

# 1.2 Tatoeba Training dataset resolution

# debug:
# all_splits = ["bre-ces"]
src_lang_subsets = [s for s in all_tatoeba_splits if src_lang_tatoeba in s]

all_train_datasets = []
train_dataset_length = 0
target_langs = []

src_tgtlang_tatoeba_splits = [s for s in srclang_tatoeba_splits if any(tgt_l in s for tgt_l in tgt_langs_tatoeba)]
if args.alignment_scores_paths is None:
    args.alignment_scores_paths = [None for _ in src_tgtlang_tatoeba_splits]
else:
    assert len(args.alignment_scores_paths) == len(src_tgtlang_tatoeba_splits)

for subset, alignment_fpath in zip(src_tgtlang_tatoeba_splits, args.alignment_scores_paths):
    # split_with_subset = "train" if not args.train_firstn else "train[:%s]" % args.train_firstn
    split_with_subset = "train"
    try:
        new_tatoeba_dataset = load_dataset(TRAIN_DATASET_ID, subset, split=split_with_subset, streaming=True)
    except ValueError:
        # ValueError: Unknown split "train".
        print("Subset %s does not contain train split; skipping." % subset)
        continue

    # NLLB-specific adjustment: lang codes match the Flores collection
    # assuming that the whole dataset subset contains consistent lang pair
    tgt_lang_tatoeba = subset.replace(src_lang_tatoeba, "").replace("-", "")
    tgt_lang_fl = match_flores_langs(tgt_lang_tatoeba)
    if len(tgt_lang_fl) > 1:
        print("Ambiguous tgt lang resolution for %s. Skipping and dropping %s from eval datasets."
              % (tgt_lang_tatoeba, tgt_lang_fl))

        eval_dataset = eval_dataset.filter(lambda row: not (any(l_fl in row["source_lang"] for l_fl in tgt_lang_fl)
                                                            or any(l_fl in row["target_lang"] for l_fl in tgt_lang_fl)))
    if len(tgt_lang_fl) != 1:
        print("No matching source lang found for %s. Skipping." % tgt_lang_tatoeba)
        continue

    # consistent ordering of languages to allow quick column-wise access:
    new_tatoeba_dataset = new_tatoeba_dataset.map(lambda row: row if row["source_lang"] == src_lang_tatoeba
                                                              else {"source_text": row["target_text"],
                                                                    "target_text": row["source_text"],
                                                                    "source_lang": row["target_lang"],
                                                                    "target_lang": row["source_lang"]})
    if alignment_fpath is not None:
        with open(alignment_fpath) as scores_f:
            alignments = [float(row) for row in scores_f]

        alignments_iter = iter(alignments)
        new_tatoeba_dataset = new_tatoeba_dataset.map(lambda row: {**row, "alignment_scores": next(alignments_iter)})
        print("Filtering training data for %s down to %s samples" % (subset, sum(a >= args.alignment_scores_threshold for a in alignments)))
        new_tatoeba_dataset = new_tatoeba_dataset.filter(lambda row: row["alignment_scores"] >= args.alignment_scores_threshold)

    target_langs.append(tgt_lang_fl[0])
    new_tatoeba_dataset = new_tatoeba_dataset.map(lambda row: {"source_lang": src_lang_fl,
                                                               "target_lang": match_flores_langs(row["target_lang"])[0]})

    train_dataset_length += new_tatoeba_dataset._info.splits["train"].num_examples
    all_train_datasets.append(new_tatoeba_dataset)

print("Target training langs: %s" % target_langs)

train_dataset_fwd = interleave_datasets(all_train_datasets).shuffle(seed=42, buffer_size=train_dataset_length//args.train_data_buffer_ratio)
train_dataset_bwd = interleave_datasets(all_train_datasets).shuffle(seed=42, buffer_size=train_dataset_length//args.train_data_buffer_ratio)


def col_iterator_from_dataset(dataset: IterableDataset, column: str) -> Iterator[str]:
    dataset_iter = iter(dataset)
    if args.use_teacher_targets and column == "target_text":
        # pre-constructing targets with a teacher
        source_texts_batch = []
        for sample in dataset_iter:
            source_texts_batch.append(sample["source_text"])
            if len(source_texts_batch) >= args.teacher_targets_batch_size:
                teacher_outputs = teacher_wrapper.get(source_texts_batch)
                for output in teacher_outputs:
                    yield output
                source_texts_batch = []
    else:
        for sample in dataset_iter:
            yield sample[column]


def all_iterators_from_dataset(dataset: IterableDataset, keys: List[str]) -> List[Iterator[str]]:
    forked_datasets = itertools.tee(dataset, len(keys))
    return [col_iterator_from_dataset(forked_datasets[i], k) for i, k in enumerate(keys)]


iters_keys = ["source_text", "target_text", "source_lang", "target_lang"]

train_iters = {
    "fwd": dict(zip(iters_keys, all_iterators_from_dataset(train_dataset_fwd, iters_keys))),
    "bwd": dict(zip(iters_keys, all_iterators_from_dataset(train_dataset_bwd, iters_keys)))
}


# train_src_iter_fwd, train_tgt_iter_fwd, train_tgt_lang_iter_fwd = all_iterators_from_dataset(train_dataset_fwd)
# train_src_iter_bwd, train_tgt_iter_bwd, train_src_lang_iter_bwd = all_iterators_from_dataset(train_dataset_bwd)


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


STUDENT_MODEL_PATH = os.path.join(args.checkpoint_dir, "init_student_model")

if args.construct_new_student:
    student_model = construct_student_from_teacher(teacher_model,
                                                   args.teacher_model,
                                                   M2M100ForConditionalGeneration,
                                                   args.reset_weights)
    student_model.save_pretrained(STUDENT_MODEL_PATH)
    AutoTokenizer.from_pretrained(args.teacher_model).save_pretrained(STUDENT_MODEL_PATH)

# 3. Initialize two objectives: 1. forward with data: {lang}-{others}, 2. backward with data: {others}-{lang}
#   the data can be constructed in advance by concatenating all relevant subsets of HF Opus-100 dataset
lang_module = LangModule(STUDENT_MODEL_PATH)
evaluators = [BLEU()]

fwd_objective = BaselineSeq2Seq(lang_module=lang_module,
                              batch_size=args.batch_size,
                              val_evaluators=evaluators,

                              texts_or_path=train_iters["fwd"]["source_text"],
                              texts_langs=train_iters["fwd"]["source_lang"],
                              labels_or_path=train_iters["fwd"]["target_text"],
                              labels_langs=train_iters["fwd"]["target_lang"],
                              train_dataset_length=train_dataset_length,

                              val_texts_or_path=eval_dataset["source_text"],
                              val_texts_langs=eval_dataset["source_lang"],
                              val_labels_or_path=eval_dataset["target_text"],
                              val_labels_langs=eval_dataset["target_lang"],

                              # source_lang_id="en", target_lang_id="cs"
                              objective_id="%s->X" % args.src_lang
                              )
bwd_objective = BaselineSeq2Seq(lang_module,
                              batch_size=args.batch_size,
                              val_evaluators=evaluators,

                              texts_or_path=train_iters["fwd"]["source_text"],
                              texts_langs=train_iters["fwd"]["source_lang"],
                              labels_or_path=train_iters["fwd"]["target_text"],
                              labels_langs=train_iters["fwd"]["target_lang"],
                              train_dataset_length=train_dataset_length,

                              val_texts_or_path=eval_dataset["target_text"],
                              val_texts_langs=eval_dataset["target_lang"],
                              val_labels_or_path=eval_dataset["source_text"],
                              val_labels_langs=eval_dataset["source_lang"],

                              # source_lang_id="en", target_lang_id="cs"
                              objective_id="X->%s" % args.src_lang
                              )
train_objectives = [fwd_objective, bwd_objective]

missing_langs = set(target_langs) - set(fwd_objective.tokenizer.additional_special_tokens)
if missing_langs:
    raise ValueError("These langs are missing from the model vocab: %s" % missing_langs)

training_arguments = AdaptationArguments(output_dir=args.checkpoint_dir,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=args.effective_batch_size // args.batch_size,
                                         evaluation_strategy="steps",
                                         # log_level="critical",
                                         logging_steps=200,
                                         eval_steps=args.eval_steps,
                                         num_train_epochs=10,
                                         save_steps=args.save_steps,
                                         no_cuda=not USE_CUDA,
                                         )

schedule = ParallelSchedule(train_objectives, training_arguments)
adapter = Adapter(lang_module, schedule, training_arguments)
adapter.train()

# 4. Consider whether to weigh the distil loss with the standard seq2seq
