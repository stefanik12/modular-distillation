from typing import List, Iterable, Iterator, Union, Dict, Optional, Any

import torch
from adaptor.lang_module import LangModule
from adaptor.objectives.distillation import Distillation
from adaptor.objectives.seq2seq import Sequence2Sequence, SequentialMixin
from transformers import BatchEncoding


class DistilledSeq2Seq(Distillation, Sequence2Sequence):
    pass


class DistilledNLLB(DistilledSeq2Seq):

    def __init__(self,
                 *args,
                 texts_langs: Iterable[str],
                 labels_langs: Iterable[str],
                 val_texts_langs: Iterable[str],
                 val_labels_langs: Iterable[str],
                 # use_teacher_targets: bool = False,  # on-the-fly generation not supported for now
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.texts_langs = texts_langs
        self.labels_langs = labels_langs
        self.val_texts_langs = val_texts_langs
        self.val_labels_langs = val_labels_langs
        self.use_teacher_targets = False

    def _get_inputs_iterator(self, split: str) -> Iterator[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        # TODO: zip the output iterator with the definition of input and output languages
        """
        Creates a default iterator over encodings with aligned input and output texts.
        :param split: Data split. `train` or `eval`.
        :return: Iterator of model input encodings.
        """
        source_texts_iter, target_texts_iter = self._per_split_iterators(split)
        src_langs = self.texts_langs if split == "train" else self.val_texts_langs
        tgt_langs = self.labels_langs if split == "train" else self.val_labels_langs

        collated_iter = self._get_seq2seq_collated_iterator(source_texts_iter, target_texts_iter, src_langs, tgt_langs)

        return collated_iter

    def _get_teacher_ids(self, teacher_inputs: BatchEncoding) -> torch.LongTensor:
        with torch.no_grad():
            teacher_outputs = self.teacher_model.generate(**teacher_inputs.to(self.teacher_model.device),
                                                          max_length=len(teacher_inputs['input_ids'][0]))
        if (teacher_inputs["labels"] < 0).any():
            ignore_loss_id = teacher_inputs["labels"][teacher_inputs["labels"] < 0][0]
            # excluding labels from the loss -> this should be pertained
            for batch_i in range(len(teacher_outputs)):
                sample_eos_pos = torch.where(teacher_outputs[batch_i] == self.tokenizer.eos_token_id)[0].max()
                teacher_outputs[batch_i, sample_eos_pos+1:] = ignore_loss_id
        return teacher_outputs

    def _get_seq2seq_collated_iterator(self,
                                       source_texts: Iterable[str],
                                       target_texts: Iterable[str],
                                       source_langs: Iterable[str],
                                       target_langs: Iterable[str]) -> Iterator[BatchEncoding]:
        """
        Creates an iterator over batches of encoded `source_texts` as inputs and `target_texts` as labels.
        Override this to implement custom mapping, or unsupervised seq2seq objective. See e.g. BackTranslation.
        :param source_texts: Input texts.
        :param target_texts: Output (expected) texts to translate input texts into.
        :return: Iterator of encoded batches.
        """
        features_batch = []
        for source_text, target_text, source_lang, target_lang in zip(source_texts, target_texts,
                                                                      source_langs, target_langs):
            # TODO: this should be fine, but let's check the encoded inputs
            self.tokenizer.src_lang = source_lang
            self.tokenizer.tgt_lang = target_lang
            sample_features = self.tokenizer(source_text, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                sample_targets = self.tokenizer(target_text, truncation=True)

            features_batch.append({"input_ids": sample_features.input_ids,
                                   "attention_mask": sample_features.attention_mask,
                                   "labels": sample_targets.input_ids})
            if len(features_batch) == self.batch_size:
                out_batch = self.collator(features_batch)
                if self.use_teacher_targets:
                    out_batch["labels"] = self._get_teacher_ids(out_batch)
                yield out_batch
                features_batch = []

        if features_batch:
            # yield last nonempty residual batch
            out_batch = self.collator(features_batch)
            if self.use_teacher_targets:
                out_batch["labels"] = self._get_teacher_ids(out_batch)
            yield out_batch

    def register_compatible_head_model(self, lang_module: LangModule,
                                       other_objective: Optional["Objective"] = None,
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None,
                                       merge_objective_module: bool = True) -> torch.nn.Module:

        head_module = super(DistilledSeq2Seq, self).register_compatible_head_model(lang_module,
                                                                                   other_objective,
                                                                                   objective_args_for_head_config,
                                                                                   preloaded_module,
                                                                                   merge_objective_module)
        return head_module

