import os
from typing import Iterable, List

import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer


class CachedTeacherTranslator:

    def __init__(self, teacher_model: PreTrainedModel, teacher_tokenizer: PreTrainedTokenizer, cache_dir: str):
        self.model = teacher_model
        self.tokenizer = teacher_tokenizer
        self.device = self.model.device
        cache_file = f"{self.model.config.name_or_path.replace( '/', '-')}.tsv"
        cache_dir = os.path.join(cache_dir, cache_file)

        self.cache_writer = open(cache_dir, "a")
        self.cache = pd.read_csv(cache_dir, sep="\t", names=["a", "b"], error_bad_lines=False, on_bad_lines="warn",
                                 ).set_index("a", drop=True)["b"].drop_duplicates().to_dict()

    def get(self, input_strs: List[str]) -> List[str]:
        try:
            return [self.cache[input_str] for input_str in input_strs]
        except KeyError:
            inputs = self.tokenizer(input_strs, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.model.generate(**inputs)
            output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for input_str, output_str in zip(input_strs, output_strs):
                self.cache_writer.write(f"{input_str}\t{output_str}\n")
            return output_strs
