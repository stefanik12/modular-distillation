import logging
import os
from typing import List, Tuple, Union

from datasets import DatasetDict, Dataset
from tqdm import tqdm

logger = logging.getLogger()

TATOEBA_DATA_DIR = 'data/example_data_dir'
TARGET_HF_PATH = "michal-stefanik/toy_tatoeba_dataset"

TARGET_LANG: Union[str, None] = None

subdirs = os.listdir(TATOEBA_DATA_DIR)

lang_subdirs = [sdir for sdir in subdirs if len(sdir.split("-")) == 2
                and len(sdir.split("-")[0]) <= 3 and len(sdir.split("-")[1]) <= 3]

logger.warning("Directory %s contains %s lang pairs", TATOEBA_DATA_DIR, len(lang_subdirs))

if TARGET_LANG is not None:
    lang_subdirs = [l for l in lang_subdirs if TARGET_LANG in l]
    logger.warning("Subsetting to %s lang pairs containing %s", len(lang_subdirs), TARGET_LANG)


def read_rows(dir: str, split: str) -> Tuple[List[str], List[str]]:
    path = os.path.join(dir, split)
    if os.path.exists(path+".src"):
        with open(path+".src") as src_f:
            with open(path+".trg") as trg_f:
                return [s.strip() for s in src_f.readlines()], [s.strip() for s in trg_f.readlines()]

    elif os.path.exists(path+".src.gz"):
        import io
        import gzip

        with io.TextIOWrapper(io.BufferedReader(gzip.open(path+".src.gz"))) as src_f:  # type: ignore
            with io.TextIOWrapper(io.BufferedReader(gzip.open(path+".trg.gz"))) as trg_f:  # type: ignore
                return [s.strip() for s in src_f.readlines()], [s.strip() for s in trg_f.readlines()]
    else:
        logger.warning("Path %s not found, skipping", path)
        return [], []


for subdir in tqdm(lang_subdirs, desc="Uploading langs"):
    logger.warning("Processing lang subdir %s", subdir)
    # if TARGET_LANG is not None and TARGET_LANG not in subdir:
    #     logger.warning("Skipping lang pair %s", subdir)
    #     continue

    subdir_path = os.path.join(TATOEBA_DATA_DIR, subdir)
    lang_pair_dataset = DatasetDict()
    for split in ["train", "val", "dev", "test"]:
        if split == "dev":
            split = "val"
        src, trg = read_rows(subdir_path, split)
        if src and trg:
            lang_pair_dataset[split] = Dataset.from_dict({"source_text": src, "target_text": trg,
                                                          "source_lang": [subdir.split("-")[0]]*len(src),
                                                          "target_lang": [subdir.split("-")[1]]*len(trg)
                                                          })

    logger.warning("Pushing subset %s with splits %s into %s", subdir, lang_pair_dataset.keys(), TARGET_HF_PATH)
    lang_pair_dataset.push_to_hub(TARGET_HF_PATH, config_name=subdir)
