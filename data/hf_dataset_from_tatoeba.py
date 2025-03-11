import logging
import os
from functools import partial
from typing import List, Tuple, Union, Iterator, Dict

from datasets import DatasetDict, Dataset, Value, Features
from tqdm import tqdm

logger = logging.getLogger()

# local:
# TATOEBA_DATA_DIR = 'data/example_data_dir'
# TARGET_HF_PATH = "michal-stefanik/toy_tatoeba_dataset"
#
# TARGET_LANG: Union[str, None] = None
# SOURCE_LANGS: Union[str, None] = None

# lumi:
TATOEBA_DATA_DIR = '/scratch/project_462000447/data/Tatoeba-Challenge/v2023-09-26'
TARGET_HF_PATH = "michal-stefanik/tatoeba_mt_ces-x"

TARGET_LANG: Union[str, None] = "ces"
SOURCE_LANGS: Union[str, None] = "eng"


subdirs = os.listdir(TATOEBA_DATA_DIR)

lang_subdirs = [sdir for sdir in subdirs if len(sdir.split("-")) == 2
                and len(sdir.split("-")[0]) <= 3 and len(sdir.split("-")[1]) <= 3]

logger.warning("Directory %s contains %s lang pairs", TATOEBA_DATA_DIR, len(lang_subdirs))

if TARGET_LANG is not None:
    lang_subdirs = [l for l in lang_subdirs if TARGET_LANG in l]
    logger.warning("Subsetting to %s lang pairs containing %s", len(lang_subdirs), TARGET_LANG)

if SOURCE_LANGS:
    lang_subdirs = [l for l in lang_subdirs if any(src_l in l for src_l in SOURCE_LANGS.split(","))]
    logger.warning("Subsetting to %s lang pairs containing %s", len(lang_subdirs), SOURCE_LANGS)


def read_rows(dir: str, split: str) -> Tuple[List[str], List[str]]:
    path = os.path.join(dir, split)
    if os.path.exists(path+".src"):
        with open(path+".src") as src_f:
            with open(path+".trg") as trg_f:
                # TODO: convert to stream in cases exceeding file size of cs-de (sucessfully processed on 512GB RAM):
                #  -rw-rw-r-- 1 tiedeman project_462000447 3,1G okt  1  2023 train.src.gz
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


def generate_entries(dir: str, split: str) -> Union[None, Iterator[Dict[str, str]]]:
    path = os.path.join(dir, split)
    if os.path.exists(path+".src"):
        with open(path+".src") as src_f:
            with open(path+".trg") as trg_f:
                # TODO: convert to stream in cases exceeding file size of cs-de (sucessfully processed on 512GB RAM):
                #  -rw-rw-r-- 1 tiedeman project_462000447 3,1G okt  1  2023 train.src.gz
                for src, tgt in zip(src_f.readlines(), trg_f.readlines()):
                    yield {"source_text": src.strip(),
                           "target_text": tgt.strip(),
                           "source_lang": subdir.split("-")[0],
                           "target_lang": subdir.split("-")[1]}
    elif os.path.exists(path+".src.gz"):
        import io
        import gzip

        with io.TextIOWrapper(io.BufferedReader(gzip.open(path+".src.gz"))) as src_f:  # type: ignore
            with io.TextIOWrapper(io.BufferedReader(gzip.open(path+".trg.gz"))) as trg_f:  # type: ignore
                for src, tgt in zip(src_f.readlines(), trg_f.readlines()):
                    yield {"source_text": src.strip(),
                           "target_text": tgt.strip(),
                           "source_lang": subdir.split("-")[0],
                           "target_lang": subdir.split("-")[1]}
    else:
        logger.warning("Path %s not found, skipping", path)
        return False


for subdir in tqdm(lang_subdirs, desc="Uploading langs"):
    logger.warning("Processing lang subdir %s", subdir)

    subdir_path = os.path.join(TATOEBA_DATA_DIR, subdir)
    lang_pair_dataset = DatasetDict()
    for split in ["train", "val", "dev", "test"]:
        if split == "dev":
            split = "val"

        data_generator = generate_entries(subdir_path, split)
        if not data_generator:
            # No files found for the current split
            continue
        try:
            next(data_generator)
        except StopIteration:
            # some of the iterated documents are empty
            continue

        # explicit schema definition
        features = Features({"source_text": Value("string"), "target_text": Value("string"),
                             "source_lang": Value("string"), "target_lang": Value("string")})
        lang_pair_dataset[split] = Dataset.from_generator(partial(generate_entries, dir=subdir_path, split=split), features=features)

    logger.warning("Pushing subset %s with splits %s into %s", subdir, lang_pair_dataset.keys(), TARGET_HF_PATH)
    lang_pair_dataset.push_to_hub(TARGET_HF_PATH, config_name=subdir)
