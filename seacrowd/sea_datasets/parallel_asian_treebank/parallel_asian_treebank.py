import itertools
from pathlib import Path
from typing import List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_DATASETNAME = "parallel_asian_treebank"

_LANGUAGES = ["khm", "lao", "mya", "ind", "fil", "zlm", "tha", "vie"]
_LANGUAGES_TO_FILENAME_LANGUAGE_CODE = {
    "khm": "khm",
    "lao": "lo",
    "mya": "my",
    "ind": "id",
    "fil": "fil",
    "zlm": "ms",
    "tha": "th",
    "vie": "vi",
    "eng": "en",
    "hin": "hi",
    "jpn": "ja",
    "zho": "zh",
}
_LOCAL = False
_CITATION = """\
@inproceedings{riza2016introduction,
  title={Introduction of the asian language treebank},
  author={Riza, Hammam and Purwoadi, Michael and Uliniansyah, Teduh and Ti, Aw Ai and Aljunied, Sharifah Mahani and Mai, Luong Chi and Thang, Vu Tat and Thai, Nguyen Phuong and Chea, Vichet and Sam, Sethserey and others},
  booktitle={2016 Conference of The Oriental Chapter of International Committee for Coordination and Standardization of Speech Databases and Assessment Techniques (O-COCOSDA)},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT.
It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016).
Then, it was developed under ASEAN IVO.
The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages.
ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
"""

_HOMEPAGE = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = {
    "data": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/ALT-Parallel-Corpus-20191206.zip",
    "train": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-train.txt",
    "dev": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-dev.txt",
    "test": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-test.txt",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class ParallelAsianTreebank(datasets.GeneratorBasedBuilder):
    """The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT"""

    BUILDER_CONFIGS = []
    lang_combinations = list(itertools.combinations(_LANGUAGES_TO_FILENAME_LANGUAGE_CODE.keys(), 2))
    for lang_a, lang_b in lang_combinations:
        if lang_a not in _LANGUAGES and lang_b not in _LANGUAGES:
            # Don't create a subset if both languages are not from SEA
            pass
        else:
            BUILDER_CONFIGS.append(
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{lang_a}_{lang_b}_source",
                    version=_SOURCE_VERSION,
                    description=f"{_DATASETNAME} source schema",
                    schema="source",
                    subset_id=f"{_DATASETNAME}_{lang_a}_{lang_b}_source",
                )
            )
            BUILDER_CONFIGS.append(
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{lang_a}_{lang_b}_seacrowd_t2t",
                    version=_SOURCE_VERSION,
                    description=f"{_DATASETNAME} seacrowd schema",
                    schema="seacrowd_t2t",
                    subset_id=f"{_DATASETNAME}_{lang_a}_{lang_b}_seacrowd_t2t",
                )
            )

    def _info(self):
        # The features are the same for both source and seacrowd
        features = schemas.text2text_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        def _split_at_n(text: str, n: int) -> Tuple[str, str]:
            """Split text on the n-th instance"""
            return ("_".join(text.split("_")[:n]), "_".join(text.split("_")[n:]))

        _, subset = _split_at_n(self.config.subset_id, 3)
        lang_pair, _ = _split_at_n(subset, 2)
        lang_a, lang_b = lang_pair.split("_")

        data_dir = Path(dl_manager.download_and_extract(_URLS["data"])) / "ALT-Parallel-Corpus-20191206"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "lang_a": lang_a, "lang_b": lang_b, "split_file": dl_manager.download(_URLS["train"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "lang_a": lang_a, "lang_b": lang_b, "split_file": dl_manager.download(_URLS["test"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "lang_a": lang_a, "lang_b": lang_b, "split_file": dl_manager.download(_URLS["dev"])},
            ),
        ]

    def _generate_examples(self, data_dir: Path, lang_a: str, lang_b: str, split_file: str):
        with open(data_dir / f"data_{_LANGUAGES_TO_FILENAME_LANGUAGE_CODE[lang_a]}.txt", "r") as f:
            lang_a_texts = [line.strip() for line in f.readlines()]

        with open(data_dir / f"data_{_LANGUAGES_TO_FILENAME_LANGUAGE_CODE[lang_b]}.txt", "r") as f:
            lang_b_texts = [line.strip() for line in f.readlines()]

        with open(split_file, "r") as f:
            split_ref = [line.strip() for line in f.readlines()]

    # def _generate_examples(self, data_dir: str, split: str):

    #     if self.config.schema not in ["source", "seacrowd_t2t"]:
    #         raise ValueError(f"Invalid config: {self.config.name}")

    #     mapping_data = {}

    #     for language in _LANGUAGES:
    #         lines = open(f"{data_dir}/data_{_LANGUAGES_TO_FILENAME_LANGUAGE_CODE[language]}.txt.{split}", "r").readlines()

    #         for line in lines:
    #             id, sentence = line.split("\t")
    #             sentence = sentence.rsplit()

    #             if id not in mapping_data:
    #                 mapping_data[id] = {}

    #             mapping_data[id][language] = sentence

    #     combination_languages = list(itertools.combinations(_LANGUAGES, 2))
    #     breakpoint()

    #     i = 0

    #     for id in mapping_data:
    #         for each_pair in combination_languages:
    #             if each_pair[0] in mapping_data[id] and each_pair[1] in mapping_data[id]:
    #                 yield i, {
    #                     "id": f"{id}-{each_pair[0]}-{each_pair[1]}",
    #                     "text_1": mapping_data[id][each_pair[0]],
    #                     "text_2": mapping_data[id][each_pair[1]],
    #                     "text_1_name": each_pair[0],
    #                     "text_2_name": each_pair[1],
    #                 }

    #                 i += 1

    #                 yield i, {
    #                     "id": f"{id}-{each_pair[1]}-{each_pair[0]}",
    #                     "text_1": mapping_data[id][each_pair[1]],
    #                     "text_2": mapping_data[id][each_pair[0]],
    #                     "text_1_name": each_pair[1],
    #                     "text_2_name": each_pair[0],
    #                 }

    #                 i += 1
