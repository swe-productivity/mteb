from __future__ import annotations

from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata

_FINMTEB_CITATION = r"""
@inproceedings{tang-yang-2025-finmteb,
  author = {Tang, Yixuan and Yang, Yi},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  title = {FinMTEB: Finance Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2025.emnlp-main.179/},
  year = {2025},
}
"""


class CCKS2019Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="CCKS2019Clustering",
        description="A Chinese financial knowledge graph event clustering dataset from CCKS 2019.",
        reference="https://huggingface.co/datasets/FinanceMTEB/CCKS2019",
        dataset={
            "path": "FinanceMTEB/CCKS2019",
            "revision": "3ee6454e35b145c3c17413f0e3337b39fffdb1d5",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2019-01-01", "2019-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify the event type of a Chinese financial text",
        bibtex_citation=_FINMTEB_CITATION,
    )


class CCKS2020Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="CCKS2020Clustering",
        description="A Chinese financial knowledge graph event clustering dataset from CCKS 2020.",
        reference="https://huggingface.co/datasets/FinanceMTEB/CCKS2020",
        dataset={
            "path": "FinanceMTEB/CCKS2020",
            "revision": "747e32650174620b2ab1b21684a7a5b135a0980b",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2020-01-01", "2020-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify the event type of a Chinese financial text",
        bibtex_citation=_FINMTEB_CITATION,
    )


class CCKS2022Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="CCKS2022Clustering",
        description="A Chinese financial knowledge graph event clustering dataset from CCKS 2022.",
        reference="https://huggingface.co/datasets/FinanceMTEB/CCKS2022",
        dataset={
            "path": "FinanceMTEB/CCKS2022",
            "revision": "109293c66467e029ad33a11ee398791fa002aaac",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2022-01-01", "2022-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify the event type of a Chinese financial text",
        bibtex_citation=_FINMTEB_CITATION,
    )


class FinNLClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="FinNLClustering",
        description="A Chinese financial news categorization dataset. Given financial news, the model classifies it into one of fifteen categories including company, industry, broad market, China, foreign, international, economy, policy, politics, futures, bonds, real estate, foreign exchange, virtual currency, and energy.",
        reference="https://arxiv.org/abs/2302.09432",
        dataset={
            "path": "FinanceMTEB/FinNL",
            "revision": "e88f3020ce6bc8e67b1800fe20b071d58ddbb468",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Identify the category of a Chinese financial news article",
        bibtex_citation=_FINMTEB_CITATION,
    )


class MInDS14ZhClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="MInDS14ZhClustering",
        description="MINDS-14 is a dataset for intent detection in e-banking in Chinese, covering 14 intents.",
        reference="https://arxiv.org/pdf/2104.08524",
        dataset={
            "path": "FinanceMTEB/MInDS-14-zh",
            "revision": "f42bc3bba1506f41174f2457fc08ec82ab0de162",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify the intent of a Chinese banking customer inquiry",
        bibtex_citation=_FINMTEB_CITATION,
    )
