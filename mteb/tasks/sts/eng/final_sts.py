from __future__ import annotations

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class FINALSts(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FINALSts",
        description="A dataset for discovering financial signals in narrative financial reports, used as a semantic textual similarity benchmark.",
        reference="https://aclanthology.org/2023.acl-long.800.pdf",
        dataset={
            "path": "FinanceMTEB/Final",
            "revision": "00506d0f1853ebe9fcc5112fb36a4cc4cc521695",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2023-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{tang-yang-2025-finmteb,
  author = {Tang, Yixuan and Yang, Yi},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  title = {FinMTEB: Finance Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2025.emnlp-main.179/},
  year = {2025},
}
""",
    )
