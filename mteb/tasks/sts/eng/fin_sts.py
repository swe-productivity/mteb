from __future__ import annotations

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class FinSTSSts(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="FinSTSSts",
        description="A benchmark for detecting subtle semantic shifts in financial narratives.",
        reference="https://arxiv.org/pdf/2403.14341",
        dataset={
            "path": "FinanceMTEB/FinSTS",
            "revision": "09e270b1afe87a65dd41c6292e3c8905220bc290",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2024-01-01", "2024-12-31"),
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
