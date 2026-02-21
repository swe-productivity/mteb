from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ESGClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ESGClassification",
        description="A finance dataset for sentence classification under the environmental, social, and corporate governance (ESG) framework.",
        reference="https://huggingface.co/datasets/FinanceMTEB/ESG",
        dataset={
            "path": "FinanceMTEB/ESG",
            "revision": "521d56feabadda80b11d6adcc6b335d4c5ad8285",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Classify a given financial sentence by its ESG category",
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
