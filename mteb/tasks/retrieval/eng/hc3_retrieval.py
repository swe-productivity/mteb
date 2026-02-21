from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HC3Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HC3Retrieval",
        description="A human-ChatGPT comparison finance corpus for retrieval.",
        reference="https://arxiv.org/pdf/2301.07597",
        dataset={
            "path": "FinanceMTEB/HPC3-finance",
            "revision": "7018353fb281b866e5934eeb496251be4ad3585f",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Given a financial question, retrieve relevant answers"},
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
