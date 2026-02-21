from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TheGoldmanEnRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TheGoldmanEnRetrieval",
        description="A retrieval task based on the Goldman Sachs Financial Dictionary.",
        reference="https://huggingface.co/datasets/FinanceMTEB/TheGoldmanEncyclopedia-en",
        dataset={
            "path": "FinanceMTEB/TheGoldmanEncyclopedia-en",
            "revision": "c33ce9ced2224b9c6d9e425196edc36f53400ec1",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Given a financial term, retrieve the relevant definition"},
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
