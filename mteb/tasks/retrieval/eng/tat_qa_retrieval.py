from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TATQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TATQARetrieval",
        description="A question answering benchmark on a hybrid of tabular and textual content in finance.",
        reference="https://arxiv.org/pdf/2105.07624",
        dataset={
            "path": "FinanceMTEB/TATQA",
            "revision": "11b15221dd850044dc2261ce633e692851c8b7e2",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Given a financial question, retrieve the relevant passage"},
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
