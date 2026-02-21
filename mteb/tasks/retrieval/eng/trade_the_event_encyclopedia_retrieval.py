from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TradeTheEventEncyclopediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TradeTheEventEncyclopediaRetrieval",
        description="A retrieval task based on financial terms and their explanations.",
        reference="https://aclanthology.org/2021.findings-acl.186.pdf",
        dataset={
            "path": "FinanceMTEB/TradeTheEventEncyclopedia",
            "revision": "7fa70ba6624011d65a311df86193b4b5587969bc",
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
