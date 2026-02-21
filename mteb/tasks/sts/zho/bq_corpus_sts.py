from __future__ import annotations

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class BQCorpusSts(AbsTaskSTS):
    min_score = 0
    max_score = 1

    metadata = TaskMetadata(
        name="BQCorpusSts",
        description="Bank Question Corpus: A Chinese corpus for sentence semantic equivalence identification (SSEI).",
        reference="https://aclanthology.org/D18-1536/",
        dataset={
            "path": "FinanceMTEB/bq_corpus",
            "revision": "24a4a7cfa6fb8ab07f214809fefed1cd4e8250cb",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=("2018-01-01", "2018-12-31"),
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
