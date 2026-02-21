from __future__ import annotations

from collections import defaultdict

from mteb.abstasks.retrieval import AbsTaskRetrieval
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


def _load_old_reranking_format(task: AbsTaskRetrieval) -> None:
    """Load old-format reranking dataset with query/positive/negative columns."""
    import datasets
    from datasets import Dataset

    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData

    raw = datasets.load_dataset(**task.metadata.dataset)

    task.dataset = defaultdict(lambda: defaultdict(dict))  # type: ignore[assignment]

    for split in raw:
        split_data = raw[split]
        split_data = split_data.filter(
            lambda x: len(x["positive"]) > 0 and len(x["negative"]) > 0
        )

        corpus = []
        queries = []
        relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
        top_ranked: dict[str, list[str]] = defaultdict(list)

        for idx, example in enumerate(split_data):
            query_id = f"{split}_query{idx}"
            queries.append({"id": query_id, "text": example["query"]})

            for i, pos in enumerate(example["positive"]):
                doc_id = f"apositive_{query_id}_{str(i).zfill(5)}"
                corpus.append({"id": doc_id, "title": "", "text": pos})
                top_ranked[query_id].append(doc_id)
                relevant_docs[query_id][doc_id] = 1

            for i, neg in enumerate(example["negative"]):
                doc_id = f"negative_{query_id}_{str(i).zfill(5)}"
                corpus.append({"id": doc_id, "title": "", "text": neg})
                top_ranked[query_id].append(doc_id)
                relevant_docs[query_id][doc_id] = 0

        task.dataset["default"][split] = RetrievalSplitData(
            corpus=Dataset.from_list(corpus),
            queries=Dataset.from_list(queries),
            relevant_docs=relevant_docs,
            top_ranked=top_ranked,
        )

    task.data_loaded = True


class FinFactReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinFactReranking",
        description="A benchmark dataset for financial fact checking and explanation generation.",
        reference="https://arxiv.org/pdf/2309.08793",
        dataset={
            "path": "FinanceMTEB/FinFact-reranking",
            "revision": "70435b4984c687051ecf38b9d2686f33345dcced",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2023-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a financial claim, retrieve relevant evidence passages"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return
        _load_old_reranking_format(self)


class FiQA2018Reranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FiQA2018Reranking",
        description="Financial opinion mining and question answering reranking task.",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "FinanceMTEB/FiQA-reranking",
            "revision": "f6934c7980c19a3acb8aeba3b66b4766fbb4b9db",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2018-01-01", "2018-12-31"),
        domains=["Web", "Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Given a financial question, retrieve relevant answers"},
        bibtex_citation=_FINMTEB_CITATION,
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return
        _load_old_reranking_format(self)


class HC3Reranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HC3Reranking",
        description="A human-ChatGPT comparison finance corpus reranking task.",
        reference="https://arxiv.org/pdf/2301.07597",
        dataset={
            "path": "FinanceMTEB/HPC3-reranking",
            "revision": "538a43bb89e86fd40a8d1b9c7e630f8c3aa67a6c",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2023-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Given a financial question, retrieve relevant answers"},
        bibtex_citation=_FINMTEB_CITATION,
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return
        _load_old_reranking_format(self)
