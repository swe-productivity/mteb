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


class DISCFinLLMReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DISCFinLLMReranking",
        description="A Chinese financial scenario QA reranking dataset.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DISCFinLLM-reranking",
            "revision": "e97ba5a0620442edb312b1f8c0ab4fe04096ed59",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map_at_1000",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve relevant answers"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return
        _load_old_reranking_format(self)


class FinEvaReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinEvaReranking",
        description="A Chinese financial scenario QA reranking dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaRetrieval-reranking",
            "revision": "20a5ca0e51f8948f59dd79ee648e9d7f2f76f25f",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map_at_1000",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve relevant answers"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return
        _load_old_reranking_format(self)
