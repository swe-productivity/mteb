from __future__ import annotations

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


class AlphaFinRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlphaFinRetrieval",
        description="A Chinese financial Q&A dataset including NLI, financial QA, and stock trend predictions.",
        reference="https://github.com/AlphaFin-proj/AlphaFin",
        dataset={
            "path": "FinanceMTEB/AlphaFin",
            "revision": "f901493ea0dcb2faaad9624f586ece8238b06a52",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
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


class DISCFinLLMComputingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DISCFinLLMComputingRetrieval",
        description="A Chinese financial scenario QA dataset including a computational retrieval task.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DISCFinLLM-Computing",
            "revision": "2342751577b08c5ee989174fdac8f08d6d7f3e88",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve relevant passages"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class DISCFinLLMRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DISCFinLLMRetrieval",
        description="A Chinese financial scenario QA dataset including a retrieval task.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DISCFinLLM-Retrieval",
            "revision": "00fb9b68b7e204b1fd03a3433ad81cbc6282aa0c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve relevant passages"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class DuEEFinRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DuEEFinRetrieval",
        description="A Chinese financial news bulletin event retrieval dataset.",
        reference="https://github.com/FudanDISC/DISC-FinLLM/",
        dataset={
            "path": "FinanceMTEB/DuEE-fin",
            "revision": "1129a95b1b81a298497929be04e8e681da48eba4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2022-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question about news events, retrieve relevant passages"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class FinEvaEncyclopediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinEvaEncyclopediaRetrieval",
        description="A Chinese financial dataset providing terminology used in the financial industry.",
        reference="https://github.com/alipay/financial_evaluation_dataset",
        dataset={
            "path": "FinanceMTEB/FinEvaEncyclopedia",
            "revision": "3f4e6b66d58ba1514718e3823f0cb818609dce25",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial term, retrieve the relevant definition"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class FinEvaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinEvaRetrieval",
        description="A Chinese financial scenario QA dataset including a retrieval task.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinEvaRetrieval",
            "revision": "ddd402799f8c6d8c8c7ec662a18641e276cfad31",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve relevant passages"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class FinTruthQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinTruthQARetrieval",
        description="A Chinese dataset for evaluating the quality of financial information disclosure.",
        reference="https://arxiv.org/pdf/2406.12009",
        dataset={
            "path": "FinanceMTEB/FinTruthQA",
            "revision": "73d569f6d438e430261a924f418ea99981196209",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-06-30"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve the relevant passage"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class SmoothNLPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SmoothNLPRetrieval",
        description="A Chinese financial Q&A retrieval dataset.",
        reference="https://huggingface.co/datasets/FinanceMTEB/SmoothNLP",
        dataset={
            "path": "FinanceMTEB/SmoothNLPNews",
            "revision": "2d2476d29f78ed46e818b00ba589d7d756839595",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial question, retrieve the relevant answer"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class TheGoldmanZhRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TheGoldmanZhRetrieval",
        description="A retrieval task based on the Goldman Sachs Financial Dictionary in Chinese.",
        reference="https://huggingface.co/datasets/FinanceMTEB/TheGoldmanEncyclopedia-zh",
        dataset={
            "path": "FinanceMTEB/TheGoldmanEncyclopedia-zh",
            "revision": "09cf73149a1e1a81b32ab9e968cdfef9a7a4a1a5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial term, retrieve the relevant definition"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )


class THUCNewsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="THUCNewsRetrieval",
        description="A Chinese financial news retrieval dataset from THUCNews.",
        reference="https://huggingface.co/datasets/FinanceMTEB/THUCNews",
        dataset={
            "path": "FinanceMTEB/THUCNews",
            "revision": "e80c8838c291b31e3335064b6463947b9d178667",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2010-01-01", "2016-12-31"),
        domains=["News", "Written", "Financial"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a Chinese financial news headline, retrieve the full article"
        },
        bibtex_citation=_FINMTEB_CITATION,
    )
