from __future__ import annotations

from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
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


class ComplaintsClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="ComplaintsClustering",
        description="The Consumer Complaint Database is a collection of complaints about consumer financial products and services that were sent to companies for response.",
        reference="https://huggingface.co/datasets/CFPB/consumer-finance-complaints",
        dataset={
            "path": "FinanceMTEB/Complaints",
            "revision": "6704122294b7693f5e544cdde1e4a3e80b291b76",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2010-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Identify the category of a consumer financial complaint",
        bibtex_citation=_FINMTEB_CITATION,
    )


class FinanceArxivP2PClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="FinanceArxivP2PClustering",
        description="Clustering of titles and abstracts from financial arxiv papers (q-fin).",
        reference="https://arxiv.org/",
        dataset={
            "path": "FinanceMTEB/FinanceArxiv-p2p",
            "revision": "5cc8768f04d39e9a16b84fc236dec93d8b173b64",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2023-12-31"),
        domains=["Academic", "Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Identify the main category of a financial arxiv paper based on its title and abstract",
        bibtex_citation=_FINMTEB_CITATION,
    )


class FinanceArxivS2SClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="FinanceArxivS2SClustering",
        description="Clustering of titles from financial arxiv papers (q-fin).",
        reference="https://arxiv.org/",
        dataset={
            "path": "FinanceMTEB/FinanceArxiv-s2s",
            "revision": "78f66d3bbea9b1d7f11df84bc55b21b24302dcee",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2023-12-31"),
        domains=["Academic", "Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Identify the main category of a financial arxiv paper based on its title",
        bibtex_citation=_FINMTEB_CITATION,
    )


class MInDS14EnClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="MInDS14EnClustering",
        description="MINDS-14 is a dataset for intent detection in e-banking, covering 14 intents across 14 languages.",
        reference="https://arxiv.org/pdf/2104.08524",
        dataset={
            "path": "FinanceMTEB/MInDS-14-en",
            "revision": "141ac6a9010b851452a9327edfda190d37399b15",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify the intent of a banking customer inquiry",
        bibtex_citation=_FINMTEB_CITATION,
    )


class PiiClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="PiiClustering",
        description="Synthetic financial documents containing Personally Identifiable Information (PII).",
        reference="https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual",
        dataset={
            "path": "FinanceMTEB/synthetic_pii_finance_en",
            "revision": "5021671d60a324d576a7b57e4c4e13bfcf857a4d",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2020-01-01", "2023-12-31"),
        domains=["Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Identify the document type of a synthetic financial document",
        bibtex_citation=_FINMTEB_CITATION,
    )


class WikiCompany2IndustryClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="WikiCompany2IndustryClustering",
        description="Clustering of companies to their related industry domain based on company descriptions.",
        reference="https://aclanthology.org/W18-6532.pdf",
        dataset={
            "path": "FinanceMTEB/WikiCompany2Industry-en",
            "revision": "9b7c45122b764fc1e09c5b29cee887cfa4f8f395",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2018-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written", "Financial"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Identify the industry sector of a company based on its description",
        bibtex_citation=_FINMTEB_CITATION,
    )
