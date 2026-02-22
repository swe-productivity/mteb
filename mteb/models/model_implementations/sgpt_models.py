import numpy as np

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader
from mteb.types import PromptType

SGPT_CITATION = """@article{muennighoff2022sgpt,
  title={SGPT: GPT Sentence Embeddings for Semantic Search},
  author={Muennighoff, Niklas},
  journal={arXiv preprint arXiv:2202.08904},
  year={2022}
}"""


class _SGPTSpecBModel:
    def __init__(self, model_name: str, revision: str | None = None, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
        texts = [text for batch in inputs for text in batch["text"]]
        if prompt_type == PromptType.query:
            texts = ["[" + t + "]" for t in texts]
        else:
            texts = ["{" + t + "}" for t in texts]
        embeddings = self.model.encode(texts, **kwargs)
        if hasattr(embeddings, "cpu"):
            return embeddings.cpu().detach().float().numpy()
        return np.array(embeddings)


def sgpt_specb_loader(model_name: str, revision: str | None = None, **kwargs):
    return _SGPTSpecBModel(model_name, revision=revision, **kwargs)


bigscience__sgpt_bloom_7b1_msmarco = ModelMeta(
    name="bigscience/sgpt-bloom-7b1-msmarco",
    model_type=["dense"],
    revision="dc579f3d2d5a0795eba2049e16c3e36c74007ad3",
    release_date="2022-08-26",
    languages=None,
    loader=sentence_transformers_loader,
    n_parameters=7_100_000_000,
    n_embedding_parameters=1_026_793_472,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=4096,
    license=None,
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets={"MSMARCO"},
    adapted_from="/gpfsscratch/rech/six/commun/commun/experiments/muennighoff/bloomckpt/6b3/bloom-7b1",
    superseded_by=None,
    citation=SGPT_CITATION,
)

muennighoff__sgpt_5b8_weightedmean_msmarco_specb_bitfit = ModelMeta(
    name="Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="2dbba11efed19bb418811eac04be241ddc42eb99",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=5_900_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={"MSMARCO"},
    adapted_from="EleutherAI/gpt-j-6b",
    superseded_by=None,
    citation=SGPT_CITATION,
)
