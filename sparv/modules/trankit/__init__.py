"""Tokenization, POS tagging, lemmatisation, dependency parsing and NER with Trankit."""

from sparv.api import Config

from . import trankit

__config__ = [
    Config(
        "trankit.sentence_chunk",
        default="<text>",
        description="Text chunk annotation to use as input for sentence segmentation",
        datatype=str,
    ),
    Config(
        "trankit.cache_dir",
        default="/app/trankit/cache",
        description="Directory where Trankit stores downloaded model files",
        datatype=str,
    ),
    Config(
        "trankit.use_gpu",
        default=True,
        description="Use GPU instead of CPU if available",
        datatype=bool,
    ),
    Config(
        "trankit.threads",
        default=0,
        description="Number of CPU threads Trankit may use for inference. 0 means use all "
        "available cores. Ignored when running on GPU. Sparv runs each annotator as a "
        "Snakemake job with OMP_NUM_THREADS=1, which would otherwise pin inference to a "
        "single core; lower this only to avoid oversubscription when several corpora are "
        "processed concurrently.",
        datatype=int,
    ),
    Config(
        "trankit.embedding",
        default="xlm-roberta-base",
        description="XLM-RoBERTa embedding variant to use ('xlm-roberta-base' or 'xlm-roberta-large')",
        datatype=str,
    ),
]
