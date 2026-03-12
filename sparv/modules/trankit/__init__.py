"""POS tagging, lemmatisation, dependency parsing and NER with Trankit."""

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
        "trankit.embedding",
        default="xlm-roberta-base",
        description="XLM-RoBERTa embedding variant to use ('xlm-roberta-base' or 'xlm-roberta-large')",
        datatype=str,
    ),
]
