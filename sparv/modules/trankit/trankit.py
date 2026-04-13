"""POS tagging, lemmatisation, dependency parsing and NER with Trankit."""

import os
import warnings

# Put HuggingFace into offline mode before trankit (and therefore transformers /
# huggingface_hub) is imported anywhere. The trankit plugin pre-fetches all the
# models it needs during provisioning, so at runtime there is nothing legitimate
# to download — but transformers will otherwise still issue cache-revalidation
# HEAD requests to huggingface.co for files like xlm-roberta-base/config.json,
# which stall for ~50s per file when the network is flaky. setdefault leaves an
# escape hatch for the provisioning job to override with HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sparv.api import Annotation, Config, Language, Output, SparvErrorMessage, Text, annotator, get_logger

logger = get_logger(__name__)

# Mapping from Sparv ISO 639-3 codes to Trankit language names.
_LANG_MAP = {
    "eng": "english",
    "fin": "finnish",
    "swe": "swedish",
}

# Subset of our supported languages that have a trained NER model in Trankit.
# Finnish and Swedish are not in Trankit's `langwithner` set.
_NER_LANGS = {"english"}


# ---------------------------------------------------------------------------
# Preloader
# ---------------------------------------------------------------------------

def _preload_pipeline(lang, cache_dir, use_gpu, embedding):
    """Load the Trankit Pipeline once per worker for reuse across all source files."""
    from trankit import Pipeline
    trankit_lang = _LANG_MAP.get(lang)
    logger.info("Preloading Trankit pipeline for language '%s'", trankit_lang)
    return _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding)


def _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding):
    """Construct a Pipeline while suppressing noisy third-party FutureWarnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return Pipeline(trankit_lang, gpu=use_gpu, cache_dir=cache_dir, embedding=embedding)


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

@annotator(
    "POS, lemma and dependency parsing with Trankit",
    language=["eng", "fin", "swe"],
    preloader=_preload_pipeline,
    preloader_params=["lang", "cache_dir", "use_gpu", "embedding"],
    preloader_target="pipeline",
    preloader_shared=False,  # Each worker loads its own copy; safer with PyTorch/CUDA
)
def annotate(
    corpus_text: Text = Text(),
    lang: Language = Language(),
    sentence_chunk: Annotation = Annotation("[trankit.sentence_chunk]"),
    out_sentence: Output = Output(
        "trankit.sentence", cls="sentence", description="Sentence segments from Trankit"
    ),
    out_token: Output = Output(
        "trankit.token", cls="token", description="Token segments from Trankit"
    ),
    out_upos: Output = Output(
        "<token>:trankit.upos", cls="token:upos", description="Universal POS tags from Trankit"
    ),
    out_pos: Output = Output(
        "<token>:trankit.pos", cls="token:pos", description="Language-specific POS tags from Trankit"
    ),
    out_baseform: Output = Output(
        "<token>:trankit.baseform", cls="token:baseform", description="Lemmas from Trankit"
    ),
    out_feats: Output = Output(
        "<token>:trankit.ufeats",
        cls="token:ufeats",
        description="Universal morphological features from Trankit",
    ),
    out_deprel: Output = Output(
        "<token>:trankit.deprel", cls="token:deprel", description="Dependency relations from Trankit"
    ),
    out_ref: Output = Output(
        "<token>:trankit.ref",
        cls="token:ref",
        description="Sentence-relative token index from Trankit",
    ),
    out_dephead_ref: Output = Output(
        "<token>:trankit.dephead_ref",
        cls="token:dephead_ref",
        description="Sentence-relative dependency head positions from Trankit",
    ),
    out_dephead: Output = Output(
        "<token>:trankit.dephead",
        cls="token:dephead",
        description="Absolute dependency head positions from Trankit",
    ),
    cache_dir: str = Config("trankit.cache_dir"),
    use_gpu: bool = Config("trankit.use_gpu"),
    embedding: str = Config("trankit.embedding"),
    pipeline: object = None,  # Injected by the preloader when running under 'sparv preload'
):
    """Annotate corpus with Trankit: tokenization, sentence segmentation, POS, lemma and depparse.

    Use 'sparv preload' to load the XLM-RoBERTa model once per worker and share
    it across all source files, avoiding a cold start for each file.
    """
    trankit_lang = _LANG_MAP.get(lang)
    if trankit_lang is None:
        raise SparvErrorMessage(
            f"Language '{lang}' is not supported by the trankit annotator. "
            f"Supported languages: {', '.join(_LANG_MAP)}"
        )

    text_data = corpus_text.read()
    text_spans = list(sentence_chunk.read_spans())

    # Collect non-empty chunks with their corpus offsets.
    chunks = [
        (text_span[0], text_data[text_span[0]:text_span[1]])
        for text_span in text_spans
        if text_data[text_span[0]:text_span[1]].strip()
    ]

    if not chunks:
        _write_empty_outputs(
            out_sentence, out_token, out_upos, out_pos, out_baseform,
            out_feats, out_ref, out_deprel, out_dephead_ref, out_dephead,
        )
        return

    # Load pipeline if the preloader wasn't used (cold start for this file).
    if pipeline is None:
        try:
            from trankit import Pipeline
        except ImportError:
            raise SparvErrorMessage(
                "Could not import trankit. Install it into the Sparv virtual environment, "
                "or run 'sparv preload' to use the preloader."
            )
        logger.info("Loading Trankit pipeline for language '%s'", trankit_lang)
        pipeline = _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding)

    # Progress bar: one step per chunk, plus one for the final write.
    logger.progress(total=len(chunks) + 1)

    sentence_segments = []
    token_segments = []
    upos_list = []
    pos_list = []
    baseform_list = []
    feats_list = []
    ref_list = []
    deprel_list = []
    dephead_ref_list = []
    dephead_list = []

    # global_token_count tracks total tokens written so far, used to compute
    # corpus-absolute dephead values across sentences and chunks.
    global_token_count = 0

    for offset, chunk_text in chunks:
        result = pipeline(chunk_text)

        for sent in result.get("sentences", []):
            sent_dspan = sent.get("dspan", [0, 0])
            sentence_segments.append((offset + sent_dspan[0], offset + sent_dspan[1]))

            tokens = sent.get("tokens", [])
            sent_len = len(tokens)

            for token in tokens:
                tok_dspan = token.get("dspan", [0, 0])
                token_segments.append((offset + tok_dspan[0], offset + tok_dspan[1]))

                upos_list.append(token.get("upos") or "_")
                pos_list.append(token.get("xpos") or "_")
                baseform_list.append(token.get("lemma") or "_")
                feats_list.append(token.get("feats") or "_")
                ref_list.append(str(token.get("id", "")))
                deprel_list.append(token.get("deprel") or "_")

                head = token.get("head") or 0
                dephead_ref_list.append(str(head) if head > 0 else "")
                dephead_list.append(str(head - 1 + global_token_count) if head > 0 else "-")

            global_token_count += sent_len

        logger.progress()  # One chunk done

    out_sentence.write(sentence_segments)
    out_token.write(token_segments)
    out_upos.write(upos_list)
    out_pos.write(pos_list)
    out_baseform.write(baseform_list)
    out_feats.write(feats_list)
    out_ref.write(ref_list)
    out_deprel.write(deprel_list)
    out_dephead_ref.write(dephead_ref_list)
    out_dephead.write(dephead_list)

    logger.progress()  # Write step done


@annotator(
    "Named entity recognition with Trankit",
    language=["eng"],
    preloader=_preload_pipeline,
    preloader_params=["lang", "cache_dir", "use_gpu", "embedding"],
    preloader_target="pipeline",
    preloader_shared=False,
)
def annotate_ner(
    corpus_text: Text = Text(),
    lang: Language = Language(),
    sentence_chunk: Annotation = Annotation("[trankit.sentence_chunk]"),
    out_ne: Output = Output(
        "trankit.ne", cls="named_entity", description="Named entity segments from Trankit"
    ),
    out_ne_type: Output = Output(
        "trankit.ne:trankit.ne_type",
        cls="token:named_entity_type",
        description="Named entity types from Trankit",
    ),
    cache_dir: str = Config("trankit.cache_dir"),
    use_gpu: bool = Config("trankit.use_gpu"),
    embedding: str = Config("trankit.embedding"),
    pipeline: object = None,
):
    """Named entity recognition with Trankit (English only)."""
    trankit_lang = _LANG_MAP.get(lang)

    text_data = corpus_text.read()
    text_spans = list(sentence_chunk.read_spans())

    chunks = [
        (text_span[0], text_data[text_span[0]:text_span[1]])
        for text_span in text_spans
        if text_data[text_span[0]:text_span[1]].strip()
    ]

    if not chunks:
        out_ne.write([])
        out_ne_type.write([])
        return

    if pipeline is None:
        try:
            from trankit import Pipeline
        except ImportError:
            raise SparvErrorMessage(
                "Could not import trankit. Install it into the Sparv virtual environment, "
                "or run 'sparv preload' to use the preloader."
            )
        logger.info("Loading Trankit pipeline for language '%s'", trankit_lang)
        pipeline = _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding)

    logger.progress(total=len(chunks) + 1)

    ne_segments = []
    ne_types = []

    for offset, chunk_text in chunks:
        result = pipeline(chunk_text)

        for sent in result.get("sentences", []):
            tokens = sent.get("tokens", [])
            ne_start = None
            ne_type_val = None
            for token in tokens:
                ner_tag = token.get("ner") or "O"
                if ner_tag.startswith("B-"):
                    if ne_start is not None:
                        ne_segments.append(ne_start)
                        ne_types.append(ne_type_val)
                    tok_dspan = token["dspan"]
                    ne_start = (offset + tok_dspan[0], offset + tok_dspan[1])
                    ne_type_val = ner_tag[2:]
                elif ner_tag.startswith("I-") and ne_start is not None:
                    tok_dspan = token["dspan"]
                    ne_start = (ne_start[0], offset + tok_dspan[1])
                else:
                    if ne_start is not None:
                        ne_segments.append(ne_start)
                        ne_types.append(ne_type_val)
                    ne_start = None
                    ne_type_val = None
            # Flush any open entity at sentence end
            if ne_start is not None:
                ne_segments.append(ne_start)
                ne_types.append(ne_type_val)

        logger.progress()

    out_ne.write(ne_segments)
    out_ne_type.write(ne_types)

    logger.progress()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_empty_outputs(
    out_sentence, out_token, out_upos, out_pos, out_baseform,
    out_feats, out_ref, out_deprel, out_dephead_ref, out_dephead,
):
    """Write empty annotation lists when there is no input text."""
    for output in (
        out_sentence, out_token, out_upos, out_pos, out_baseform,
        out_feats, out_ref, out_deprel, out_dephead_ref, out_dephead,
    ):
        output.write([])
