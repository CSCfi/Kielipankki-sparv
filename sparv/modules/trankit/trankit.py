"""Tokenization, POS tagging, lemmatisation, dependency parsing and NER with Trankit.

The module is split into three annotators so that tokenization is done exactly
once, by Trankit, and every downstream annotator works on that tokenization:

- `tokenize` produces `trankit.sentence` and `trankit.token` from the source
  text (the corpus config is expected to bind the `sentence` and `token`
  classes to these).
- `annotate` and `annotate_ner` read the existing `<sentence>`/`<token>`
  annotations and feed Trankit *pretokenized* input, so their output
  attributes are aligned with `<token>` by construction. (A previous version
  let Trankit re-tokenize the text internally, which produced attribute lists
  of a different length than the token annotation and broke the exports.)
"""

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


# ---------------------------------------------------------------------------
# Preloader
# ---------------------------------------------------------------------------

def _set_torch_threads(use_gpu, threads):
    """Set the CPU thread count for Trankit's torch inference.

    Sparv runs each annotator as a Snakemake job, and Snakemake exports
    OMP_NUM_THREADS set to the rule's thread count (1 by default), which would
    otherwise pin Trankit's transformer inference to a single core. torch's
    runtime set_num_threads takes precedence over that inherited value. `threads`
    of 0 means "use all available cores".

    Gated on actual CUDA availability, not the `use_gpu` config flag: Trankit
    treats `use_gpu` as best-effort and silently falls back to CPU when CUDA is
    absent (the common case here), so keying off the flag would skip the thread
    setting on exactly the CPU-only hosts that need it.
    """
    import torch
    if use_gpu and torch.cuda.is_available():
        return  # Actually running on GPU; CPU thread count is irrelevant.
    n = threads or os.cpu_count() or 1
    torch.set_num_threads(n)
    logger.info("Using %d CPU thread(s) for Trankit inference", n)


def _preload_pipeline(lang, cache_dir, use_gpu, embedding, threads):
    """Load the Trankit Pipeline once per worker for reuse across all source files."""
    from trankit import Pipeline
    trankit_lang = _LANG_MAP.get(lang)
    logger.info("Preloading Trankit pipeline for language '%s'", trankit_lang)
    return _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding, threads)


def _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding, threads):
    """Construct a Pipeline while suppressing noisy third-party FutureWarnings."""
    _set_torch_threads(use_gpu, threads)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return Pipeline(trankit_lang, gpu=use_gpu, cache_dir=cache_dir, embedding=embedding)


def _get_pipeline(pipeline, lang, cache_dir, use_gpu, embedding, threads):
    """Return the preloaded pipeline, or load one (cold start for this file)."""
    trankit_lang = _LANG_MAP.get(lang)
    if trankit_lang is None:
        raise SparvErrorMessage(
            f"Language '{lang}' is not supported by the trankit annotator. "
            f"Supported languages: {', '.join(_LANG_MAP)}"
        )
    if pipeline is not None:
        return pipeline
    try:
        from trankit import Pipeline
    except ImportError:
        raise SparvErrorMessage(
            "Could not import trankit. Install it into the Sparv virtual environment, "
            "or run 'sparv preload' to use the preloader."
        )
    logger.info("Loading Trankit pipeline for language '%s'", trankit_lang)
    return _load_pipeline(Pipeline, trankit_lang, use_gpu, cache_dir, embedding, threads)


# ---------------------------------------------------------------------------
# Annotators
# ---------------------------------------------------------------------------

@annotator(
    "Sentence segmentation and tokenization with Trankit",
    language=["eng", "fin", "swe"],
    preloader=_preload_pipeline,
    preloader_params=["lang", "cache_dir", "use_gpu", "embedding", "threads"],
    preloader_target="pipeline",
    preloader_shared=False,  # Each worker loads its own copy; safer with PyTorch/CUDA
)
def tokenize(
    corpus_text: Text = Text(),
    lang: Language = Language(),
    chunk: Annotation = Annotation("[trankit.sentence_chunk]"),
    out_sentence: Output = Output(
        "trankit.sentence", cls="sentence", description="Sentence segments from Trankit"
    ),
    out_token: Output = Output(
        "trankit.token", cls="token", description="Token segments from Trankit"
    ),
    cache_dir: str = Config("trankit.cache_dir"),
    use_gpu: bool = Config("trankit.use_gpu"),
    embedding: str = Config("trankit.embedding"),
    threads: int = Config("trankit.threads"),
    pipeline: object = None,  # Injected by the preloader when running under 'sparv preload'
):
    """Segment text into sentences and tokens with Trankit.

    This is the only annotator that runs Trankit on raw text; `annotate` and
    `annotate_ner` consume the resulting tokenization. Use 'sparv preload' to
    load the XLM-RoBERTa model once per worker and share it across all source
    files, avoiding a cold start for each file.
    """
    text_data = corpus_text.read()
    text_spans = list(chunk.read_spans())

    # Collect non-empty chunks with their corpus offsets.
    chunks = [
        (text_span[0], text_data[text_span[0]:text_span[1]])
        for text_span in text_spans
        if text_data[text_span[0]:text_span[1]].strip()
    ]

    if not chunks:
        out_sentence.write([])
        out_token.write([])
        return

    pipeline = _get_pipeline(pipeline, lang, cache_dir, use_gpu, embedding, threads)

    # Progress bar: one step per chunk, plus one for the final write.
    logger.progress(total=len(chunks) + 1)

    sentence_segments = []
    token_segments = []

    for offset, chunk_text in chunks:
        result = pipeline.tokenize(chunk_text)
        if not result:
            logger.progress()
            continue

        for sent in result.get("sentences", []):
            # Trankit's token dspans occasionally include surrounding whitespace
            # (notably a trailing newline in hard-wrapped text) or even span
            # internal whitespace. A token containing whitespace breaks every
            # downstream consumer that assumes a token holds none — most visibly
            # TreeTagger, which receives tokens newline-separated and would then
            # emit more rows than it was given. Normalise each token span to
            # maximal runs of non-whitespace characters.
            sent_tokens = []
            for token in sent.get("tokens", []):
                tok_dspan = token.get("dspan", [0, 0])
                sent_tokens.extend(
                    _clean_token_spans(text_data, offset + tok_dspan[0], offset + tok_dspan[1])
                )

            if not sent_tokens:
                continue  # Skip sentences with no actual tokens (e.g. whitespace only)

            sent_dspan = sent.get("dspan", [0, 0])
            sentence_segments.append((offset + sent_dspan[0], offset + sent_dspan[1]))
            token_segments.extend(sent_tokens)

        logger.progress()  # One chunk done

    out_sentence.write(sentence_segments)
    out_token.write(token_segments)

    logger.progress()  # Write step done


@annotator(
    "POS, lemma and dependency parsing with Trankit",
    language=["eng", "fin", "swe"],
    preloader=_preload_pipeline,
    preloader_params=["lang", "cache_dir", "use_gpu", "embedding", "threads"],
    preloader_target="pipeline",
    preloader_shared=False,
)
def annotate(
    lang: Language = Language(),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    token: Annotation = Annotation("<token>"),
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
    threads: int = Config("trankit.threads"),
    pipeline: object = None,
):
    """POS tag, lemmatize and dependency parse existing tokens with Trankit.

    Reads the corpus tokenization (`<token>` grouped by `<sentence>`, normally
    produced by `trankit:tokenize`) and feeds it to Trankit pretokenized, so
    every output attribute has exactly one value per token.
    """
    sentences_all, orphans = sentence.get_children(token)
    if orphans:
        logger.warning(
            "Found %d tokens not belonging to any sentence. These will not be annotated.",
            len(orphans),
        )

    word_list = list(word.read())

    upos = word.create_empty_attribute()
    pos = word.create_empty_attribute()
    baseform = word.create_empty_attribute()
    feats = word.create_empty_attribute()
    ref = word.create_empty_attribute()
    deprel = word.create_empty_attribute()
    dephead_ref = word.create_empty_attribute()
    dephead = word.create_empty_attribute()

    sentences = [s for s in sentences_all if s]
    pretokenized = [[_model_token(word_list[i]) for i in s] for s in sentences]

    if pretokenized:
        pipeline = _get_pipeline(pipeline, lang, cache_dir, use_gpu, embedding, threads)

        # Progress bar: tagging/parsing, lemmatization, writing.
        logger.progress(total=3)

        tagged_doc = pipeline.posdep(pretokenized)["sentences"]
        logger.progress()

        # The public lemmatize() entry point discards POS tags for pretokenized
        # input (obmit_tag=True), which degrades lemmas for ambiguous word
        # forms. Chain the tagger output into the lemmatizer the same way
        # Pipeline.__call__ does, keeping tag-conditioned lemmatization.
        tagged_doc = pipeline._lemmatize_doc(tagged_doc)
        logger.progress()

        for sent, tagged_sent in zip(sentences, tagged_doc, strict=True):
            for w_index, w in zip(sent, tagged_sent["tokens"], strict=True):
                upos[w_index] = w.get("upos") or "_"
                pos[w_index] = w.get("xpos") or "_"
                baseform[w_index] = w.get("lemma") or "_"
                feats[w_index] = w.get("feats") or "_"
                ref[w_index] = str(w.get("id", ""))
                deprel[w_index] = w.get("deprel") or "_"

                head = w.get("head") or 0
                dephead_ref[w_index] = str(head) if head > 0 else ""
                dephead[w_index] = str(sent[head - 1]) if head > 0 else "-"

    out_upos.write(upos)
    out_pos.write(pos)
    out_baseform.write(baseform)
    out_feats.write(feats)
    out_ref.write(ref)
    out_deprel.write(deprel)
    out_dephead_ref.write(dephead_ref)
    out_dephead.write(dephead)

    if pretokenized:
        logger.progress()  # Write step done


@annotator(
    "Named entity recognition with Trankit",
    language=["eng"],  # Of our languages, only English has a Trankit NER model
    preloader=_preload_pipeline,
    preloader_params=["lang", "cache_dir", "use_gpu", "embedding", "threads"],
    preloader_target="pipeline",
    preloader_shared=False,
)
def annotate_ner(
    lang: Language = Language(),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    token: Annotation = Annotation("<token>"),
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
    threads: int = Config("trankit.threads"),
    pipeline: object = None,
):
    """Named entity recognition with Trankit (English only).

    Reads the corpus tokenization (`<token>` grouped by `<sentence>`) and runs
    only Trankit's NER component on it, building entity spans from the BIO tag
    sequence over the token spans.
    """
    sentences_all, _orphans = sentence.get_children(token)

    word_list = list(word.read())
    token_spans = list(token.read_spans())

    sentences = [s for s in sentences_all if s]
    pretokenized = [[_model_token(word_list[i]) for i in s] for s in sentences]

    if not pretokenized:
        out_ne.write([])
        out_ne_type.write([])
        return

    pipeline = _get_pipeline(pipeline, lang, cache_dir, use_gpu, embedding, threads)

    # Progress bar: NER tagging, writing.
    logger.progress(total=2)

    ner_doc = pipeline.ner(pretokenized)["sentences"]
    logger.progress()

    ne_segments = []
    ne_types = []

    for sent, ner_sent in zip(sentences, ner_doc, strict=True):
        ne_start = None
        ne_type_val = None
        for w_index, w in zip(sent, ner_sent["tokens"], strict=True):
            ner_tag = w.get("ner") or "O"
            tok_span = token_spans[w_index]
            if ner_tag.startswith("B-"):
                if ne_start is not None:
                    ne_segments.append(ne_start)
                    ne_types.append(ne_type_val)
                ne_start = (tok_span[0], tok_span[1])
                ne_type_val = ner_tag[2:]
            elif ner_tag.startswith("I-") and ne_start is not None:
                ne_start = (ne_start[0], tok_span[1])
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

    out_ne.write(ne_segments)
    out_ne_type.write(ne_types)

    logger.progress()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_token_spans(text, start, end):
    """Yield (start, end) sub-spans covering maximal runs of non-whitespace in text[start:end].

    Trankit token dspans sometimes include leading/trailing whitespace (notably a
    trailing newline in hard-wrapped source text) or even span internal whitespace.
    Emitting such a span as a single token breaks every downstream consumer that
    assumes a token holds no whitespace — most visibly TreeTagger, which receives
    tokens newline-separated and would then emit more rows than it was given.
    Splitting on whitespace here keeps `trankit.token` whitespace-free by
    construction; an all-whitespace span yields nothing and is dropped.
    """
    i = start
    while i < end:
        while i < end and text[i].isspace():
            i += 1
        if i >= end:
            break
        run_start = i
        while i < end and not text[i].isspace():
            i += 1
        yield (run_start, i)


def _model_token(text):
    """Return token text as fed to Trankit; it rejects empty/whitespace-only tokens."""
    return text if text.strip() else "_"
