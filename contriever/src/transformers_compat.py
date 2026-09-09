# Compatibility shim for transformers >= 5 (e.g. TokenizersBackend), which removed
# the legacy tokenizer methods `batch_encode_plus` / `encode_plus`. They map 1:1
# onto the modern `tokenizer(...)` entry point. No-op on transformers < 5).
import importlib

_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    try:
        tub = importlib.import_module("transformers.tokenization_utils_base")
    except Exception:
        return

    def _batch_encode_plus(self, *args, **kwargs):
        return self(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs):
        return self(*args, **kwargs)

    for klass in (tub.PreTrainedTokenizerBase,):
        if not hasattr(klass, "batch_encode_plus"):
            klass.batch_encode_plus = _batch_encode_plus
        if not hasattr(klass, "encode_plus"):
            klass.encode_plus = _encode_plus