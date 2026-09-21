from .core import SCHEMA_VERSION, cache_key

SOURCE_PREPROCESS_VERSION="source-loader-v1"

def assert_source_tensor_identity(original, adapted):
    if original.keys()!=adapted.keys(): raise AssertionError("source feature keys differ")
    for key in original:
        a,b=original[key],adapted[key]
        if getattr(a,"shape",None)!=getattr(b,"shape",None): raise AssertionError(f"{key} shape differs")
        if not (a==b).all(): raise AssertionError(f"{key} values differ")
