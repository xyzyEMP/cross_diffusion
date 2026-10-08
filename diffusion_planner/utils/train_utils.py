"""Read checkpoint-normalizer JSON from local or mmengine-supported storage."""
import json
from mmengine import fileio

def openjson(path):
    return json.loads(fileio.get_text(path))
