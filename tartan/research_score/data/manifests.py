import json
from pathlib import Path

def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
