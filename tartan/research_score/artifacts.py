"""Write small metadata or checkpoints atomically on the destination volume."""
import json
import re
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile


def validate_output_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Output names must use ASCII letters, numbers, dots, hyphens or underscores")
    return name


def new_run_id(label):
    validate_output_name(label)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label


def publish(obj, path, is_json=False):
    if is_json:return publish_text(json.dumps(obj, indent=2) + '\n', path)
    path = Path(path)
    validate_output_name(path.name)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        import torch
        torch.save(obj, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_text(content, path):
    """Atomic metadata with cache invalidation for observed FSX stale null pages."""
    path=Path(path);validate_output_name(path.name);expected=content.encode('utf-8')
    if path.is_file():
        with path.open('rb') as handle:os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        if path.read_bytes()==expected:return
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(expected);handle.flush();os.fsync(handle.fileno())
            os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        os.replace(temporary,path)
        with path.open('rb') as handle:os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        if path.read_bytes()!=expected:raise OSError('metadata publication mismatch: '+str(path))
    finally:
        Path(temporary).unlink(missing_ok=True)
