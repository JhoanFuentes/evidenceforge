"""Portable, deterministic artifact utilities. No pickle or executable data formats."""

import hashlib
import json
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def download(url: str, target: Path, *, expected_sha256: str | None = None) -> None:
    """HTTPS only; bounded read, verified TLS, atomic replace after checksum validation."""
    if not url.startswith("https://"):
        raise ValueError("Downloads require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    # Include certifi's CA set when installed; retain the platform's trust roots.
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    request = urllib.request.Request(url, headers={"User-Agent": "EvidenceForge/0.1"})
    temp = target.with_suffix(target.suffix + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=90, context=context) as response:
            if not response.url.startswith("https://"):
                raise ValueError("Insecure download redirect")
            with temp.open("wb") as stream:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 512 * 1024 * 1024:
                        raise ValueError("Download exceeds 512 MiB limit")
                    stream.write(chunk)
        if expected_sha256 and sha256_file(temp) != expected_sha256:
            raise ValueError("Download checksum mismatch")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
