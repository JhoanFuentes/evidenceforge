"""SciFact in BEIR format. Labels are used by evaluation only, never by retrieval."""

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .io import download, fingerprint, sha256_file, write_json

DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
DATA_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
DATA_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return (self.title + "\n" + self.text).strip()


@dataclass
class Dataset:
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    manifest: dict


def prepare_scifact(destination: Path, archive: Path | None = None) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = destination / "scifact.zip"
        if not archive.exists() or sha256_file(archive) != DATA_SHA256:
            download(DATA_URL, archive, expected_sha256=DATA_SHA256)
    if sha256_file(archive) != DATA_SHA256:
        raise ValueError("SciFact archive SHA-256 mismatch; refusing to extract")
    # Extract only the known files, never trust archive paths or symlinks.
    with zipfile.ZipFile(archive) as bundle:
        for name in DATA_FILES:
            info = bundle.getinfo("scifact/" + name)
            if info.file_size > 100 * 1024 * 1024:
                raise ValueError("Dataset file exceeds size limit")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(info))
    manifest = {
        "name": "BEIR/SciFact", "source": DATA_URL, "sha256": DATA_SHA256,
        "upstream_md5": DATA_MD5,
        "files": {name: sha256_file(destination / name) for name in DATA_FILES},
        "license_notes": "See docs/DATA_AND_MODELS.md; dataset not covered by project MIT license",
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_dataset(path: Path, split: str = "test") -> Dataset:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name in DATA_FILES:
        if sha256_file(path / name) != manifest["files"][name]:
            raise ValueError(f"Dataset file changed: {name}")
    rows = _read_jsonl(path / "corpus.jsonl")
    documents = sorted((Document(str(r["_id"]), r["title"], r["text"]) for r in rows),
                       key=lambda d: d.id)
    doc_ids = {d.id for d in documents}
    if len(doc_ids) != len(documents):
        raise ValueError("Duplicate document IDs")
    rows = _read_jsonl(path / "queries.jsonl")
    all_queries = {str(r["_id"]): r["text"] for r in rows}
    if len(rows) != len(all_queries):
        raise ValueError("Duplicate query IDs")
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qid, doc_id = row["query-id"], row["corpus-id"]
            if qid not in all_queries or doc_id not in doc_ids:
                raise ValueError("Qrels reference an unknown query or document")
            if doc_id in qrels.get(qid, {}):
                raise ValueError("Duplicate relevance judgment")
            qrels.setdefault(qid, {})[doc_id] = int(row["score"])
    queries = {qid: all_queries[qid] for qid in sorted(qrels)}
    return Dataset(documents, queries, qrels, {
        **manifest, "split": split, "corpus_count": len(documents),
        "query_count": len(queries), "doc_order_sha256": fingerprint([d.id for d in documents]),
        "corpus_sha256": manifest["files"]["corpus.jsonl"],
    })
