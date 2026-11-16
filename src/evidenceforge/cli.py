"""EvidenceForge command line. Retrieval is local; only `answer` calls a paid API."""

import argparse
import json
import sys
from pathlib import Path

from .dataset import Document, load_dataset, prepare_scifact
from .evidence import Answer, Citation, EvidenceDocument, document_errors, validate_answer
from .io import fingerprint, write_json
from .retrieval import BM25, reciprocal_rank_fusion


def build_pack(query: str, documents: list[Document], hits, method: str) -> dict:
    by_id = {d.id: d for d in documents}
    pack = {
        "schema_version": 1, "query": query, "method": method,
        "documents": [{"id": hit.document_id, "title": by_id[hit.document_id].title,
                       "text": by_id[hit.document_id].text} for hit in hits],
        "retrieval": [hit.to_dict() for hit in hits],
        "notice": "Retrieved evidence; relevance does not mean support or truth. Scores are not confidence.",
    }
    pack["fingerprint"] = fingerprint(pack)
    return pack


def read_pack(path: Path) -> tuple[dict, list[EvidenceDocument]]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Evidence pack exceeds 2 MiB")
    pack = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict) or pack.get("schema_version") != 1:
        raise ValueError("Unsupported evidence pack")
    expected = pack.get("fingerprint")
    if expected != fingerprint({k: v for k, v in pack.items() if k != "fingerprint"}):
        raise ValueError("Evidence pack fingerprint mismatch")
    if not isinstance(pack.get("query"), str) or not pack["query"].strip():
        raise ValueError("Evidence pack requires a query")
    if not isinstance(pack.get("documents"), list):
        raise ValueError("Invalid evidence pack documents")
    documents = []
    for raw in pack["documents"]:
        if not isinstance(raw, dict) or set(raw) != {"id", "title", "text"}:
            raise ValueError("Invalid evidence document fields")
        documents.append(EvidenceDocument(**raw))
    if document_errors(documents):
        raise ValueError("Invalid evidence documents")
    return pack, documents


def demo(output: Path) -> dict:
    """Authored toy data for a network-free contract demo, not a quality benchmark."""
    documents = [
        Document("runbook", "Snapshot recovery", "A checkpoint stores the last committed state. Recovery resumes from that checkpoint."),
        Document("index", "Index compatibility", "An embedding index must match the model and document collection used to build it."),
        Document("retrieval", "Relevance is not truth", "A relevant document may contradict the query. Retrieval scores are not truth probabilities."),
        Document("citations", "Citation checks", "An exact quote check verifies text occurrence. It does not prove that the answer follows from the quote."),
    ]
    query = "How does recovery use a checkpoint?"
    hits = BM25(documents).search(query, 2)
    pack = build_pack(query, documents, hits, "bm25")
    evidence = [EvidenceDocument(**d) for d in pack["documents"]]
    valid = Answer("answer", "Recovery resumes from the saved checkpoint.",
                   [Citation("runbook", "Recovery resumes from that checkpoint.")])
    fabricated = Answer("answer", "Recovery is always lossless.",
                        [Citation("runbook", "Recovery is always lossless.")])
    unsupported = Answer("answer", "Checkpoints guarantee zero downtime.",
                         [Citation("runbook", "A checkpoint stores the last committed state.")])
    result = {"kind": "authored contract demo; no model inference", "evidence": pack,
              "cases": {
                  "exact_quote": {"answer": valid.to_dict(), "validation": validate_answer(valid, evidence)},
                  "fabricated_quote": {"answer": fabricated.to_dict(), "validation": validate_answer(fabricated, evidence)},
                  "unsupported_conclusion_with_real_quote": {
                      "answer": unsupported.to_dict(), "validation": validate_answer(unsupported, evidence)},
              }, "lesson": "The unsupported conclusion passes quote integrity. Semantic grounding requires its own evaluation."}
    write_json(output / "evidence.json", pack)
    write_json(output / "demo.json", result)
    return {"output": str(output), "top_document": hits[0].document_id,
            "exact_quote_valid": result["cases"]["exact_quote"]["validation"]["valid"],
            "fabricated_quote_valid": result["cases"]["fabricated_quote"]["validation"]["valid"],
            "unsupported_conclusion_quote_valid": result["cases"]["unsupported_conclusion_with_real_quote"]["validation"]["valid"],
            "lesson": result["lesson"]}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="EvidenceForge: auditable retrieval experiments")
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("demo", help="Run authored, offline citation-boundary demo")
    command.add_argument("--output", type=Path, default=Path("runs/demo"))
    command = commands.add_parser("download", help="Download checksum-pinned BEIR/SciFact")
    command.add_argument("--data", type=Path, default=Path("data/scifact"))
    command.add_argument("--archive", type=Path)
    command = commands.add_parser("download-model", help="Download the pinned BGE ONNX revision")
    command.add_argument("--model-dir", type=Path, default=Path("models/bge-small-en-v1.5"))
    command = commands.add_parser("benchmark", help="Run the frozen three-system retrieval experiment")
    command.add_argument("--data", type=Path, default=Path("data/scifact"))
    command.add_argument("--config", type=Path, default=Path("configs/scifact-v1.json"))
    command.add_argument("--model-dir", type=Path, default=Path("models/bge-small-en-v1.5"))
    command.add_argument("--index", type=Path, default=Path("indexes/scifact-v1"))
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("search", help="Search locally and save the exact evidence pack")
    command.add_argument("query")
    command.add_argument("--method", choices=["bm25", "dense", "rrf"], default="bm25")
    command.add_argument("--data", type=Path, default=Path("data/scifact"))
    command.add_argument("--model-dir", type=Path, default=Path("models/bge-small-en-v1.5"))
    command.add_argument("--index", type=Path, default=Path("indexes/scifact-v1"))
    command.add_argument("--top-k", type=int, default=5)
    command.add_argument("--output", type=Path, default=Path("runs/evidence.json"))
    command = commands.add_parser("case", help="Inspect a saved benchmark query with local source text; no inference")
    command.add_argument("--query-id", required=True)
    command.add_argument("--results", type=Path, default=Path("examples/scifact-v1/results.json"))
    command.add_argument("--data", type=Path, default=Path("data/scifact"))
    command.add_argument("--top-k", type=int, default=5)
    command.add_argument("--output", type=Path, default=Path("runs/case.json"))
    command = commands.add_parser("answer", help="Send one evidence pack to OpenAI; requires key and explicit model")
    command.add_argument("--pack", type=Path, required=True)
    command.add_argument("--model", required=True, help="Explicit model ID; prefer a dated snapshot")
    command.add_argument("--max-output-tokens", type=int, default=2000)
    command.add_argument("--output", type=Path, default=Path("runs/answer.json"))
    command = commands.add_parser("report", help="Render a standalone HTML benchmark report")
    command.add_argument("results", type=Path)
    command.add_argument("--output", type=Path, default=Path("runs/report.html"))
    command = commands.add_parser("verify", help="Recompute published metrics and paired intervals")
    command.add_argument("results", type=Path)
    return root


def run(args) -> dict:
    if args.command == "demo":
        return demo(args.output)
    if args.command == "download":
        return prepare_scifact(args.data, args.archive)
    if args.command == "download-model":
        from .embeddings import prepare_model
        return prepare_model(args.model_dir)
    if args.command == "benchmark":
        from .experiment import benchmark, validate_config
        from .report import write_report
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        dataset = load_dataset(args.data, config["split"])
        result = benchmark(dataset, config, args.model_dir, args.index, args.output)
        write_report(result, args.output / "report.html")
        return {"systems": {name: system["aggregate"] for name, system in result["systems"].items()},
                "output": str(args.output)}
    if args.command == "search":
        if not 1 <= args.top_k <= 100 or not args.query.strip():
            raise ValueError("Search requires a nonempty query and top-k from 1 to 100")
        dataset = load_dataset(args.data)
        sparse_hits = BM25(dataset.documents).search(args.query, 100)
        hits = sparse_hits
        if args.method in {"dense", "rrf"}:
            from .embeddings import Encoder, corpus_vectors
            from .retrieval import ExactDense
            encoder = Encoder(args.model_dir)
            vectors, _ = corpus_vectors(dataset.documents, encoder, args.index)
            dense = ExactDense([doc.id for doc in dataset.documents], vectors)
            hits = dense.search(encoder.encode([args.query], query=True)[0], 100)
            if args.method == "rrf":
                hits = reciprocal_rank_fusion([sparse_hits, hits], 100)
        pack = build_pack(args.query, dataset.documents, hits[:args.top_k], args.method)
        write_json(args.output, pack)
        return {"output": str(args.output), "fingerprint": pack["fingerprint"],
                "hits": [{"document_id": d["id"], "title": d["title"]} for d in pack["documents"]]}
    if args.command == "case":
        from .experiment import verify_result
        if type(args.top_k) is not int or not 1 <= args.top_k <= 100:
            raise ValueError("Case inspection requires top-k from 1 to 100")
        verify_result(args.results)
        result = json.loads(args.results.read_text(encoding="utf-8"))
        dataset = load_dataset(args.data, result["dataset"]["split"])
        if dataset.manifest["corpus_sha256"] != result["dataset"]["corpus_sha256"]:
            raise ValueError("Local corpus differs from the saved benchmark corpus")
        qid = args.query_id
        if qid not in result["qrels"] or qid not in dataset.queries or qid not in dataset.qrels:
            raise ValueError("Unknown query ID in benchmark or local dataset")
        if dataset.qrels[qid] != result["qrels"][qid]:
            raise ValueError("Local relevance judgments differ from the saved benchmark")
        by_id = {document.id: document for document in dataset.documents}
        judgments = result["qrels"][qid]
        if any(document_id not in by_id for document_id in judgments):
            raise ValueError("Saved relevance judgments contain an unknown document ID")
        systems = {}
        for name, system in result["systems"].items():
            if qid not in system["rankings"] or qid not in system["per_query"]:
                raise ValueError("Saved system is missing the requested query ID")
            ranking = system["rankings"][qid]
            if any(document_id not in by_id for document_id in ranking):
                raise ValueError("Saved ranking contains an unknown document ID")
            systems[name] = {
                "scores": system["per_query"][qid],
                "documents": [{"rank": rank, "document_id": document_id,
                               "title": by_id[document_id].title, "text": by_id[document_id].text,
                               "is_judged_relevant": judgments.get(document_id, 0) > 0}
                              for rank, document_id in enumerate(ranking[:args.top_k], 1)],
            }
        notice = ("Relevance is not support or truth: relevant abstracts may support or contradict the query. "
                  "Judgments can be incomplete; is_judged_relevant=false does not establish irrelevance. "
                  "This file contains local source text; rankings are saved results, not fresh inference.")
        case = {"schema_version": 1, "query_id": qid, "query": dataset.queries[qid],
                "split": result["dataset"]["split"], "corpus_sha256": dataset.manifest["corpus_sha256"],
                "top_k": args.top_k, "systems": systems, "notice": notice}
        write_json(args.output, case)
        return {"output": str(args.output), "query_id": qid,
                "systems": {name: {"scores": system["scores"], "documents": [
                    {key: value for key, value in document.items() if key != "text"}
                    for document in system["documents"]]} for name, system in systems.items()},
                "notice": notice}
    if args.command == "answer":
        from .providers import OpenAIResponses, ProviderError
        pack, documents = read_pack(args.pack)
        provider = OpenAIResponses(model=args.model, max_output_tokens=args.max_output_tokens)
        try:
            answer = provider.generate(pack["query"], documents)
        except ProviderError as error:
            # Rejected output may still consume tokens. Persist only safe adapter
            # metadata; never the HTTP body, request headers, or invalid answer.
            write_json(args.output, {
                "status": "error", "error_code": error.code,
                "model": args.model, "usage": provider.last_usage,
                "response_id": provider.response_id, "validation": provider.last_validation,
                "evidence_fingerprint": pack["fingerprint"],
            })
            raise
        result = {"status": "completed", "answer": answer.to_dict(), "validation": provider.last_validation,
                  "model": args.model, "usage": provider.last_usage,
                  "response_id": provider.response_id, "evidence_fingerprint": pack["fingerprint"]}
        write_json(args.output, result)
        return result
    if args.command == "verify":
        from .experiment import verify_result
        return verify_result(args.results)
    if args.command == "report":
        from .report import write_report
        write_report(json.loads(args.results.read_text(encoding="utf-8")), args.output)
        return {"output": str(args.output)}
    raise ValueError("Unknown command")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as error:
        print(f"EvidenceForge: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0
