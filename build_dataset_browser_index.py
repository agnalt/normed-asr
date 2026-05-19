from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = BASE_DIR / "artifacts_local/processed/final_dataset"
DEFAULT_INDEX_PATH = DEFAULT_DATASET_DIR / "browser_index.sqlite"
TERM_CATEGORIES = ("selected", "common", "excluded")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def iter_metadata_paths(dataset_dir: Path) -> list[Path]:
    metadata_dir = dataset_dir / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Missing metadata directory: {metadata_dir}")
    paths = sorted(metadata_dir.glob("chunk_*.jsonl"))
    if not paths:
        raise ValueError(f"No metadata shards found under: {metadata_dir}")
    return paths


def iter_metadata_rows(dataset_dir: Path) -> tuple[Path, dict[str, Any]]:
    for path in iter_metadata_paths(dataset_dir):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield path, json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON in {path}:{line_number}") from error


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def flatten_terms(used_terms: dict[str, list[str]]) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    for category in TERM_CATEGORIES:
        for term in used_terms.get(category, []):
            terms.append((str(term), category))
    return terms


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;

        DROP TABLE IF EXISTS samples;
        DROP TABLE IF EXISTS sample_terms;
        DROP TABLE IF EXISTS terms;

        CREATE TABLE samples (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            text_norm TEXT NOT NULL,
            terms_text_norm TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            sample_rate INTEGER NOT NULL,
            language_key TEXT,
            conditioning_audio_id TEXT,
            conditioning_audio_file TEXT,
            exaggeration REAL,
            cfg_weight REAL,
            trimmed_trailing_artifact INTEGER NOT NULL,
            tts_chunks_json TEXT NOT NULL,
            used_terms_json TEXT NOT NULL,
            metadata_source TEXT NOT NULL
        );

        CREATE TABLE sample_terms (
            sample_id TEXT NOT NULL,
            term TEXT NOT NULL,
            term_norm TEXT NOT NULL,
            category TEXT NOT NULL,
            FOREIGN KEY(sample_id) REFERENCES samples(id)
        );

        CREATE TABLE terms (
            term TEXT NOT NULL,
            term_norm TEXT NOT NULL,
            category TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            PRIMARY KEY(term_norm, category)
        );
        """
    )


def insert_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX idx_samples_text_norm ON samples(text_norm);
        CREATE INDEX idx_samples_duration ON samples(duration_sec);
        CREATE INDEX idx_samples_language ON samples(language_key);
        CREATE INDEX idx_samples_trimmed ON samples(trimmed_trailing_artifact);
        CREATE INDEX idx_sample_terms_term ON sample_terms(term_norm, category);
        CREATE INDEX idx_sample_terms_sample ON sample_terms(sample_id);
        CREATE INDEX idx_terms_count ON terms(sample_count DESC);
        """
    )


def insert_row(
    connection: sqlite3.Connection,
    row: dict[str, Any],
    source_path: Path,
    term_counts: dict[tuple[str, str], Counter[str]],
) -> None:
    sample_id = str(row["id"])
    text = str(row["text"])
    used_terms = row.get("used_terms") or {}
    terms = flatten_terms(used_terms)
    terms_text_norm = normalize_text(" ".join(term for term, _category in terms))

    connection.execute(
        """
        INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sample_id,
            text,
            normalize_text(text),
            terms_text_norm,
            str(row["audio_path"]),
            float(row["duration_sec"]),
            int(row["sample_rate"]),
            row.get("language_key"),
            row.get("conditioning_audio_id"),
            row.get("conditioning_audio_file"),
            row.get("exaggeration"),
            row.get("cfg_weight"),
            int(bool(row["trimmed_trailing_artifact"])),
            json.dumps(row.get("tts_chunks") or [], ensure_ascii=False),
            json.dumps(used_terms, ensure_ascii=False),
            str(source_path.relative_to(source_path.parents[1])),
        ),
    )

    for term, category in terms:
        term_norm = normalize_text(term)
        connection.execute(
            "INSERT INTO sample_terms VALUES (?, ?, ?, ?)",
            (sample_id, term, term_norm, category),
        )
        term_counts[(term_norm, category)][term] += 1


def insert_term_counts(
    connection: sqlite3.Connection,
    term_counts: dict[tuple[str, str], Counter[str]],
) -> None:
    rows = []
    for (term_norm, category), counter in term_counts.items():
        display_term = counter.most_common(1)[0][0]
        rows.append((display_term, term_norm, category, sum(counter.values())))
    connection.executemany("INSERT INTO terms VALUES (?, ?, ?, ?)", rows)


def build_index(dataset_dir: Path, index_path: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(".sqlite.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    term_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with sqlite3.connect(temporary_path) as connection:
        create_schema(connection)
        paths = iter_metadata_paths(dataset_dir)
        total_rows = sum(1 for path in paths for line in path.open(encoding="utf-8") if line.strip())
        with connection:
            for source_path, row in tqdm(
                iter_metadata_rows(dataset_dir),
                total=total_rows,
                desc="Indexing dataset",
            ):
                insert_row(connection, row, source_path, term_counts)
            insert_term_counts(connection, term_counts)
            insert_indexes(connection)
        connection.execute("PRAGMA optimize")

    temporary_path.replace(index_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a SQLite index for the local dataset browser.")
    parser.add_argument("--dataset-dir", type=resolve_path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-path", type=resolve_path, default=DEFAULT_INDEX_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_index(args.dataset_dir, args.index_path)
    print(f"Wrote browser index: {args.index_path}")


if __name__ == "__main__":
    main()
