from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
from nltk.stem.snowball import SnowballStemmer

from build_dataset_browser_index import build_index


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = BASE_DIR / "artifacts_local/processed/final_dataset"
DEFAULT_INDEX_PATH = DEFAULT_DATASET_DIR / "browser_index.sqlite"
TERM_CATEGORIES = ("common", "selected", "excluded")
CATEGORY_LABELS = {
    "common": "Vanlig begrep",
    "selected": "Utvalgt begrep",
    "excluded": "Ekskludert begrep",
}

TERM_ALLOWED_PATTERN = re.compile(r"^[0-9A-Za-zÆØÅæøå-]+$")
HYPHENATED_NUMBER_PATTERN = re.compile(r"^\d+(?:-\d+)+$")

APP_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}
.app-header h1 {
    margin-bottom: 0.25rem;
    font-size: 1.55rem;
}
.app-header p {
    margin: 0.2rem 0;
    color: #4b5563;
    font-size: 0.92rem;
    line-height: 1.35;
}
#audio-preview {
    min-height: 92px !important;
}
#sample-info {
    margin: -0.35rem 0 -0.1rem 0;
}
#sample-info .sample-info-line {
    align-items: center;
    color: #374151;
    display: flex;
    font-size: 0.92rem;
    gap: 0.85rem;
    justify-content: space-between;
}
#sample-info .sample-metadata {
    min-width: 0;
}
#sample-info .legend {
    align-items: center;
    display: flex;
    flex-shrink: 0;
    gap: 0.55rem;
}
.legend-item {
    align-items: center;
    display: inline-flex;
    gap: 0.25rem;
}
.legend-swatch {
    border: 1px solid #9ca3af;
    border-radius: 3px;
    height: 0.8rem;
    width: 0.8rem;
}
#sample-text {
    max-height: 225px;
    overflow-y: auto;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.75rem 0.9rem;
    background: #ffffff;
}
.target-text {
    color: #111827;
    font-size: 1.08rem;
    line-height: 1.62;
}
.term-highlight {
    border-radius: 4px;
    color: #111827 !important;
    padding: 0.03rem 0.16rem;
    text-decoration: none;
}
.term-highlight:hover {
    outline: 1px solid #4b5563;
}
.term-common {
    background: #bbf7d0;
}
.term-selected {
    background: #fef08a;
}
.term-excluded {
    background: #fecaca;
}
.term-unlinked {
    cursor: default;
}
#search-row {
    margin-top: 0.45rem;
}
#results-table textarea {
    font-size: 0.9rem !important;
}
#results-table table th:nth-child(3),
#results-table table td:nth-child(3) {
    width: 76px !important;
    max-width: 76px !important;
}
#results-table table th:nth-child(5),
#results-table table td:nth-child(5) {
    min-width: 360px !important;
}
#clicked-term,
#clicked-term-button {
    height: 1px !important;
    left: -10000px !important;
    opacity: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    top: auto !important;
    width: 1px !important;
}
"""


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


@lru_cache(maxsize=1)
def norwegian_stemmer() -> SnowballStemmer:
    return SnowballStemmer("norwegian")


def normalize_term(term: str) -> str:
    """Match prompt_creation.processing._normalize_term for browser matching."""
    normalized_parts = []
    for part in term.split("-"):
        if not part:
            normalized_parts.append(part)
            continue
        lower = part.lower()
        if lower.isnumeric():
            normalized_parts.append(lower)
            continue
        normalized_parts.append(norwegian_stemmer().stem(lower))
    return "-".join(normalized_parts)


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def connect(index_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    return connection


def term_id_from_item_id(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]


def iter_resource_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=4)
def load_term_lookup(dataset_dir_string: str) -> dict[str, dict[str, str]]:
    dataset_dir = Path(dataset_dir_string)
    lookup: dict[str, dict[str, str]] = {}
    resource_paths = [
        dataset_dir / "resources" / "snomed_10000.jsonl",
        dataset_dir / "resources" / "snomed_terms.jsonl",
    ]
    for path in resource_paths:
        for row in iter_resource_rows(path):
            term_id = row.get("term_id")
            if not term_id:
                continue
            names = [row.get("term"), *row.get("variants", [])]
            for name in names:
                if name:
                    lookup.setdefault(
                        normalize_text(str(name)),
                        {
                            "term": str(row["term"]),
                            "term_id": str(term_id),
                        },
                    )
    return lookup


@lru_cache(maxsize=4)
def load_variants_by_term_id(dataset_dir_string: str) -> dict[str, list[str]]:
    dataset_dir = Path(dataset_dir_string)
    variants_by_id: dict[str, list[str]] = {}
    resource_paths = [
        dataset_dir / "resources" / "snomed_10000.jsonl",
        dataset_dir / "resources" / "snomed_terms.jsonl",
    ]
    for path in resource_paths:
        for row in iter_resource_rows(path):
            term_id = row.get("term_id")
            if not term_id:
                continue
            names = [row.get("term"), *row.get("variants", [])]
            for name in names:
                if name and str(name) not in variants_by_id.setdefault(str(term_id), []):
                    variants_by_id[str(term_id)].append(str(name))
    return variants_by_id


@lru_cache(maxsize=4)
def load_term_by_id(dataset_dir_string: str) -> dict[str, str]:
    dataset_dir = Path(dataset_dir_string)
    terms_by_id: dict[str, str] = {}
    resource_paths = [
        dataset_dir / "resources" / "snomed_10000.jsonl",
        dataset_dir / "resources" / "snomed_terms.jsonl",
    ]
    for path in resource_paths:
        for row in iter_resource_rows(path):
            term_id = row.get("term_id")
            term = row.get("term")
            if term_id and term:
                terms_by_id.setdefault(str(term_id), str(term))
    return terms_by_id


def used_terms_by_category(used_terms_json: str) -> dict[str, list[str]]:
    used_terms = json.loads(used_terms_json)
    return {
        category: [str(term) for term in used_terms.get(category, [])]
        for category in TERM_CATEGORIES
    }


def compact_used_terms(used_terms_json: str) -> str:
    used_terms = used_terms_by_category(used_terms_json)
    terms = [
        term
        for category in TERM_CATEGORIES
        for term in used_terms[category]
    ]
    return ", ".join(terms)


def short_text(text: str, max_chars: int = 180) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def result_row(row: sqlite3.Row) -> list[Any]:
    return [
        row["id"],
        term_id_from_item_id(row["id"]),
        row["language_key"] or "",
        compact_used_terms(row["used_terms_json"]),
        short_text(row["text"]),
    ]


def search_samples(
    index_path: str,
    query: str,
    limit: int | str | None,
) -> tuple[list[list[Any]], list[str], str]:
    query_norm = normalize_text(query)
    params: list[Any] = []
    where_sql = "1 = 1"
    if query_norm:
        query_like = f"%{escape_like(query_norm)}%"
        term_id_like = f"{escape_like(query_norm)}\\_%"
        where_sql += (
            " AND (s.text_norm LIKE ? ESCAPE '\\' "
            "OR s.id = ? "
            "OR s.id LIKE ? ESCAPE '\\' "
            "OR s.id LIKE ? ESCAPE '\\')"
        )
        params.extend([query_like, query_norm, query_like, term_id_like])

    row_limit = int(limit or 100)
    order_sql = "RANDOM()" if not query_norm else "s.id ASC"
    sql = f"""
        SELECT s.id, s.text, s.used_terms_json, s.language_key
        FROM samples s
        WHERE {where_sql}
        ORDER BY {order_sql}
        LIMIT ?
    """
    with connect(Path(index_path)) as connection:
        total = connection.execute(
            f"SELECT COUNT(*) AS total FROM samples s WHERE {where_sql}",
            params,
        ).fetchone()["total"]
        rows = connection.execute(sql, [*params, row_limit]).fetchall()

    results = [result_row(row) for row in rows]
    item_ids = [row["id"] for row in rows]
    if query_norm:
        return results, item_ids, f"Showing {len(results)} out of {total}"
    return results, item_ids, f"Showing {len(results)} random samples out of {total}"


def search_samples_by_clicked_term(
    index_path: str,
    clicked_value: str,
    limit: int | str | None,
) -> tuple[list[list[Any]], list[str], str, str]:
    query = clicked_value.removeprefix("text:")
    if not query:
        return [], [], "", ""
    results, item_ids, status = search_samples(index_path, query, limit)
    return results, item_ids, status, query


def search_samples_by_term_id(
    index_path: str,
    term_id: str,
    term_label: str,
    limit: int | str | None,
) -> tuple[list[list[Any]], list[str], str, str]:
    if not term_id:
        return [], [], "", ""

    sql = """
        SELECT s.id, s.text, s.used_terms_json, s.language_key
        FROM samples s
        WHERE s.id LIKE ?
        ORDER BY s.id ASC
        LIMIT ?
    """
    with connect(Path(index_path)) as connection:
        total = connection.execute(
            "SELECT COUNT(*) AS total FROM samples s WHERE s.id LIKE ?",
            (f"{term_id}_%",),
        ).fetchone()["total"]
        rows = connection.execute(sql, (f"{term_id}_%", int(limit or 100))).fetchall()

    results = [result_row(row) for row in rows]
    item_ids = [row["id"] for row in rows]
    label = term_label or term_id
    return results, item_ids, f"Showing {len(results)} out of {total} for term id `{term_id}` ({label})", label


def selected_row_index(event: gr.SelectData) -> int | None:
    if event.index is None:
        return None
    if isinstance(event.index, (list, tuple)):
        if not event.index:
            return None
        row_index = event.index[0]
        if isinstance(row_index, (list, tuple)):
            row_index = row_index[0]
        return int(row_index)
    return int(event.index)


def selected_item_id(item_ids: list[str], event: gr.SelectData) -> str:
    row_index = selected_row_index(event)
    if row_index is None or row_index >= len(item_ids):
        return ""
    return item_ids[row_index]


def is_complex_chemical(term: str) -> bool:
    if re.fullmatch(r"\d+[\-,]\d+", term):
        return False

    prefix_pattern = r"^(cis|trans|ortho|para|meta|[NOLSD])\-[a-z0-9]"
    chem_suffixes = r"(ase|yl|id|at|in|an|ol|en)\b"
    structure_pattern = r"(\d+,\d+|alfa|beta|gamma|delta|epsilon|kappa)"
    bond_pattern = r"[A-Za-zÆØÅæøå]-\d+(,\d+)*-[A-Za-zÆØÅæøå]"

    if re.search(prefix_pattern, term):
        return True
    if re.search(bond_pattern, term, re.IGNORECASE):
        return True
    if re.search(chem_suffixes, term, re.IGNORECASE) and re.search(
        structure_pattern,
        term,
        re.IGNORECASE,
    ):
        return True
    return len(term) > 12 and term.count("-") >= 2 and re.search(r"\d", term) is not None


def iter_token_spans(text: str) -> list[tuple[str, int, int]]:
    """Tokenize like prompt_creation.processing.tokenize_terms, preserving spans."""
    spans: list[tuple[str, int, int]] = []
    for space_match in re.finditer(r"\S+", text):
        token = space_match.group(0)
        base_start = space_match.start()
        token_pattern = r"[0-9A-Za-zÆØÅæøå]+" if is_complex_chemical(token) else r"[0-9A-Za-zÆØÅæøå-]+"
        for part_match in re.finditer(token_pattern, token):
            stripped = part_match.group(0).strip("-")
            if not stripped:
                continue
            leading_trim = len(part_match.group(0)) - len(part_match.group(0).lstrip("-"))
            start = base_start + part_match.start() + leading_trim
            end = start + len(stripped)
            if stripped.isdigit() or HYPHENATED_NUMBER_PATTERN.match(stripped):
                continue
            if TERM_ALLOWED_PATTERN.match(stripped):
                spans.append((stripped, start, end))
    return spans


def iter_match_candidates(token: str, start: int) -> list[tuple[str, int, int]]:
    candidates = [(token, start, start + len(token))]
    if "-" not in token:
        return candidates

    offset = 0
    parts_with_offsets = []
    for part in token.split("-"):
        part_start = token.find(part, offset)
        part_end = part_start + len(part)
        offset = part_end + 1
        if part:
            parts_with_offsets.append((part, part_start, part_end))

    seen = {token}
    for start_index in range(len(parts_with_offsets)):
        for end_index in range(start_index + 1, len(parts_with_offsets) + 1):
            if start_index == 0 and end_index == len(parts_with_offsets):
                continue
            candidate = "-".join(part for part, _part_start, _part_end in parts_with_offsets[start_index:end_index])
            if len(candidate) < 3 or candidate in seen or not any(char.isalpha() for char in candidate):
                continue
            seen.add(candidate)
            candidate_start = start + parts_with_offsets[start_index][1]
            candidate_end = start + parts_with_offsets[end_index - 1][2]
            candidates.append((candidate, candidate_start, candidate_end))
    return candidates


def term_patterns(used_terms: dict[str, list[str]], dataset_dir: str) -> dict[str, tuple[str, str | None, str]]:
    term_lookup = load_term_lookup(dataset_dir)
    variants_by_term_id = load_variants_by_term_id(dataset_dir)
    normalized_to_term: dict[str, tuple[str, str | None, str]] = {}
    for category in ("excluded", "selected", "common"):
        for term in sorted(used_terms[category], key=len, reverse=True):
            names = [term]
            lookup_item = term_lookup.get(normalize_text(term))
            term_id = lookup_item["term_id"] if lookup_item else None
            if term_id:
                names.extend(variants_by_term_id.get(term_id, []))
            for name in names:
                if name:
                    normalized_to_term.setdefault(normalize_term(name), (category, term_id, term))
    return normalized_to_term


def highlighted_text_html(text: str, used_terms_json: str, dataset_dir: str) -> str:
    used_terms = used_terms_by_category(used_terms_json)
    normalized_to_term = term_patterns(used_terms, dataset_dir)
    if not normalized_to_term:
        return f"<div class=\"target-text\">{html.escape(text)}</div>"

    matches: list[tuple[int, int, str, str, str | None]] = []
    occupied_spans: list[tuple[int, int]] = []
    for token, start, _end in iter_token_spans(text):
        for candidate, candidate_start, candidate_end in iter_match_candidates(token, start):
            term_info = normalized_to_term.get(normalize_term(candidate))
            if term_info is None:
                continue
            if any(candidate_start < existing_end and candidate_end > existing_start for existing_start, existing_end in occupied_spans):
                continue
            category, term_id, canonical_term = term_info
            matches.append((candidate_start, candidate_end, category, canonical_term, term_id))
            occupied_spans.append((candidate_start, candidate_end))
            break

    if not matches:
        return f"<div class=\"target-text\">{html.escape(text)}</div>"

    parts: list[str] = []
    previous_end = 0

    for start, end, category, canonical_term, term_id in sorted(matches):
        matched_text = text[start:end]
        parts.append(html.escape(text[previous_end:start]))
        escaped_text = html.escape(matched_text)
        css_class = f"term-highlight term-{category}"
        title = CATEGORY_LABELS[category]
        click_value = html.escape(f"text:{canonical_term}", quote=True)
        term_id_suffix = f": {html.escape(term_id)}" if term_id else ""
        parts.append(
            f"<a class=\"{css_class}\" href=\"#\" data-term=\"{click_value}\" "
            f"title=\"{title}{term_id_suffix}\">{escaped_text}</a>"
        )
        previous_end = end

    parts.append(html.escape(text[previous_end:]))
    return f"<div class=\"target-text\">{''.join(parts)}</div>"


def preview_sample(
    index_path: str,
    dataset_dir: str,
    item_id: str,
) -> tuple[str | None, str, str]:
    if not item_id:
        return None, "", ""

    with connect(Path(index_path)) as connection:
        row = connection.execute("SELECT * FROM samples WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return None, "", f"<p>Sample not found: <code>{html.escape(item_id)}</code></p>"

    audio_path = Path(dataset_dir) / row["audio_path"]
    return (
        str(audio_path),
        sample_info_html(row, dataset_dir),
        highlighted_text_html(row["text"], row["used_terms_json"], dataset_dir),
    )


def sample_info_html(row: sqlite3.Row, dataset_dir: str) -> str:
    item_id = html.escape(row["id"])
    raw_term_id = term_id_from_item_id(row["id"])
    term_id = html.escape(raw_term_id)
    term = html.escape(load_term_by_id(dataset_dir).get(raw_term_id, "-"))
    language_key = html.escape(str(row["language_key"] or ""))
    return (
        "<div class=\"sample-info-line\">"
        "<div class=\"sample-metadata\">"
        f"<strong>item_id:</strong> <code>{item_id}</code>"
        f" &nbsp; <strong>term_id:</strong> <code>{term_id}</code>"
        f" &nbsp; <strong>term:</strong> <code>{term}</code>"
        f" &nbsp; <strong>language_key:</strong> <code>{language_key}</code>"
        "</div>"
        "<div class=\"legend\">"
        "<span class=\"legend-item\"><span class=\"legend-swatch term-common\"></span>common</span>"
        "<span class=\"legend-item\"><span class=\"legend-swatch term-selected\"></span>selected</span>"
        "<span class=\"legend-item\"><span class=\"legend-swatch term-excluded\"></span>excluded</span>"
        "</div>"
        "</div>"
    )


def select_result(
    index_path: str,
    dataset_dir: str,
    item_ids: list[str],
    event: gr.SelectData,
) -> tuple[str | None, str, str]:
    return preview_sample(index_path, dataset_dir, selected_item_id(item_ids, event))


def ensure_index(dataset_dir: Path, index_path: Path, rebuild_index: bool) -> None:
    if rebuild_index or not index_path.exists():
        build_index(dataset_dir, index_path)


def initial_results(
    index_path: str,
    limit: int | str | None,
    request: gr.Request,
) -> tuple[list[list[Any]], list[str], str, str]:
    query_params = request.query_params if request else {}
    term_id = str(query_params.get("term_id", "")).strip()
    term_label = str(query_params.get("term", "")).strip()
    if term_id:
        results, item_ids, status, query = search_samples_by_term_id(
            index_path,
            term_id,
            term_label,
            limit,
        )
        return results, item_ids, status, query

    results, item_ids, status = search_samples(index_path, "", limit)
    return results, item_ids, status, ""


def header_html() -> str:
    return """
    <div class="app-header">
      <h1>NORMED-ASR – Norsk syntetisk medisinsk datasett for automatisk talegjenkjenning</h1>
      <p>
        Dette er en forhåndsvisning av et stort norsk syntetisk medisinsk datasett for
        talegjenkjenningsapplikasjoner, utviklet av
        <a href="https://www.spki.no/" target="_blank">Senter for pasientnær kunstig intelligens</a>
        ved <a href="https://www.unn.no/" target="_blank">Universitetssykehuset i Nord-Norge (UNN)</a>
        og <a href="https://ai.nb.no/" target="_blank">Nasjonalbiblioteket sin KI-lab (NB AI-lab)</a>.
        Arbeidet inngår i Helse-Nord-prosjektet
        <a href="https://www.helse-nord.no/helsefaglig/forskning-og-innovasjon/innovasjon/innovasjonsprosjekter/talegjenkjenning-og-redusert-dokumentasjonsbyrde-med-bruk-av-kunstig-intelligens/" target="_blank">
        Talegjenkjenning og redusert dokumentasjonsbyrde med bruk av kunstig intelligens (KI-DOK)</a>.
      </p>
      <p>
        Det fulle datasettet består av rundt 1600 timer lyd fordelt på om lag 200 000 lydklipp
        og omfatter 22 000 ulike medisinske norske begreper. Omtrent 10 000 utvalgte begreper
        er balansert for hyppig forekomst, med mål om minst 100 forekomster totalt.
      </p>
      <p><strong>Søk på fagtermer og forhåndsvis tilknyttede lydklipp.</strong></p>
    </div>
    """


def build_app(dataset_dir: Path, index_path: Path) -> gr.Blocks:
    with gr.Blocks(
        title="NORMED-ASR dataset browser",
        css=APP_CSS,
        js="""
        () => {
            window.pauseNormedPreviewAudio = () => {
                document.querySelectorAll("audio").forEach((audio) => {
                    audio.pause();
                    audio.currentTime = 0;
                    audio.load();
                });
            };
            document.addEventListener("click", (event) => {
                if (event.target.closest("#results-table")) {
                    window.pauseNormedPreviewAudio();
                }
                const target = event.target.closest("a[data-term]");
                if (!target) {
                    return;
                }
                event.preventDefault();
                const input = document.querySelector("#clicked-term textarea, #clicked-term input");
                const button = document.querySelector("#clicked-term-button button, #clicked-term-button");
                if (!input || !button) {
                    return;
                }
                input.value = target.dataset.term;
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
                window.setTimeout(() => button.click(), 0);
            });
        }
        """,
    ) as app:
        index_path_state = gr.State(str(index_path))
        dataset_dir_state = gr.State(str(dataset_dir))
        item_ids_state = gr.State([])
        clicked_term = gr.Textbox(elem_id="clicked-term", container=False)
        clicked_term_button = gr.Button("Clicked term", elem_id="clicked-term-button")

        gr.HTML(header_html(), padding=False)
        audio = gr.Audio(label="Audio", type="filepath", elem_id="audio-preview")
        sample_info = gr.HTML(elem_id="sample-info", padding=False)
        text = gr.HTML(elem_id="sample-text", padding=False)

        with gr.Row(elem_id="search-row"):
            query = gr.Textbox(
                label="Search text",
                placeholder="Search transcript text",
                scale=5,
                elem_id="query-box",
            )
            limit = gr.Dropdown(
                choices=[25, 50, 100, 200, 500],
                value=100,
                label="Rows",
                scale=1,
            )
            search_button = gr.Button("Search", variant="primary", scale=1, elem_id="search-button")

        status = gr.Markdown()
        results = gr.Dataframe(
            headers=["item_id", "term_id", "language_key", "used_terms", "text"],
            datatype=["str", "str", "str", "str", "str"],
            label="Results",
            interactive=False,
            wrap=True,
            elem_id="results-table",
        )

        search_inputs = [index_path_state, query, limit]
        search_outputs = [results, item_ids_state, status]
        search_button.click(search_samples, inputs=search_inputs, outputs=search_outputs)
        query.submit(search_samples, inputs=search_inputs, outputs=search_outputs)
        clicked_term_button.click(
            search_samples_by_clicked_term,
            inputs=[index_path_state, clicked_term, limit],
            outputs=[results, item_ids_state, status, query],
        )
        results.select(
            select_result,
            inputs=[index_path_state, dataset_dir_state, item_ids_state],
            outputs=[audio, sample_info, text],
            js="() => { window.pauseNormedPreviewAudio && window.pauseNormedPreviewAudio(); }",
        )
        app.load(
            initial_results,
            inputs=[index_path_state, limit],
            outputs=[results, item_ids_state, status, query],
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a local Gradio browser for the final dataset.")
    parser.add_argument("--dataset-dir", type=resolve_path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-path", type=resolve_path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Whether to create a public Gradio share link.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_index(args.dataset_dir, args.index_path, args.rebuild_index)
    app = build_app(args.dataset_dir, args.index_path)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
