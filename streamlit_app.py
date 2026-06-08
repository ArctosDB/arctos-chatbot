import os
import json
import datetime
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from app import (
    ArctosEntityExtractor,
    QuerybotClassifier,
    ArctosURLBuilder,
    LLMExtractor,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_FALLBACK_MODEL,
)

st.set_page_config(page_title="Arctos Querybot", page_icon="🦎", layout="centered")

st.title("🦎 Arctos Querybot")
st.caption("Enter a natural-language query to search the Arctos collections database.")

CSV_PATH = "_portals.csv"

# ── Google Sheets logging ─────────────────────────────────────────────────────

SHEET_NAME      = "Arctos Querybot User Logs"
WORKSHEET_NAME  = "Sheet1"
SHEET_SCOPES    = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Column header order — must match the append_row call below
SHEET_HEADERS = [
    "timestamp",
    "query",
    "route",
    "coverage",
    "output_url",
    "extracted_fields",
    "feedback",       # 👍 / 👎 / blank
    "comment",        # free-text, optional
]

@st.cache_resource
def get_worksheet():
    """Authenticate and return the target worksheet (or None on failure)."""
    try:
        creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SHEET_SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)

        # Write header row if the sheet is empty
        if sheet.row_count == 0 or sheet.cell(1, 1).value != "timestamp":
            sheet.insert_row(SHEET_HEADERS, index=1)

        return sheet
    except Exception as e:
        # Logging is non-critical — don't crash the app if it fails
        st.warning(f"⚠ Google Sheets logging unavailable: {e}")
        return None

def log_search(sheet, query, route, coverage, url, fields):
    """Append one row for a search result (feedback columns left blank)."""
    if sheet is None:
        return
    try:
        sheet.append_row([
            datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            route,
            f"{coverage*100:.1f}%",
            url,
            json.dumps(fields),
            "",   # feedback — filled in later by update_feedback
            "",   # comment  — filled in later by update_feedback
        ])
    except Exception:
        pass  # Logging failures are silent so the user experience is unaffected

def update_feedback(sheet, row_index, feedback, comment):
    """Write feedback and comment into the last-appended row."""
    if sheet is None:
        return
    try:
        # feedback is column 7, comment is column 8
        sheet.update_cell(row_index, 7, feedback)
        sheet.update_cell(row_index, 8, comment)
    except Exception:
        pass

worksheet = get_worksheet()

# ── Pipeline setup ────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    extractor = ArctosEntityExtractor(CSV_PATH)
    classifier = QuerybotClassifier(extractor, threshold=0.90)
    url_builder = ArctosURLBuilder()
    return extractor, classifier, url_builder

entity_extractor, classifier, url_builder = load_pipeline()

@st.cache_resource
def load_llm():
    """
    Pushes secrets into env vars, then returns an LLMExtractor (or None).

    Reads from st.secrets (Streamlit Cloud secrets tab):
        GEMINI_API_KEY      — required for LLM calls
        LLM_PRIMARY_MODEL   — primary model name (optional, has a default)
        LLM_FALLBACK_MODEL  — fallback model name (optional, has a default)
    """
    # API key
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    # Model names — only override if explicitly set in secrets
    primary_model = st.secrets.get("LLM_PRIMARY_MODEL", "")
    if primary_model:
        os.environ["LLM_PRIMARY_MODEL"] = primary_model

    fallback_model = st.secrets.get("LLM_FALLBACK_MODEL", "")
    if fallback_model:
        os.environ["LLM_FALLBACK_MODEL"] = fallback_model

    try:
        return LLMExtractor()
    except EnvironmentError:
        return None

llm = load_llm()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Local-route coverage threshold",
        min_value=0.50, max_value=1.00, value=0.90, step=0.05,
        help="Queries where known-entity coverage exceeds this are handled locally (no LLM call)."
    )
    classifier.threshold = threshold

    st.divider()
    if llm is not None:
        st.success("LLM ready ✓")
        with st.expander("Model info"):
            st.caption(f"**Primary:** `{llm.primary_model}`")
            st.caption(f"**Fallback:** `{llm.fallback_model}`")
    else:
        st.warning("No API key — LLM route unavailable.")

# ── Main input ────────────────────────────────────────────────────────────────
query = st.text_input(
    "Query",
    placeholder="e.g. What specimens of Taricha rivularis are in MVZ:Herp?",
)

run = st.button("Search", type="primary", disabled=not query)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run and query:
    classification = classifier.classify(query)
    route = classification["route"]
    coverage = classification["coverage"]
    entities = classification["entities"]

    # Route badge
    if route == "local":
        st.success(f"**Route: LOCAL** — {coverage*100:.0f}% entity coverage")
    else:
        st.info(f"**Route: LLM** — {coverage*100:.0f}% entity coverage")

    # Build fields dict
    fields: dict = {}
    error: str | None = None
    usage: dict = {}

    if route == "local":
        guids: set = set()
        guids.update(entities["guid_prefixes"])

        # Only expand institutions/collections if no exact guid_prefix was matched.
        # This prevents "MVZ:Herp" from ballooning into every MVZ collection.
        if not guids:
            for inst in entities["institutions"]:
                guids.update(entity_extractor.institution_to_guids.get(inst, set()))
            for coll in entities["collections"]:
                guids.update(entity_extractor.collection_to_guids.get(coll, set()))

        if guids:
            fields["guid_prefix"] = ",".join(sorted(guids))
    else:
        if llm is None:
            error = "LLM route required but no API key is configured. Set GEMINI_API_KEY in Streamlit secrets."
        else:
            with st.spinner("Calling Gemini…"):
                try:
                    fields, usage = llm.extract(query)
                    # Expand bare institution abbreviations in guid_prefix
                    if "guid_prefix" in fields:
                        fields["guid_prefix"] = entity_extractor.expand_guid_prefix(fields["guid_prefix"])
                except Exception as e:
                    error = str(e)

    if error:
        st.error(error)
    else:
        url = url_builder.build(fields)

        st.divider()

        # URL output
        st.subheader("Generated URL")
        st.code(url, language=None)
        st.link_button("Open in Arctos ↗", url)

        # Cost & model info (LLM route only)
        if route == "llm" and usage:
            model_label = usage.get("model_used", "unknown")
            fallback_note = " ⚠ fallback" if usage.get("fallback_used") else ""
            st.caption(
                f"🪙 {usage['input_tokens']} input tokens · "
                f"{usage['output_tokens']} output tokens · "
                f"**${usage['cost_usd']:.6f}** · "
                f"model: `{model_label}`{fallback_note}"
            )
            if usage.get("fallback_used"):
                st.warning(
                    f"Primary model (`{llm.primary_model}`) failed — "
                    f"fell back to `{llm.fallback_model}`.\n\n"
                    f"Primary error: {usage['primary_error']}"
                )

        # Extracted fields
        if fields:
            st.subheader("Extracted fields")
            st.json(fields)

        # Entity details (always show)
        if any(entities.values()):
            with st.expander("Entity matches (from preprocessor)"):
                st.json(entities)

        # ── Auto-log the search result ────────────────────────────────────────
        # Append a row immediately; feedback columns start blank
        log_search(worksheet, query, route, coverage, url, fields)
        # Store the row index so feedback can update the same row later.
        # get_all_values() includes the header, so last data row = len - 1 + 1 = len
        if worksheet is not None:
            try:
                logged_row = len(worksheet.get_all_values())
            except Exception:
                logged_row = None
        else:
            logged_row = None

        # ── Feedback section ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Was this result helpful?")

        col1, col2 = st.columns(2)
        thumbs_up   = col1.button("👍 Yes")
        thumbs_down = col2.button("👎 No")

        comment = st.text_input(
            "Optional comment",
            placeholder="e.g. Wrong collection returned, missing specimens…",
            key="feedback_comment",
        )

        if (thumbs_up or thumbs_down) and logged_row:
            feedback = "👍" if thumbs_up else "👎"
            update_feedback(worksheet, logged_row, feedback, comment)
            st.success("Thanks for the feedback!")
