import os
import streamlit as st

from app import (
    ArctosEntityExtractor,
    QuerybotClassifier,
    ArctosURLBuilder,
    LLMExtractor,
)

st.set_page_config(page_title="Arctos Querybot", page_icon="🦎", layout="centered")

st.title("🦎 Arctos Querybot")
st.caption("Enter a natural-language query to search the Arctos collections database.")

CSV_PATH = "_portals (2).csv"

@st.cache_resource
def load_pipeline():
    extractor = ArctosEntityExtractor(CSV_PATH)
    classifier = QuerybotClassifier(extractor, threshold=0.90)
    url_builder = ArctosURLBuilder()
    return extractor, classifier, url_builder

entity_extractor, classifier, url_builder = load_pipeline()

@st.cache_resource
def load_llm():
    """Returns LLMExtractor or None if API key is missing."""
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
    st.subheader("Gemini API Key")
    api_key_input = st.text_input(
        "GEMINI_API_KEY", type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Required only for LLM-routed queries."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        if llm is None:
            # Re-attempt initialisation now that key is set
            try:
                llm = LLMExtractor()
                st.success("LLM ready ✓")
            except Exception as e:
                st.error(f"LLM init failed: {e}")
        else:
            st.success("LLM ready ✓")
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

    if route == "local":
        guids: set = set()
        guids.update(entities["guid_prefixes"])
        for inst in entities["institutions"]:
            guids.update(entity_extractor.institution_to_guids.get(inst, set()))
        for coll in entities["collections"]:
            guids.update(entity_extractor.collection_to_guids.get(coll, set()))
        if guids:
            fields["guid_prefix"] = ",".join(sorted(guids))
    else:
        if llm is None:
            error = "LLM route required but no API key is set. Add your Gemini key in the sidebar."
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

        # Cost (LLM route only)
        if route == "llm" and "usage" in dir():
            st.caption(
                f"🪙 {usage['input_tokens']} input tokens · "
                f"{usage['output_tokens']} output tokens · "
                f"**${usage['cost_usd']:.6f}**"
            )

        # Extracted fields
        if fields:
            st.subheader("Extracted fields")
            st.json(fields)

        # Entity details (always show)
        if any(entities.values()):
            with st.expander("Entity matches (from preprocessor)"):
                st.json(entities)