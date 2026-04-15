import csv
import re
from collections import defaultdict
from flashtext import KeywordProcessor



FILLER_WORDS = {
    # articles / determiners
    "a", "an", "the",
    # some basic prepositions
    "in", "at", "of", "from", "for", "by", "to", "with", "on", "into",
    # conjunctions
    "and", "or", "but",
    # common query verbs
    "show", "find", "get", "give", "list", "search", "look", "display",
    "fetch", "retrieve",
    # common query words
    "me", "all", "any", "some", "what", "which", "how", "many",
    "are", "is", "was", "were", "be", "been",
    # vague scope words
    "only", "just", "please", "can", "you",
    # pronouns
    "i", "we",
    # Implicity filler
    "specimen", "specimens", "records", "collections", "data",
}



class ArctosEntityExtractor:
    def __init__(self, csv_filepath: str):
        self.institutions: set = set()
        self.collections: set = set()
        self.guid_prefixes: set = set()

        self.abbreviation_to_institution: dict = {}
        self.collection_abbrev_to_full: defaultdict = defaultdict(set)
        self.guid_to_collection: dict = {}

        # Reverse lookups: institution/collection name -> set of guid_prefixes
        self.institution_to_guids: defaultdict = defaultdict(set)
        self.collection_to_guids: defaultdict = defaultdict(set)

        self._load_csv(csv_filepath)
        self._initialize_processors()

    def _load_csv(self, csv_filepath: str) -> None:
        with open(csv_filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                institution = row["INSTITUTION"].strip()
                collection = row["COLLECTION"].strip()
                guid_prefix = row["GUID_PREFIX"].strip()

                if institution:
                    self.institutions.add(institution)
                if collection:
                    self.collections.add(collection)
                if guid_prefix:
                    self.guid_prefixes.add(guid_prefix)

                    if collection:
                        self.guid_to_collection[guid_prefix] = collection
                        self.collection_to_guids[collection].add(guid_prefix)

                    if institution:
                        self.institution_to_guids[institution].add(guid_prefix)

                    if ":" in guid_prefix:
                        parts = guid_prefix.split(":")
                        inst_abbrev = parts[0]
                        if inst_abbrev and institution:
                            self.abbreviation_to_institution[inst_abbrev] = institution
                        if len(parts) > 1:
                            coll_abbrev = parts[1]
                            if coll_abbrev and collection:
                                self.collection_abbrev_to_full[coll_abbrev].add(collection)

    def expand_guid_prefix(self, guid_prefix_value: str) -> str:
        """
        If guid_prefix_value contains bare institution abbreviations (no ':'),
        expand each one to all known guid_prefixes for that institution.
        Full prefixes like 'MVZ:Herp' are kept as-is.
        Returns a comma-separated string of guid_prefixes.
        """
        parts = [p.strip() for p in guid_prefix_value.split(",") if p.strip()]
        expanded: set = set()
        for part in parts:
            if ":" in part:
                # Already a full guid_prefix — keep as-is
                expanded.add(part)
            else:
                # Bare abbreviation — expand to all guid_prefixes for that institution
                institution = self.abbreviation_to_institution.get(part)
                if institution:
                    guids = self.institution_to_guids.get(institution, set())
                    if guids:
                        expanded.update(guids)
                    else:
                        expanded.add(part)  # fallback: keep as-is
                else:
                    expanded.add(part)  # unknown abbreviation — keep as-is
        return ",".join(sorted(expanded))

    def _initialize_processors(self) -> None:
        self.institution_processor = KeywordProcessor(case_sensitive=False)
        self.collection_processor = KeywordProcessor(case_sensitive=False)
        self.guid_processor = KeywordProcessor(case_sensitive=True)

        for institution in self.institutions:
            self.institution_processor.add_keyword(institution)

        for abbrev, institution in self.abbreviation_to_institution.items():
            self.institution_processor.add_keyword(abbrev, institution)

        for collection in self.collections:
            self.collection_processor.add_keyword(collection)

        for guid_prefix in self.guid_prefixes:
            self.guid_processor.add_keyword(guid_prefix)

    def matched_chars(self, text: str) -> int:
        spans = []
        for processor in (
            self.institution_processor,
            self.collection_processor,
            self.guid_processor,
        ):
            for _, start, end in processor.extract_keywords(text, span_info=True):
                spans.append((start, end))

        if not spans:
            return 0

        spans.sort()
        merged = [list(spans[0])]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return sum(
            len(re.findall(r"[a-zA-Z0-9]", text[s:e])) for s, e in merged
        )

    def extract(self, text: str) -> dict:
        institutions = self.institution_processor.extract_keywords(text)
        collections = self.collection_processor.extract_keywords(text)
        guid_prefixes = self.guid_processor.extract_keywords(text)

        for guid in guid_prefixes:
            if guid in self.guid_to_collection:
                collection_name = self.guid_to_collection[guid]
                if collection_name not in collections:
                    collections.append(collection_name)

        return {
            "institutions": institutions,
            "collections": collections,
            "guid_prefixes": guid_prefixes,
        }




class QuerybotClassifier:
    def __init__(self, extractor: ArctosEntityExtractor, threshold: float = 0.90):
        self.extractor = extractor
        self.threshold = threshold


    @staticmethod
    def _alphanum_len(text: str) -> int:
        return len(re.findall(r"[a-zA-Z0-9]", text))

    @staticmethod
    def _find_filler_in_query(query: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        return [t for t in tokens if t in FILLER_WORDS]


    def classify(self, query: str) -> dict:
        entities = self.extractor.extract(query)

        total_chars = self._alphanum_len(query)
        filler_found = self._find_filler_in_query(query)
        filler_chars = sum(len(w) for w in filler_found)
        meaningful_chars = total_chars - filler_chars
        entity_chars = self.extractor.matched_chars(query)

        if meaningful_chars <= 0:
            # Edge case where query is entirely filler words
            return {
                "route": "local",
                "coverage": 1.0,
                "total_chars": total_chars,
                "filler_chars": filler_chars,
                "entity_chars": entity_chars,
                "filler_found": filler_found,
                "entities": entities,
            }

        coverage = min(entity_chars / meaningful_chars, 1.0)
        route = "local" if coverage >= self.threshold else "llm"

        return {
            "route": route,
            "coverage": round(coverage, 4),
            "total_chars": total_chars,
            "filler_chars": filler_chars,
            "entity_chars": entity_chars,
            "filler_found": filler_found,
            "entities": entities,
        }



# =============================================================================
# ARCTOS URL BUILDER
# =============================================================================
# Mapping from our internal/flat field names to the confirmed Arctos URL params
# (sourced directly from the Catalog_Record_Search.html form field names).
#
# Fields marked DIRECT need no renaming. Others have an explicit mapping.
# Fields marked PLACE are handled specially via place_term + place_term_type.
# =============================================================================

FLAT_TO_URL_PARAM: dict[str, str] = {
    # --- Identifiers ---
    "guid_prefix":              "guid_prefix",           # DIRECT
    "guid":                     "guid",                  # DIRECT
    "cat_num":                  "cat_num",               # DIRECT

    # --- Taxonomy (granular) ---
    "kingdom":                  "kingdom",               # DIRECT
    "phylum":                   "phylum",                # DIRECT
    "phylclass":                "phylclass",             # DIRECT
    "phylorder":                "phylorder",             # DIRECT
    "suborder":                 "suborder",              # DIRECT
    "superfamily":              "superfamily",           # DIRECT
    "family":                   "family",                # DIRECT
    "subfamily":                "subfamily",             # DIRECT
    "tribe":                    "tribe",                 # DIRECT
    "subtribe":                 "subtribe",              # DIRECT
    "genus":                    "genus",                 # DIRECT
    "species":                  "species",               # DIRECT
    "subspecies":               "subspecies",            # DIRECT

    # --- Taxonomy (catch-all) ---
    "taxonomy_search":          "taxonomy_search",       # DIRECT
    "scientific_name":          "scientific_name",       # DIRECT

    # --- Agents ---
    "collectors":               "collector",             # renamed
    "identifiedby":             "identified_agent",      # renamed

    # --- Dates ---
    "began_date":               "began_date",            # DIRECT (ISO format)
    "ended_date":               "ended_date",            # DIRECT (ISO format)
    "verbatim_date":            "verbatim_date",         # DIRECT
    "month":                    "month",                 # DIRECT
    "day":                      "day",                   # DIRECT

    # --- Geography (asserted fields — direct params) ---
    "country":                  "country",               # DIRECT
    "state_prov":               "state_prov",            # DIRECT
    "county":                   "county",                # DIRECT

    # --- Locality ---
    "spec_locality":            "spec_locality",         # DIRECT
    "locality_name":            "locality_name",         # DIRECT
    "verbatim_locality":        "verbatim_locality",     # DIRECT
    "habitat":                  "habitat",               # DIRECT

    # --- Elevation ---
    "min_elev_in_m":            "minimum_elevation",     # renamed
    "max_elev_in_m":            "maximum_elevation",     # renamed

    # --- Record metadata ---
    "typestatus":               "type_status",           # renamed
    "verificationstatus":       "verificationstatus",    # DIRECT
    "collecting_method":        "collecting_method",     # DIRECT
    "collecting_source":        "collecting_source",     # DIRECT
    "has_tissues":              "is_tissue",             # renamed
    "identification_remarks":   "identification_remarks",# DIRECT
}

# place_term_type values accepted by Arctos (from the dropdown in the HTML).
# When the LLM extracts a place field that maps to place_term, we also need
# to set place_term_type to one of these values.
PLACE_TERM_TYPES = {
    "continent", "country", "county", "drainage", "feature",
    "geography search term", "island", "island_group", "locality_name",
    "ocean", "previous geography", "quad", "sea", "spatial place name",
    "specific locality", "state_prov",
}

BASE_URL = "https://arctos.database.museum/search.cfm"


class ArctosURLBuilder:
    """
    Converts a flat field dict (from LLMExtractor or the local classifier)
    into a valid Arctos search URL.

    Expected input dict keys are flat column names (see FLAT_TO_URL_PARAM).
    Special keys handled separately:
      - "place_term"      : the place value (e.g. "Iowa")
      - "place_term_type" : the place type (e.g. "state_prov", "county")
    """

    def build(self, fields: dict) -> str:
        from urllib.parse import urlencode, quote

        params: dict[str, str] = {}

        for flat_key, value in fields.items():
            if not value:
                continue
            value = str(value).strip()
            if not value:
                continue

            # place_term pair — pass through directly
            if flat_key in ("place_term", "place_term_type"):
                params[flat_key] = value
                continue

            # All other fields — map via lookup table
            url_param = FLAT_TO_URL_PARAM.get(flat_key)
            if url_param:
                params[url_param] = value
            # Silently skip unknown/unmapped keys (e.g. internal-only fields)

        if not params:
            return BASE_URL

        return f"{BASE_URL}?{urlencode(params)}"


# =============================================================================
# LLM EXTRACTOR  (Gemini 3.1 Flash-Lite Preview)
# =============================================================================

# JSON schema we constrain Gemini to — all fields are optional strings.
# Using the flat column names as keys so the output feeds directly into
# ArctosURLBuilder without any remapping.
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        # Identifiers
        "guid_prefix":            {"type": "string"},
        "guid":                   {"type": "string"},
        "cat_num":                {"type": "string"},
        # Taxonomy
        "taxonomy_search":        {"type": "string"},
        "scientific_name":        {"type": "string"},
        "kingdom":                {"type": "string"},
        "phylum":                 {"type": "string"},
        "phylclass":              {"type": "string"},
        "phylorder":              {"type": "string"},
        "suborder":               {"type": "string"},
        "superfamily":            {"type": "string"},
        "family":                 {"type": "string"},
        "subfamily":              {"type": "string"},
        "tribe":                  {"type": "string"},
        "subtribe":               {"type": "string"},
        "genus":                  {"type": "string"},
        "species":                {"type": "string"},
        "subspecies":             {"type": "string"},
        # Agents
        "collectors":             {"type": "string"},
        "identifiedby":           {"type": "string"},
        # Dates (must be ISO format YYYY-MM-DD)
        "began_date":             {"type": "string"},
        "ended_date":             {"type": "string"},
        "verbatim_date":          {"type": "string"},
        "month":                  {"type": "string"},
        "day":                    {"type": "string"},
        # Geography
        "country":                {"type": "string"},
        "state_prov":             {"type": "string"},
        "county":                 {"type": "string"},
        "place_term":             {"type": "string"},
        "place_term_type":        {"type": "string"},
        # Locality
        "spec_locality":          {"type": "string"},
        "locality_name":          {"type": "string"},
        "verbatim_locality":      {"type": "string"},
        "habitat":                {"type": "string"},
        # Elevation
        "min_elev_in_m":          {"type": "string"},
        "max_elev_in_m":          {"type": "string"},
        # Record metadata
        "typestatus":             {"type": "string"},
        "verificationstatus":     {"type": "string"},
        "collecting_method":      {"type": "string"},
        "collecting_source":      {"type": "string"},
        "has_tissues":            {"type": "string"},
        "identification_remarks": {"type": "string"},
    },
    "required": [],
}

_SYSTEM_PROMPT = """You are a structured data extractor for the Arctos natural history collections database.

Given a user query, extract ONLY the fields that are explicitly or clearly implied in the query.
Return a JSON object with ONLY the fields you are confident about — omit everything else entirely.

Rules:
- Dates must be ISO format (YYYY-MM-DD). Resolve relative expressions like "last year" using today's date.
  "after 1990" means began_date=1990-01-01. "in 1987" means began_date=1987-01-01 + ended_date=1987-12-31.
  Never put dates in verbatim_date unless the query uses explicitly non-standard date text (e.g. "Spring 1987").

- guid_prefix looks like "MVZ:Herp" or "ACUNHC:Mamm" — always INSTITUTION:COLLECTIONCODE format.
  ALWAYS put these in guid_prefix. Never put a guid_prefix value in any other field.

- Taxonomy: if a binomial species name is given (e.g. "Taricha rivularis"), put it in scientific_name only —
  do NOT also copy it into taxonomy_search.
  For common names like "birds" or "mammals", use the appropriate rank field (phylclass=Aves, phylclass=Mammalia).
  Never put a taxon name in a locality field like verbatim_locality or spec_locality.

- For place: use place_term + place_term_type together (e.g. place_term="Iowa", place_term_type="state_prov").
  NEVER also set state_prov / county / country for the same place — use place_term + place_term_type ONLY.
  Use just the name in place_term without the geographic type word — e.g. place_term="Alameda" not "Alameda County".
  If a query mentions multiple places (e.g. "California versus Oregon"), extract only the first one
  and ignore the rest — multi-geography queries cannot be expressed in a single Arctos URL.

- Never infer fields from context or assumptions. Only extract what is directly and explicitly stated
  in the query — if the user did not say it, do not include it. When in doubt, omit.
- Return only valid JSON, no markdown, no explanation.
"""


class LLMExtractor:
    """
    Calls Gemini 3.1 Flash-Lite Preview to extract structured search fields
    from a free-text query. Returns a dict ready for ArctosURLBuilder.
    """

    def __init__(self):
        import os
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = "gemini-3.1-flash-lite-preview"

    # Pricing for gemini-3.1-flash-lite-preview (USD per million tokens)
    INPUT_COST_PER_M  = 0.25
    OUTPUT_COST_PER_M = 1.50

    def extract(self, query: str) -> tuple[dict, dict]:
        import json
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_EXTRACTION_SCHEMA,
            ),
        )

        raw = response.text.strip()
        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw: {raw}")

        # Strip out any empty-string values the model may have included
        fields = {k: v for k, v in extracted.items() if v and str(v).strip()}

        # Token usage & cost
        meta = response.usage_metadata
        input_tokens  = meta.prompt_token_count or 0
        output_tokens = meta.candidates_token_count or 0
        cost_usd = (
            input_tokens  / 1_000_000 * self.INPUT_COST_PER_M +
            output_tokens / 1_000_000 * self.OUTPUT_COST_PER_M
        )
        usage = {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      cost_usd,
        }

        return fields, usage


if __name__ == "__main__":
    import sys

    CSV_PATH = "_portals (2).csv"

    entity_extractor = ArctosEntityExtractor(CSV_PATH)
    classifier = QuerybotClassifier(entity_extractor, threshold=0.90)
    url_builder = ArctosURLBuilder()

    test_queries = [
        # Local route (high entity coverage)
        "Show me ALMNH:Bird and APSU:Fish collections",
        "Find all specimens from Museum of Vertebrate Zoology",
        # LLM route (low entity coverage — needs semantic extraction)
        "What specimens of Taricha rivularis are in MVZ:Herp?",
        "How many species were collected in Alameda County in 1987?",
        "Get all bird specimens collected by John Doe after 1990",
        "Compare the diversity of mammals in California versus Oregon",
    ]

    print("=" * 80)
    print("QUERYBOT V2 — FULL PIPELINE DEMO")
    print(f"Threshold: {classifier.threshold * 100:.0f}%")
    print("=" * 80)

    # initialise the LLM extractor only if needed
    llm_extractor: LLMExtractor | None = None

    for i, query in enumerate(test_queries, 1):
        result = classifier.classify(query)
        print(f"\n{i}. Query : {query}")
        print(f"   Route         : {result['route'].upper()}")
        print(f"   Coverage      : {result['coverage'] * 100:.1f}%")

        if result["route"] == "local":
            entities = result["entities"]
            guids: set = set()

            # guid_prefix matches
            guids.update(entities["guid_prefixes"])

            # expand institution matches
            for inst in entities["institutions"]:
                guids.update(entity_extractor.institution_to_guids.get(inst, set()))

            # expand collection matches
            for coll in entities["collections"]:
                guids.update(entity_extractor.collection_to_guids.get(coll, set()))

            fields: dict = {}
            if guids:
                fields["guid_prefix"] = ",".join(sorted(guids))

            if entities["institutions"]:
                print(f"   Institutions  : {entities['institutions']}")
            if entities["collections"]:
                print(f"   Collections   : {entities['collections']}")

            url = url_builder.build(fields)
            print(f"   Fields        : {fields}")
            print(f"   URL           : {url}")

        else:
            # LLM route
            if llm_extractor is None:
                try:
                    llm_extractor = LLMExtractor()
                except EnvironmentError as e:
                    print(f"   [LLM SKIPPED] {e}")
                    continue

            try:
                fields, usage = llm_extractor.extract(query)
                if "guid_prefix" in fields:
                    fields["guid_prefix"] = entity_extractor.expand_guid_prefix(fields["guid_prefix"])
                url = url_builder.build(fields)
                print(f"   Fields        : {fields}")
                print(f"   URL           : {url}")
                print(f"   Tokens        : {usage['input_tokens']} in / {usage['output_tokens']} out")
                print(f"   Cost          : ${usage['cost_usd']:.6f}")
            except Exception as e:
                print(f"   [LLM ERROR]   {e}")

    print("\n" + "=" * 80)