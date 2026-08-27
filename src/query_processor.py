import re
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class ProcessedQuery:
    """Structured container for the processed & expanded query."""
    original_query: str
    normalized_query: str
    expanded_query: str
    detected_themes: List[str]
    detected_puranas: List[str]
    detected_deities: List[str]


class ScriptureQueryProcessor:
    """
    Architecture Block 3: Pre-Retrieval & Query Expansion Layer (v2 Production).

    Key Features:
      - Normalizes 18 canonical Mahapuranas & deity names (including literal 'vishnu').
      - Regex word-boundary matching (rf'\b{re.escape(trigger)}\b') prevents false
        positive semantic drift ('sin' in 'business', 'act' in 'reaction').
      - Maps emotional, philosophical, and cosmological queries to canonical Puranic themes.
    """

    # --- 18 Canonical Mahapuranas lookup & aliases ---
    PURANA_MAP: Dict[str, str] = {
        "vishnu": "Vishnu Purana",
        "shiva": "Shiva Purana",
        "siva": "Shiva Purana",
        "bhagavata": "Bhagavata Purana",
        "bhagavat": "Bhagavata Purana",
        "srimad bhagavatam": "Bhagavata Purana",
        "markandeya": "Markandeya Purana",
        "garuda": "Garuda Purana",
        "skanda": "Skanda Purana",
        "brahma": "Brahma Purana",
        "brahmanda": "Brahmanda Purana",
        "brahmavaivarta": "Brahmavaivarta Purana",
        "brahma vaivarta": "Brahmavaivarta Purana",
        "vayu": "Vayu Purana",
        "varaha": "Varaha Purana",
        "kurma": "Kurma Purana",
        "matsya": "Matsya Purana",
        "vamana": "Vamana Purana",
        "padma": "Padma Purana",
        "linga": "Linga Purana",
        "narada": "Narada Purana",
        "naradiya": "Narada Purana",
        "agni": "Agni Purana",
        "bhavishya": "Bhavishya Purana",
    }

    # --- Deity normalization mapping (includes literal names + aliases) ---
    DEITY_SYNONYMS: Dict[str, str] = {
        "vishnu": "Vishnu",
        "narayana": "Vishnu",
        "hari": "Vishnu",
        "govinda": "Vishnu",
        "madhava": "Vishnu",
        "keshava": "Vishnu",
        "mahadhar": "Vishnu",
        "shiva": "Shiva",
        "siva": "Shiva",
        "mahadeva": "Shiva",
        "shankar": "Shiva",
        "shankara": "Shiva",
        "rudra": "Shiva",
        "bholenath": "Shiva",
        "brahma": "Brahma",
        "prajapati": "Brahma",
        "krishna": "Krishna",
        "vasudeva": "Krishna",
        "rama": "Rama",
        "ramachandra": "Rama",
        "devi": "Devi/Durga",
        "durga": "Devi/Durga",
        "parvati": "Devi/Durga",
        "lakshmi": "Lakshmi",
        "saraswati": "Saraswati",
        "ganesha": "Ganesha",
        "ganapati": "Ganesha",
        "kartikeya": "Kartikeya",
        "murugan": "Kartikeya",
        "indra": "Indra",
    }

    # --- Core Puranic Theme Definitions & Word-Boundary Trigger Keywords ---
    THEME_DEFINITIONS: Dict[str, Dict[str, List[str]]] = {
        "Dharma & Duty": {
            "triggers": [
                "dharma", "duty", "righteous", "righteousness", "ethics", "moral", "grihastha",
                "rules of life", "conduct", "virtue", "truth", "satya", "responsibility"
            ],
            "expansions": ["varnashrama dharma", "grihastha duties", "righteous living", "moral conduct in Puranas"]
        },
        "Karma & Fate": {
            "triggers": [
                "karma", "destiny", "fate", "consequence", "rebirth", "past life",
                "punishment", "sin", "paapa", "punya", "merit", "action"
            ],
            "expansions": ["law of karma", "prarabdha", "fruits of action", "cause and effect in scriptures"]
        },
        "Bhakti & Devotion": {
            "triggers": [
                "bhakti", "devotion", "prayer", "worship", "surrender", "faith",
                "sharanagati", "puja", "mantra", "chant", "love of god", "grace", "kripa"
            ],
            "expansions": ["navavidha bhakti", "Prahlada devotion", "divine grace", "surrender to the divine"]
        },
        "Moksha & Liberation": {
            "triggers": [
                "moksha", "liberation", "enlightenment", "soul", "atman", "death",
                "afterlife", "reincarnation", "samsara", "detachment", "vairagya", "renunciation"
            ],
            "expansions": ["atman realization", "freedom from samsara", "vairagya", "ultimate liberation"]
        },
        "Creation & Cosmology": {
            "triggers": [
                "creation", "universe", "srishti", "yuga", "kalpa", "pralaya",
                "dissolution", "origin", "time cycles", "brahmanda", "elements"
            ],
            "expansions": ["sarga and pratisarga", "four yugas", "cosmic dissolution", "creation of universe"]
        },
        "Life Solace & Overcoming Adversity": {
            "triggers": [
                "sad", "hopeless", "depressed", "failure", "stress", "fear", "anxiety",
                "lost", "bad day", "pain", "struggle", "grief", "worry", "lonely", "trouble",
                "dukh", "chinta", "udaas", "pareshani", "चिंता", "दुख", "उदास"
            ],
            "expansions": [
                "overcoming fear and suffering", "Prahlada endurance in adversity",
                "Dhruva penance", "patience and mental peace", "divine solace in Mahapuranas"
            ]
        },
        "Avatars & Cosmic Battles": {
            "triggers": [
                "avatar", "incarnation", "demon", "asura", "rakshasa", "battle",
                "samudra manthan", "churning ocean", "boon", "varaha", "narasimha", "kurma", "matsya"
            ],
            "expansions": ["dashavatara", "churning of the milk ocean", "triumph of Devas over Asuras"]
        }
    }

    @classmethod
    def process_query(cls, raw_query: str) -> ProcessedQuery:
        clean_text = raw_query.strip()
        lower_text = clean_text.lower()

        # Step 1: Detect explicit Purana mentions
        detected_puranas: List[str] = []
        for key, canonical_name in cls.PURANA_MAP.items():
            pattern = rf'\b{re.escape(key)}(\s+purana)?\b'
            if re.search(pattern, lower_text):
                if canonical_name not in detected_puranas:
                    detected_puranas.append(canonical_name)

        # Step 2: Detect Deities mentioned
        detected_deities: List[str] = []
        for synonym, canonical_deity in cls.DEITY_SYNONYMS.items():
            pattern = rf'\b{re.escape(synonym)}\b'
            if re.search(pattern, lower_text):
                if canonical_deity not in detected_deities:
                    detected_deities.append(canonical_deity)

        # Step 3: Detect Themes using word-boundary regex
        detected_themes: List[str] = []
        expansion_keywords: Set[str] = set()

        for theme_name, theme_data in cls.THEME_DEFINITIONS.items():
            matched = False
            for trigger in theme_data["triggers"]:
                if re.search(rf'\b{re.escape(trigger)}\b', lower_text):
                    matched = True
                    break
            if matched:
                detected_themes.append(theme_name)
                expansion_keywords.update(theme_data["expansions"][:2])

        # Step 4: Construct the expanded query string
        enrichment_parts = []
        if detected_puranas:
            enrichment_parts.append(f"Scriptures: {', '.join(detected_puranas)}")
        if detected_deities:
            enrichment_parts.append(f"Deities: {', '.join(detected_deities)}")
        if detected_themes:
            enrichment_parts.append(f"Themes: {', '.join(detected_themes)}")
        if expansion_keywords:
            enrichment_parts.append(f"Concepts: {', '.join(sorted(list(expansion_keywords))[:3])}")

        if enrichment_parts:
            expanded_query = f"{clean_text} | {' | '.join(enrichment_parts)}"
        else:
            expanded_query = f"{clean_text} | Mahapurana authentic scripture wisdom"

        return ProcessedQuery(
            original_query=raw_query,
            normalized_query=clean_text,
            expanded_query=expanded_query,
            detected_themes=detected_themes,
            detected_puranas=detected_puranas,
            detected_deities=detected_deities
        )


if __name__ == "__main__":
    print("--- Phase 4: Pre-Retrieval Query Processor Test (v2 Production) ---\n")
    
    test_queries = [
        "I feel hopeless and lost in my life, everything is failing",
        "What does Vishnu Purana teach about Karma and Grihastha duties?",
        "Tell me about Samudra Manthan in Bhagavata Purana",
        "How can I find mental peace when facing extreme fear?",
        "What is the story of Prahlada in Bhagavata and Vishnu Purana?",
        "mujhe bahut chinta ho rahi hai jeevan mein",
        "The reaction was fast and fearless",
        "I am doing my regular business"
    ]

    for q in test_queries:
        result = ScriptureQueryProcessor.process_query(q)
        print(f"Original Query   : '{result.original_query}'")
        print(f"Detected Puranas : {result.detected_puranas}")
        print(f"Detected Deities : {result.detected_deities}")
        print(f"Detected Themes  : {result.detected_themes}")
        print(f"Expanded Query   :\n  --> {result.expanded_query}")
        print("-" * 75)