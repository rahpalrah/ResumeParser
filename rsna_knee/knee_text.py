"""
knee_text.py - multilingual rule lexicon for the radiology reports.

The reports come from several countries and are written in several languages.
Rather than one pattern per language, each entry is a set of *stems* that
survive accent-stripping, so "menisco / menisque / meniskus" or "rotura /
rottura / ruptur" collapse into one alternation.

The rules are not meant to be the final labeller - they are a precise, sparse
feature vector handed to the transformer in step 2, plus a floor to measure it
against.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np

from knee_common import LABELS, N_LABELS

# Findings whose mere mention (unnegated) is the finding - there is no separate
# "abnormal" qualifier to look for.
STANDALONE = {"Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}
OA_LABELS = {"Medial OA", "Lateral OA", "PF OA"}

ANATOMY = {
    "ACL": r"(anterior cruciate|\bacl\b|\blca\b|lig\w* cruzado anterior|ligament croise anterieur|\bvkb\b|vorderes kreuzband|crociato anteriore)",
    "MCL": r"(medial collateral|\bmcl\b|\blcm\b|colateral medial|collateral medial|ligament collateral (medial|interne)|innenband|mediales seitenband|collaterale mediale)",
    "Medial Meniscus": r"((menisc|menisk|menisq)\w*\s+(medial|intern|mediale)|(medial|intern|mediale[ns]?)\s+(menisc|menisk|menisq)\w*|\bmm\b)",
    "Lateral Meniscus": r"((menisc|menisk|menisq)\w*\s+(lateral|extern|laterale)|(lateral|extern|laterale[ns]?)\s+(menisc|menisk|menisq)\w*|\blm\b)",
    "Medial OA": r"((compartiment|compartment|kompartiment)\w*\s+(medial|intern)|(medial|intern)\w*\s+(compartment|compartiment|kompartiment)|femorotibial (medial|intern)|medial tibiofemoral|medial joint)",
    "Lateral OA": r"((compartiment|compartment|kompartiment)\w*\s+(lateral|extern)|(lateral|extern)\w*\s+(compartment|compartiment|kompartiment)|femorotibial (lateral|extern)|lateral tibiofemoral|lateral joint)",
    "PF OA": r"(patellofemoral|patelo?femoral|femoropatellar|femoro-?patellaire|retropatellar|patellarruckflache|trochlea)",
    "Effusion": r"(effusion|derrame|epanchement|erguss|versamento|joint fluid|liquido articular|hydrops)",
    "Synovitis": r"(synovitis|sinovitis|synovite|synovialitis|sinovite|synovial (thickening|proliferation)|pannus)",
    "Baker's": r"(baker|popliteal cyst|quiste de baker|kyste poplite|bakerzyste|cisti di baker|poplitealzyste)",
    "Contusion": r"(contusion|bone bruise|bone marrow (edema|oedema)|edema (oseo|de medula)|knochenmarkodem|edema midollare|medullar\w* edema|marrow oedema)",
    "Fracture": r"(fracture|fractura|frattura|fraktur|kirik|avulsion|impaction fracture|insufficiency fracture)",
}

ABNORMAL = (r"(tear|tears|torn|rupt\w*|rott\w*|rotur\w*|ris[sx]\w*|lesion\w*|lesao\w*|lacerat\w*|desgarr\w*|"
            r"dechirure|discontinu\w*|sprain|esguince|entorse|zerrung|degenerat\w*|"
            r"signal alterat\w*|grade (ii|iii|2|3)|abnormal)")

OA_TERMS = (r"(osteoarthrit|arthros|artros|gonarthros|arthrosis|chondromalac|condromalac|"
            r"cartilage (loss|thinning|defect)|perdida de cartilago|knorpel|osteophyt|"
            r"osteofito|joint space narrowing|pincement)")

NEGATION = (r"(no |not |without |sin |sans |kein |keine |nessun|negative for|unremarkable|"
            r"intact|normal|integr\w*|ausgeschlossen|descartad)")


def norm_text(s: Optional[str]) -> str:
    """Lowercase, strip accents, squeeze whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


# ':' is deliberately NOT a break: "Medial meniscus: posterior horn tear"
# is one statement, and radiologists write findings that way constantly.
SENT_BREAK = re.compile(r"[.;\n]")


def _negated_before(chunk: str, at: int, look: int = 45) -> bool:
    """True if a negation cue sits just before position `at`, in the SAME clause.

    Clause-scoping is essential, not a refinement: "ACL is intact. Tear of the
    lateral meniscus." would otherwise have the meniscus tear suppressed by the
    negation belonging to the previous sentence, which is exactly the phrasing
    radiologists use when they enumerate normal structures before the abnormal
    one.
    """
    before = chunk[:at]
    breaks = [m.end() for m in SENT_BREAK.finditer(before)]
    clause_start = breaks[-1] if breaks else 0
    window = before[max(clause_start, at - look):]
    return re.search(NEGATION, window) is not None


def _clause_bounds(text: str, pos: int) -> tuple:
    """Start and end of the clause containing `pos`."""
    breaks = [m.end() for m in SENT_BREAK.finditer(text[:pos])]
    lo = breaks[-1] if breaks else 0
    m = SENT_BREAK.search(text, pos)
    return lo, (m.start() if m else len(text))


def window_hit(text: str, anat: str, finding: str, span: int = 90) -> int:
    """1 if a finding term appears in the same clause as an anatomy term (and
    within `span` characters of it) without a negation cue in front of it.

    Clause scoping is what makes this usable. Reports routinely read "The ACL is
    intact. Tear of the lateral meniscus." - a plain character window around the
    ACL mention would reach into the next sentence and call every such study an
    ACL tear.
    """
    for m in re.finditer(anat, text):
        lo, hi = _clause_bounds(text, m.start())
        lo, hi = max(lo, m.start() - span), min(hi, m.end() + span)
        chunk = text[lo:hi]
        fm = re.search(finding, chunk)
        if fm and not _negated_before(chunk, fm.start()):
            return 1
    return 0


def rule_features(text: str) -> np.ndarray:
    """Twelve binary rule hits, in LABELS order."""
    f = np.zeros(N_LABELS, np.float32)
    for i, lab in enumerate(LABELS):
        anat = ANATOMY[lab]
        if lab in STANDALONE:
            m = re.search(anat, text)
            f[i] = 1.0 if (m and not _negated_before(text, m.start())) else 0.0
        elif lab in OA_LABELS:
            f[i] = window_hit(text, anat, OA_TERMS)
        else:
            f[i] = window_hit(text, anat, ABNORMAL)
    return f
