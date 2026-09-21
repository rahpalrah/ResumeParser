"""
knee_text.py - multilingual rule lexicon for the radiology reports.

The corpus is ~38% English. A stopword probe over the 4,407 reports puts
Spanish/Portuguese/French around 21%, Turkish around 12%, German around 9% and
Russian around 5%, with the rest scattered. Anatomy-coverage measurements drove
every entry here: essentially every knee MRI report comments on the ACL and the
menisci, so a pattern matching only 40% of reports is a lexicon gap, not a
property of the data.

Two normalisation traps are handled in norm_text, and both silently produce
zero matches rather than an error:

  * Turkish dotless i (U+0131) has no NFKD decomposition, so accent-stripping
    leaves "yirtik" spelled with it and no ASCII pattern matches.
  * Cyrillic IS affected by accent-stripping, contrary to first instinct:
    NFKD decomposes й into и + breve and ё into е + diaeresis, so the stripped
    text reads "беикера", not "бейкера". Every Cyrillic pattern below is
    therefore written in post-normalisation form, and selftest.py asserts that
    no pattern contains a character norm_text would alter.

The rules are not the final labeller. They are a precise, sparse signal that
step 2 self-trains from, so precision is worth far more than recall here.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np

from knee_common import LABELS, N_LABELS

# Findings whose unnegated mention is itself the finding.
STANDALONE = {"Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}
OA_LABELS = {"Medial OA", "Lateral OA", "PF OA"}

# Characters with no NFKD decomposition, which accent-stripping cannot reach.
_CHAR_MAP = str.maketrans({
    "ı": "i",   # Turkish dotless i - the one that matters most here
    "ł": "l", "ø": "o", "đ": "d", "ð": "d",
    "ß": "ss", "æ": "ae", "œ": "oe", "þ": "th",
})

ANATOMY = {
    "ACL": (r"(anterior cruciate|\bacl\b|\blca\b|lig\w* cruzado anterior|"
            r"ligament croise anterieur|\bvkb\b|vorderes kreuzband|crociato anteriore|"
            r"on capraz bag|\bocb\b|anterior capraz|"
            r"передн\w* крестообразн\w*|\bпкс\b)"),
    "MCL": (r"(medial collateral|\bmcl\b|\blcm\b|colateral medial|collateral medial|"
            r"ligament collateral (medial|interne)|innenband|mediales seitenband|"
            r"collaterale mediale|ic yan bag|medial kollateral|"
            r"медиальн\w* коллатеральн\w*|внутренн\w* боков\w* связк)"),
    "Medial Meniscus": (r"((menisc|menisk|menisq)\w*\s+(medial|intern|mediale)|"
                        r"(medial|intern|mediale[ns]?)\s+(menisc|menisk|menisq)\w*|"
                        r"ic menisk\w*|menisk\w* medial|"
                        r"(медиальн\w*|внутренн\w*) мениск\w*|мениск\w* (медиальн|внутренн)\w*|"
                        r"\bmm\b)"),
    "Lateral Meniscus": (r"((menisc|menisk|menisq)\w*\s+(lateral|extern|laterale)|"
                         r"(lateral|extern|laterale[ns]?)\s+(menisc|menisk|menisq)\w*|"
                         r"dis menisk\w*|menisk\w* lateral|"
                         r"(латеральн\w*|наружн\w*) мениск\w*|мениск\w* (латеральн|наружн)\w*|"
                         r"\blm\b)"),
    "Medial OA": (r"((compartiment|compartment|kompartiment|kompartman)\w*\s+(medial|intern|ic)|"
                  r"(medial|intern)\w*\s+(compartment|compartiment|kompartiment|kompartman)|"
                  r"femorotibial (medial|intern)|medial tibiofemoral|medial joint|"
                  r"(медиальн\w*|внутренн\w*) (отдел|компартмент)\w*)"),
    "Lateral OA": (r"((compartiment|compartment|kompartiment|kompartman)\w*\s+(lateral|extern|dis)|"
                   r"(lateral|extern)\w*\s+(compartment|compartiment|kompartiment|kompartman)|"
                   r"femorotibial (lateral|extern)|lateral tibiofemoral|lateral joint|"
                   r"(латеральн\w*|наружн\w*) (отдел|компартмент)\w*)"),
    "PF OA": (r"(patellofemoral|patelo?femoral|femoropatellar|femoro-?patellaire|"
              r"retropatellar|patellarruckflache|trochlea|troklea|"
              r"пателлофеморальн\w*|бедренно-?надколенник\w*|ретропателлярн\w*)"),
    "Effusion": (r"(effusion|derrame|epanchement|erguss|versamento|joint fluid|"
                 r"liquido articular|hydrops|efuzyon|eklem sivisi|eklem ici sivi|"
                 r"выпот\w*|жидкост\w* в полости|синовиальн\w* жидкост\w*)"),
    "Synovitis": (r"(synovitis|sinovitis|synovite|synovialitis|sinovite|sinovit\w*|"
                  r"synovial (thickening|proliferation)|pannus|синовит\w*|"
                  r"утолщени\w* синовиальн\w*)"),
    "Baker's": (r"(baker|popliteal cyst|quiste de baker|kyste poplite|bakerzyste|"
                r"cisti di baker|poplitealzyste|baker kisti|popliteal kist|"
                r"киста беикера|беикера|подколенн\w* киста)"),
    "Contusion": (r"(contusion|bone bruise|bone marrow (edema|oedema)|edema (oseo|de medula)|"
                  r"knochenmarkodem|edema midollare|medullar\w* edema|marrow oedema|"
                  r"kemik iligi odem\w*|kontuzyon|kemik odem\w*|"
                  r"отек костного мозга|контузи\w*|трабекулярн\w* отек)"),
    "Fracture": (r"(fracture|fractura|frattura|fraktur|kirik|avulsion|avulsiyon|"
                 r"impaction fracture|insufficiency fracture|fissur\w*|"
                 r"перелом\w*|отрыв\w* фрагмент)"),
}

ABNORMAL = (r"(tear|tears|torn|rupt\w*|rott\w*|rotur\w*|ris[sx]\w*|lesion\w*|lesao\w*|"
            r"lacerat\w*|desgarr\w*|dechirure|discontinu\w*|sprain|esguince|entorse|"
            r"zerrung|degenerat\w*|signal alterat\w*|grade (ii|iii|2|3)|abnormal|"
            r"yirti[kglm]\w*|yirtil\w*|dejenerasyon|dejeneratif|sinyal artisi|"
            r"разрыв\w*|надрыв\w*|поврежден\w*|дегенеративн\w*|повышени\w* сигнала)")

OA_TERMS = (r"(osteoarthrit|arthros|artros|artroz|artrit|gonarthros|gonartroz|arthrosis|"
            r"chondromalac|condromalac|kondromalaz\w*|kikirdak kayb\w*|kikirdak incel\w*|"
            r"cartilage (loss|thinning|defect)|perdida de cartilago|knorpel|"
            r"osteophyt|osteofito|osteofit\w*|joint space narrowing|pincement|"
            r"eklem aralig\w* daral\w*|"
            r"артроз\w*|остеофит\w*|хондромаляц\w*|сужени\w* суставн\w* щели|"
            r"дегенеративн\w* изменени)")

# Negation that PRECEDES the finding - the normal order in English and the
# Romance and Germanic languages here.
NEGATION = (r"(no |not |without |sin |sans |kein |keine |nessun|negative for|"
            r"unremarkable|intact|normal|integr\w*|ausgeschlossen|descartad|"
            r"\bne \b|без |отсутств\w*)")

# Negation that FOLLOWS the finding.  Turkish is verb-final - "Efuzyon
# izlenmedi" is "effusion was not observed" - and Russian commonly places the
# negated verb after the noun.  Checking only backwards would take every such
# report as positive for everything it mentions, which is the worst possible
# failure for a lexicon whose whole job is precision.
NEGATION_AFTER = (r"(izlenmedi|izlenmemis\w*|saptanmadi|saptanmamis\w*|gozlenmedi|"
                  r"goruldu degil|mevcut degil|yoktur|\byok\b|dogal|normal|intakt|"
                  r"не выявлен\w*|не определя\w*|не отмеча\w*|не визуализир\w*|"
                  r"отсутств\w*|интактн\w*|сохранен\w*)")

# ':' is deliberately NOT a break: "Medial meniscus: posterior horn tear"
# is one statement, and radiologists write findings that way constantly.
SENT_BREAK = re.compile(r"[.;\n]")


def norm_text(s: Optional[str]) -> str:
    """Lowercase, map the undecomposable characters, strip accents, squeeze space."""
    if not isinstance(s, str):
        return ""
    s = s.lower().translate(_CHAR_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def _clause_bounds(text: str, pos: int) -> tuple:
    """Start and end of the clause containing `pos`."""
    breaks = [m.end() for m in SENT_BREAK.finditer(text[:pos])]
    lo = breaks[-1] if breaks else 0
    m = SENT_BREAK.search(text, pos)
    return lo, (m.start() if m else len(text))


def _negated(chunk: str, start: int, end: int, look: int = 45) -> bool:
    """True if a negation cue sits just before or just after the span."""
    if re.search(NEGATION, chunk[max(0, start - look):start]):
        return True
    return re.search(NEGATION_AFTER, chunk[end:end + look]) is not None


def _nearest(chunk: str, pattern: str, lo: int, hi: int):
    """The match of `pattern` closest to the span [lo, hi).

    Taking the nearest rather than the first finding term matters inside one
    clause: "tear of the medial meniscus, intact lateral meniscus" would
    otherwise attach the tear to whichever structure the regex reached first.
    """
    best, best_d = None, None
    for m in re.finditer(pattern, chunk):
        if m.start() >= hi:
            d = m.start() - hi
        elif m.end() <= lo:
            d = lo - m.end()
        else:
            d = 0
        if best_d is None or d < best_d:
            best, best_d = m, d
    return best


def _span_distance(lo: int, hi: int, p_lo: int, p_hi: int) -> int:
    if p_lo >= hi:
        return p_lo - hi
    if p_hi <= lo:
        return lo - p_hi
    return 0


def anatomy_spans(text: str) -> list:
    """Every anatomy mention in the text, as (start, end, label)."""
    out = []
    for lab, pat in ANATOMY.items():
        for m in re.finditer(pat, text):
            out.append((m.start(), m.end(), lab))
    return out


def window_hit(text: str, anat: str, finding: str, span: int = 90,
               label: Optional[str] = None, spans: Optional[list] = None) -> int:
    """1 if a finding term sits in the same clause as an anatomy term, within
    `span` characters, is not negated on either side, and is closer to THIS
    structure than to any other structure mentioned in the report.

    That last condition is what separates "tear of the medial meniscus, intact
    lateral meniscus" into one positive rather than two: a comma is not a
    clause break, so both menisci see the same "tear", and only the nearer one
    may claim it. Competitors are restricted to the same clause, or a structure
    named in the previous sentence would steal findings across the full stop.
    """
    spans = anatomy_spans(text) if spans is None else spans
    for m in re.finditer(anat, text):
        c_lo, c_hi = _clause_bounds(text, m.start())
        lo, hi = max(c_lo, m.start() - span), min(c_hi, m.end() + span)
        chunk = text[lo:hi]
        a_lo, a_hi = m.start() - lo, m.end() - lo
        fm = _nearest(chunk, finding, a_lo, a_hi)
        if not fm or _negated(chunk, fm.start(), fm.end()):
            continue
        f_lo, f_hi = fm.start() + lo, fm.end() + lo
        mine = _span_distance(m.start(), m.end(), f_lo, f_hi)
        # Only structures named in the SAME clause may compete for the finding.
        # "Kein Erguss. Riss des Innenbandes." has the effusion mention nearer
        # to "Riss" by raw character distance, but it belongs to another
        # sentence and has no claim on it.
        stolen = any(lab2 != label
                     and c_lo <= s2 and e2 <= c_hi
                     and _span_distance(s2, e2, f_lo, f_hi) < mine
                     for s2, e2, lab2 in spans)
        if not stolen:
            return 1
    return 0


def rule_features(text: str) -> np.ndarray:
    """Twelve binary rule hits, in LABELS order."""
    f = np.zeros(N_LABELS, np.float32)
    spans = anatomy_spans(text)
    for i, lab in enumerate(LABELS):
        anat = ANATOMY[lab]
        if lab in STANDALONE:
            hit = 0
            for m in re.finditer(anat, text):
                lo, hi = _clause_bounds(text, m.start())
                chunk = text[lo:hi]
                if not _negated(chunk, m.start() - lo, m.end() - lo):
                    hit = 1
                    break
            f[i] = hit
        elif lab in OA_LABELS:
            f[i] = window_hit(text, anat, OA_TERMS, label=lab, spans=spans)
        else:
            f[i] = window_hit(text, anat, ABNORMAL, label=lab, spans=spans)
    return f
