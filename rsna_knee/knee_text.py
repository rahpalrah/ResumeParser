"""
knee_text.py - multilingual rule lexicon for the radiology reports.

The corpus is ~38% English. Measured on the 4,407 reports: Romance ~21%,
Turkish ~12%, German ~9%, and a 5% Cyrillic block that is BULGARIAN, not
Russian - Bulgarian drops the soft sign, so "медиален" where Russian writes
"медиальный", and a Russian-shaped pattern matches none of it. Greek and
Croatian sit inside the Latin-script remainder. Turkish writes "medyal" at
least as often as "medial", while spelling "lateral" the same either way -
which is why lateral meniscus coverage ran 33 points ahead of medial until
that variant was added. Anatomy-coverage measurements drove
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

Abbreviations are deliberately absent where they collide with ordinary report
vocabulary. "MM" and "LM" for the menisci are the obvious trap: `\bmm\b`
matches every measurement in millimetres, which is most reports in every
language, and it held meniscus coverage at a flat 57% that no added
terminology could move.

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
    "ACL": (r"(anterior cruciate|\bacl\b|\blca\b|lig\w* cruzad\w*|ligamento[s]? cruzad\w*|"
            r"ligament croise anterieur|\bvkb\b|vorderes kreuzband|crociato anteriore|"
            r"on capraz bag|\bocb\b|anterior capraz|voorste kruisband|kruisband|"
            r"predn\w* krizn\w* ligament|krizn\w* ligament|"
            r"προσθι\w* χιαστ\w*|χιαστ\w*|"
            r"предна кръстна|преден кръстен|кръстн\w* (връзк|лигамент)\w*|"
            r"передн\w* крестообразн\w*|\bпкс\b)"),
    "MCL": (r"(medial collateral|\bmcl\b|\blcm\b|colateral medial|collateral medial|"
            r"ligament collateral (medial|interne)|innenband|mediales seitenband|"
            r"collaterale mediale|ic yan bag|med[iy]al kollateral|binnenband|"
            r"mediale collaterale band|"
            r"medijaln\w* kolateraln\w*|εσω πλαγι\w*|"
            r"медиал\w* колатерал\w*|вътрешн\w* странич\w*|"
            r"медиальн\w* коллатеральн\w*|внутренн\w* боков\w* связк)"),
    "Medial Meniscus": (r"((menisc|menisk|menisq)\w*\s+(medial|medyal|intern|mediale)|"
                        r"(medial|medyal|intern|mediale[ns]?|medijaln\w*)\s+(menisc|menisk|menisq)\w*|"
                        r"ic menisk\w*|menisk\w* med[iy]al|εσω μηνισκ\w*|"
                        r"(медиал\w*|вътрешн\w*)\s+мениск\w*|мениск\w*\s+(медиал|вътрешн)\w*|"
                        r"(медиальн\w*|внутренн\w*) мениск\w*)"),
    "Lateral Meniscus": (r"((menisc|menisk|menisq)\w*\s+(lateral|extern|laterale)|"
                         r"(lateral|extern|laterale[ns]?|lateraln\w*)\s+(menisc|menisk|menisq)\w*|"
                         r"dis menisk\w*|menisk\w* lateral|εξω μηνισκ\w*|"
                         r"(латерал\w*|външн\w*)\s+мениск\w*|мениск\w*\s+(латерал|външн)\w*|"
                         r"(латеральн\w*|наружн\w*) мениск\w*)"),
    "Medial OA": (r"((compartiment|compartment|kompartiment|kompartman)\w*\s+(medial|medyal|intern|ic)|"
                  r"(medial|medyal|intern)\w*\s+(compartment|compartiment|kompartiment|kompartman)|"
                  r"femorotibia\w* (medial|medyal|intern|mediaal)|(medial|mediaal)\w* femorotibia\w*|"
                  r"medial tibiofemoral|medial joint|medial femoral condyle|medial tibial plateau|"
                  r"condilo femoral medial|meseta tibial medial|mediale? (femurcondyl|tibiaplateau)\w*|"
                  r"med[iy]al (kompartman|femoral kondil|tibial plato)\w*|"
                  r"εσω (διαμερισμα|μηριαιοκνημιαι|κονδυλ)\w*|"
                  r"medijaln\w* (kondil|femoraln|tibijaln|kompartment)\w*|"
                  r"(медиал\w*|вътрешн\w*) (отдел|компартимент|кондил|тибиал)\w*|"
                  r"медиал\w* феморал\w*|"
                  r"(медиальн\w*|внутренн\w*) (отдел|компартмент)\w*)"),
    "Lateral OA": (r"((compartiment|compartment|kompartiment|kompartman)\w*\s+(lateral|extern|dis)|"
                   r"(lateral|extern)\w*\s+(compartment|compartiment|kompartiment|kompartman)|"
                   r"femorotibia\w* (lateral|extern|lateraal)|(lateral|lateraal)\w* femorotibia\w*|"
                   r"lateral tibiofemoral|lateral joint|lateral femoral condyle|lateral tibial plateau|"
                   r"condilo femoral lateral|meseta tibial lateral|laterale? (femurcondyl|tibiaplateau)\w*|"
                   r"lateral (kompartman|femoral kondil|tibial plato)\w*|"
                   r"εξω (διαμερισμα|μηριαιοκνημιαι|κονδυλ)\w*|"
                   r"lateraln\w* (kondil|femoraln|tibijaln|kompartment)\w*|"
                   r"(латерал\w*|външн\w*) (отдел|компартимент|кондил|тибиал)\w*|"
                   r"латерал\w* феморал\w*|"
                   r"(латеральн\w*|наружн\w*) (отдел|компартмент)\w*)"),
    "PF OA": (r"(patellofemoral|patelo?femoral|femoropatellar|femoro-?patellaire|"
              r"retropatellar|patellarruckflache|trochlea|troklea|trohlear|"
              r"patelofemoraln\w*|fasete patele|επιγονατιδομηριαι\w*|"
              r"пателофеморал\w*|ретропателар\w*|пателарн\w*|"
              r"пателлофеморальн\w*|бедренно-?надколенник\w*|ретропателлярн\w*)"),
    "Effusion": (r"(effusion|derrame|epanchement|erguss|versamento|joint fluid|"
                 r"liquido articular|hydrops|efuzyon|eklem ici sivi|eklemde sivi|"
                 r"gewrichtsvocht|vocht in het gewricht|"
                 r"sivi artisi|sivi miktari|izljev|zglobn\w* tekucin\w*|"
                 r"συλλογη υγρου|ενδαρθρικη συλλογη|"
                 r"ставен излив|ставния излив|излив в|"
                 r"выпот\w*|жидкост\w* в полости|синовиальн\w* жидкост\w*)"),
    "Synovitis": (r"(synovitis|sinovitis|synovite|synovialitis|sinovite|sinovit\w*|"
                  r"synovial (thickening|proliferation|hypertrophy)|pannus|υμενιτιδ\w*|"
                  r"sinovyal (kalinlas|hipertrofi|proliferasyon)\w*|"
                  r"synovia(le)? (verdikking|proliferatie|hypertrofie)|"
                  r"sinovij\w* (zadebljan|proliferacij|hipertrofij)\w*|"
                  r"υμενικ\w* (υπερτροφ|παχυνσ)\w*|синовиал\w* (задебел|пролифер)\w*|"
                  r"синовит\w*|утолщени\w* синовиальн\w*)"),
    "Baker's": (r"(baker|popliteal cyst|quiste de baker|kyste poplite|bakerzyste|"
                r"cisti di baker|poplitealzyste|baker kisti|popliteal kist|"
                r"bakerova cist\w*|κυστη (του )?baker|bakercyste|poplitea(le)? cyste|"
                r"киста в подкол\w*|подкол\w* ямка|"
                r"киста на беикер\w*|беикер\w*|подкол[яе]н\w* киста|"
                r"подколенн\w* киста)"),
    "Contusion": (r"(contusion|bone bruise|bone marrow (edema|oedema)|edema (oseo|de medula)|"
                  r"knochenmarkodem|edema midollare|medullar\w* edema|marrow oedema|"
                  r"kemik iligi odem\w*|kontuzyon|kemik odem\w*|kostan\w* edem\w*|edem kosti|"
                  r"botoedeem|beenmergoedeem|"
                  r"kostan\w* kontuzij\w*|οιδημα (του )?μυελου|"
                  r"костномозъчен едем|костно-?мозъчен оток|"
                  r"отек костного мозга|контузи\w*|трабекулярн\w* отек)"),
    "Fracture": (r"(fracture|fractura|frattura|fraktur|kirik|avulsion|avulsiyon|"
                 r"impaction fracture|insufficiency fracture|fissur\w*|prijelom\w*|"
                 r"καταγμα\w*|фрактур\w*|счупван\w*|fractuur|"
                 r"перелом\w*|отрыв\w* фрагмент)"),
}

ABNORMAL = (r"(tear|tears|torn|rupt\w*|rott\w*|rotur\w*|ris[sx]\w*|lesion\w*|lesao\w*|"
            r"lacerat\w*|desgarr\w*|dechirure|discontinu\w*|sprain|esguince|entorse|"
            r"zerrung|degenerat\w*|signal alterat\w*|grade (i{1,3}|[123])\b|abnormal|"
            r"engrosamiento|enthesopat\w*|"
            r"yirti[kglm]\w*|yirtil\w*|dejenerasyon|dejeneratif|sinyal artisi|"
            r"degeneracij\w*|ozljed\w*|lezij\w*|ruptur\w*|"
            r"ρηξη|βλαβη|εκφυλιστικ\w*|"
            r"скъсван\w*|разкъсван\w*|лезия|увред\w*|руптур\w*|дегенеративн\w*|"
            r"разрыв\w*|надрыв\w*|поврежден\w*|повышени\w* сигнала)")

OA_TERMS = (r"(osteoarthrit|arthros|artros|artroz|artrit|gonarthros|gonartroz|arthrosis|"
            r"chondromalac|condromalac|kondromalaz\w*|hondromalacij\w*|"
            r"condropat\w*|hondropat\w*|chondropath\w*|"
            r"kikirdak (kayb|incel|hasar)\w*|eklem aralig\w* daral\w*|"
            r"cartilage (loss|thinning|defect)|perdida de cartilago|knorpel|"
            r"kraakbeen\w*|artrose|chondropathie|ulceras condrales|"
            r"osteophyt|osteofito|osteofit\w*|joint space narrowing|pincement|"
            r"fisure hrskavice|hrskavic\w* (defekt|stanjen)\w*|erozivn\w*|"
            r"ostecen\w* hrskavic\w*|degenerativn\w* promjen\w*|stanjenj\w*|"
            r"χονδροπαθει\w*|χονδρομαλακ\w*|οστεοαρθριτ\w*|οστεοφυτ\w*|"
            r"хондропат\w*|остеофит\w*|изтънен\w*|дегенеративн\w* промен\w*|"
            r"артроз\w*|хондромаляц\w*|сужени\w* суставн\w* щели|"
            r"дегенеративн\w* изменени)")

# Negation that PRECEDES the finding - the usual order in English, Romance,
# Germanic, Slavic and Greek.
NEGATION = (r"(no |not |without |sin |sans |kein |keine |nessun|negative for|"
            r"unremarkable|intact|normal|integr\w*|ausgeschlossen|descartad|"
            r"bez |uredn\w*|geen |zonder |normaal|ongestoord|"
            r"δεν |χωρις|φυσιολογικ\w*|"
            r"няма|без особености|\bб\.о\.|не се|"
            r"\bne \b|без |отсутств\w*)")

# Negation that FOLLOWS the finding.  Turkish is verb-final - "Efuzyon
# izlenmedi" is "effusion was not observed" - and Bulgarian, Croatian and
# Russian all place the qualifier after the noun: "менискус с нормален мр
# образ", "ligament urednog signala". Checking only backwards would take every
# such report as positive for everything it mentions, which is the worst
# possible failure for a lexicon whose whole job is precision.
NEGATION_AFTER = (r"(izlenmedi|izlenmemis\w*|saptanmadi|saptanmamis\w*|gozlenmedi|"
                  r"goruldu degil|mevcut degil|yoktur|\byok\b|dogal|normal|intakt|"
                  r"normaldir|korunmus|"
                  r"uredn\w*|bez znakova|u kontinuitetu|"
                  r"φυσιολογικ\w*|χωρις παθολογ\w*|"
                  r"нормал\w*|правилна форма|запазен\w*|без особености|"
                  r"не се (визуализира|проследява|установява)\w*|"
                  r"не выявлен\w*|не определя\w*|не отмеча\w*|не визуализир\w*|"
                  r"отсутств\w*|интактн\w*|сохранен\w*)")

# Severity qualifiers.  A binary "the report mentions an effusion" rule fires
# on 83% of this corpus while only 60% of annotated studies are positive - the
# discriminating signal is how much, not whether.  Grading turns the rule into
# a ranked score, which is what a metric built on AUC rewards.
MILD = (r"(minimal|trace|tiny|small|slight|mild|discret\w*|leve|escas\w*|scars\w*|"
        r"gering\w*|weinig|klein|hafif|az miktarda|blag\w*|"
        r"минимал\w*|малк\w*|неголям\w*|ελαφρ\w*|μικρ\w*)")
SEVERE = (r"(large|gross|massive|marked|severe|significant|abundant|advanced|"
          r"importante|grande|avanzad\w*|ausgepragt|gevorderd|uitgebreid|belangrijk|"
          r"belirgin|yaygin|masif|ileri derecede|"
          r"голям\w*|изразен\w*|обилен|значител\w*|εκτεταμεν\w*|μεγαλ\w*|"
          r"velik\w*|obilan|uznapredoval\w*)")


def _grade(chunk: str) -> float:
    """Confidence for a hit, from the severity words around it."""
    if re.search(SEVERE, chunk):
        return 1.0
    if re.search(MILD, chunk):
        return 0.35
    return 0.7


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


# A conjunction between a finding and a following negation means the negation
# belongs to a separate predicate.
CONJUNCTION = re.compile(r"(\band\b|\bve\b|\bи\b|\bte\b|\bili\b|\bveya\b|\bund\b|,)")


def _negated(chunk: str, start: int, end: int, look: int = 45) -> bool:
    """True if a negation cue sits just before, or just after, the span.

    The conjunction guard matters for Bulgarian in particular: "връзка е
    руптурирана и не се проследява до залавните си места" is a ruptured
    ligament that CANNOT be traced - the "не се проследява" is a second
    predicate describing the consequence, not a denial of the rupture. Without
    the guard the most explicit positive phrasing in the language reads as a
    negative.
    """
    if re.search(NEGATION, chunk[max(0, start - look):start]):
        return True
    after = chunk[end:end + look]
    m = re.search(NEGATION_AFTER, after)
    if not m:
        return False
    return CONJUNCTION.search(after[:m.start()]) is None


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
               label: Optional[str] = None, spans: Optional[list] = None) -> float:
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
            return _grade(chunk)
    return 0.0


def rule_features(text: str) -> np.ndarray:
    """Twelve graded rule scores in LABELS order: 0 for no hit or a negated
    one, then 0.35 / 0.7 / 1.0 by severity. Binarise at >0.5 where a hard
    label is needed."""
    f = np.zeros(N_LABELS, np.float32)
    spans = anatomy_spans(text)
    for i, lab in enumerate(LABELS):
        anat = ANATOMY[lab]
        if lab in STANDALONE:
            hit = 0.0
            for m in re.finditer(anat, text):
                lo, hi = _clause_bounds(text, m.start())
                chunk = text[lo:hi]
                if not _negated(chunk, m.start() - lo, m.end() - lo):
                    hit = max(hit, _grade(chunk))
            f[i] = hit
        elif lab in OA_LABELS:
            f[i] = window_hit(text, anat, OA_TERMS, label=lab, spans=spans)
        else:
            f[i] = window_hit(text, anat, ABNORMAL, label=lab, spans=spans)
    return f
