"""Rule-based risk detector that decides which safety warning accompanies an answer.

Deliberately *not* a learned model: safety behaviour must be predictable, auditable and
never silently degrade. The assistant still answers (design decision), but:
  * emergency     -> leads with "seek urgent care now" (Rwanda SAMU ambulance: 912)
  * special_group -> pregnancy, breastfeeding, children, older adults, chronic disease:
                     answer + explicit "confirm with a doctor or pharmacist first"
  * prescription  -> prescription-only medicines (antibiotics, controlled drugs, ...):
                     reminds that a prescription and clinician assessment are required
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMERGENCY = [
    r"chest pain", r"(can'?t|cannot|hard to|difficulty|trouble) breath", r"short(ness)? of breath",
    r"over ?dos", r"took (too many|too much|\d+ (pills|tablets))", r"swallow(ed)? .*(bottle|pills|tablets)",
    r"poison", r"unconscious", r"passed out", r"faint(ed|ing)", r"seizure", r"convulsion", r"\bfits?\b",
    r"suicid", r"kill (my ?self|myself)", r"self[- ]harm", r"end my life",
    r"(heavy|severe|won'?t stop|not stopping) bleed", r"vomit(ing)? blood", r"blood in (my )?(stool|poo|vomit)",
    r"face (is )?droop", r"slurred speech", r"numb(ness)? (on )?one side", r"stroke",
    r"swollen (throat|tongue|lips|face)", r"(throat|tongue|lips|face) (is |are )?(swell|swollen)", r"anaphyla",
    r"stiff neck", r"snake ?bite", r"bitten by a (snake|dog)", r"dog bite", r"(severe|deep|large|serious) (burn|wound|cut)",
    r"head injury", r"hit (my|his|her) head",
]
_SPECIAL = [
    r"pregnan", r"breast ?feed", r"nursing", r"lactat", r"\bbab(y|ies)\b", r"infant", r"newborn", r"toddler",
    r"\bchild(ren)?\b", r"\bkids?\b", r"my (son|daughter)", r"\b\d{1,2}[- ]?(month|year)s?[- ]?old\b",
    r"elderly", r"old(er)? (age|adult|person|people)", r"grandparent", r"grand(ma|pa|mother|father)",
    r"kidney (disease|failure|problem)", r"liver (disease|failure|problem)", r"hepatitis", r"\bhiv\b",
    r"diabet", r"heart (disease|failure|problem|condition)", r"epilep", r"asthma", r"high blood pressure",
    r"hypertension", r"allerg", r"warfarin", r"blood thinner",
]
_PRESCRIPTION = [
    r"antibiotic", r"amoxicillin", r"azithromycin", r"ciprofloxacin", r"doxycycline", r"metronidazole",
    r"ceftriaxone", r"augmentin", r"cotrimoxazole", r"septrin", r"tramadol", r"codeine", r"morphine",
    r"diazepam", r"valium", r"misoprostol", r"without (a )?prescription", r"no prescription",
]

_RE = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in
       {"emergency": _EMERGENCY, "special_group": _SPECIAL, "prescription": _PRESCRIPTION}.items()}

MESSAGES = {
    "emergency": (
        "This may be an emergency. Please seek urgent medical care now: call 912 (SAMU ambulance, Rwanda) "
        "or go to the nearest health facility. Do not wait for an online answer."
    ),
    "special_group": (
        "Special care needed: pregnancy, breastfeeding, children, older adults and people with long-term "
        "conditions often need different doses or different medicines. Confirm with a doctor or pharmacist "
        "before taking anything."
    ),
    "prescription": (
        "This is a prescription-only medicine. It should only be used after a clinician has assessed you "
        "and given a prescription; taking antibiotics without one contributes to antimicrobial resistance."
    ),
}
DISCLAIMER = (
    "General information only, not a diagnosis. If symptoms are severe, last more than a few days or get "
    "worse, see a doctor or pharmacist."
)

# US label text (openFDA) points to US-only services; answers are shown to patients in Rwanda.
_US_HOTLINE = re.compile(r"\(?\b(?:1[-.\s])?\(?800\)?[-.\s]?222[-.\s]?1222\b\)?")  # US Poison Control only
_US_POISON_CENTER = re.compile(r"\b(?:a\s+|the\s+)?Poison Control Center\b", re.IGNORECASE)


def localize(answer: str) -> str:
    answer = _US_HOTLINE.sub("912 (SAMU, Rwanda)", answer)
    return _US_POISON_CENTER.sub("emergency services", answer)


@dataclass
class RiskAssessment:
    flags: list[str] = field(default_factory=list)
    matches: dict[str, str] = field(default_factory=dict)

    @property
    def level(self) -> str:
        for lvl in ("emergency", "special_group", "prescription"):
            if lvl in self.flags:
                return lvl
        return "routine"

    def warnings(self) -> list[str]:
        return [MESSAGES[f] for f in ("emergency", "special_group", "prescription") if f in self.flags]


def assess(text: str) -> RiskAssessment:
    ra = RiskAssessment()
    for name, rx in _RE.items():
        m = rx.search(text)
        if m:
            ra.flags.append(name)
            ra.matches[name] = m.group(0)
    return ra
