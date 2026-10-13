"""Light query understanding in front of the retriever.

Patients rarely use the generic names that drug labels are indexed under ("panadol", "after sex
pill"), greet the assistant before asking anything, and usually want one aspect of one medicine
("side effects of metformin"). These rules handle those cases so the retriever and reader see the
medicine and section the question is actually about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SMALL_TALK = [
    (re.compile(r"^(hi+|hello+|hey+|good (morning|afternoon|evening)|greetings|muraho|mwaramutse|mwiriwe|bonjour)\b[\s!.,]*$", re.I),
     "Hello! I'm the FARUMASI pharmacy assistant. Ask me about a medicine (what it is used for, dosage, side effects, "
     "interactions, storage) or a minor health problem such as a headache, a small wound or diarrhoea. "
     "For example: \"What are the side effects of amoxicillin?\""),
    (re.compile(r"^(ok(ay)?[\s,]*)?(thanks?( you)?( so much| a lot)?|thank u|thx|murakoze|merci)\b[\s!.,]*$", re.I),
     "You're welcome! Feel free to ask another question about a medicine or a health problem."),
    (re.compile(r"^(bye+|goodbye|see you|ok(ay)? bye)\b[\s!.,]*$", re.I),
     "Take care! Remember that a pharmacist or doctor can always confirm what is right for you."),
    (re.compile(r"^(who are you|what are you|what can you do|help)\b[\s?!.]*$", re.I),
     "I'm a pharmacy assistant built on trusted medicine information (Rwanda FDA patient leaflets, US FDA labels and "
     "NIH MedlinePlus). I can explain what a medicine is for, how to take it, its side effects and interactions, and "
     "give first advice for minor health problems. I don't replace a pharmacist or doctor."),
]

# Everyday and brand names used in Rwandan pharmacies -> the names the corpus is indexed under.
LAY_TERMS = [
    (r"after[- ]?sex pills?|morning[- ]after pills?|emergency (contracepti\w+|pills?)|postinor|plan b|\bp2\b|\be-?pill",
     "levonorgestrel emergency contraception"),
    (r"family planning|birth control pills?|contraceptive pills?|the pill\b", "contraception birth control"),
    (r"(before|during) sex\b.*(exp\w*(?:ence|ance)\b|perform|last longer|stamina|enjoy|strong|hard)|sex(ual)? (enhancement|performance|power|stamina)"
     r"|erect\w*|impoten\w*|last longer in bed|viagra",
     "erectile dysfunction sexual problems in men"),
    (r"panadol|tylenol|acetaminophen", "paracetamol"),
    (r"brufen|advil|nurofen", "ibuprofen"),
    (r"flagyl", "metronidazole"),
    (r"augmentin", "amoxicillin and clavulanic acid"),
    (r"amoxil", "amoxicillin"),
    (r"coartem|artemether|lumefantrine", "artemether and lumefantrine coartem"),
    (r"septrin|bactrim|co-?trimoxazole", "sulfamethoxazole and trimethoprim"),
    (r"glucophage", "metformin"),
    (r"voltaren", "diclofenac"),
    (r"ventolin", "salbutamol"),
    (r"piriton", "chlorphenamine"),
    (r"imodium", "loperamide"),
    (r"zyrtec", "cetirizine"),
    (r"valium", "diazepam"),
    (r"\bors\b|oral rehydration", "oral rehydration salts diarrhoea"),
    (r"de-?worm\w*", "albendazole mebendazole worms"),
]
_LAY = [(re.compile(p, re.I), rep) for p, rep in LAY_TERMS]

# Question wording -> section names used in the corpus titles ("Metformin: Side effects").
ASPECTS = [
    (r"side[- ]?effects?|adverse|reactions?", {"Side effects"}),
    (r"\bdos(e|es|age|ing)\b|how (much|many|often)|how to (take|use)|how (should|do|can|must) (i|you|we) (take|use)"
     r"|when (should|do) (i|you) take|\bper day\b|a day\b", {"Dosage", "How to take"}),
    (r"used for|\buses?\b|what is .* for|treat|indicat|know (more )?about|tell me about|information (on|about)|how does .* work",
     {"Uses", "Purpose"}),
    (r"pregnan|breast[- ]?feed", {"Pregnancy", "Pregnancy and breastfeeding"}),
    (r"interact|together with|\bwith (alcohol|other)|mix", {"Interactions"}),
    (r"stor(e|age)|keep|fridge", {"Storage"}),
    (r"overdose|too (much|many)|swallowed", {"Overdose"}),
    (r"food|empty stomach|meal|before (i|you) take|should not take|who can", {"Before you take", "How to take", "Dosage"}),
]
_ASPECTS = [(re.compile(p, re.I), names) for p, names in ASPECTS]

_NOT_DRUG_NAMES = {"compound", "candid", "insulin", "salt", "spacer", "troop", "biologic"}


def small_talk(text: str) -> str | None:
    t = text.strip()
    if len(t) > 40:
        return None
    for pattern, reply in SMALL_TALK:
        if pattern.match(t):
            return reply
    return None


_FOLLOW_UP = re.compile(
    r"\b(those|these|they|them|it|its|this|that|their|same|more|examples?|other ones?|which ones?|what about|and if)\b", re.I)
MAX_CONTEXT_WORDS = 30


def resolve_follow_up(question: str, previous: str | None, has_own_topic) -> str:
    """Attach the previous question's subject to a follow-up such as "what do they do?".

    Only short questions with a pronoun / "more" / "examples" that name no medicine or known lay term
    of their own are treated as follow-ups; anything else starts a new topic.
    """
    if not previous or len(question.split()) > 15 or not _FOLLOW_UP.search(question) or has_own_topic(question):
        return question
    context = " ".join(previous.split()[-MAX_CONTEXT_WORDS:])
    return f"{question}{FOLLOW_UP_MARK}{context})"


FOLLOW_UP_MARK = " (about: "


def _sections(text: str) -> set[str]:
    return set().union(*(names for rx, names in _ASPECTS if rx.search(text)))


def expand(text: str) -> str:
    extra = [rep for pattern, rep in _LAY if pattern.search(text)]
    return f"{text} ({'; '.join(extra)})" if extra else text


def drug_of(title: str, source: str = "") -> str:
    """Medicine a passage is about: "Metformin: Side effects" -> metformin; MedicationQA titles are the medicine."""
    if ": " in title:
        return title.split(": ", 1)[0].lower()
    return title.lower() if source == "MedicationQA" else ""


def section_of(title: str) -> str:
    return title.split(": ", 1)[1] if ": " in title else ""


@dataclass
class Focus:
    drugs: set[str] = field(default_factory=set)
    sections: set[str] = field(default_factory=set)


class QueryAnalyzer:
    def __init__(self, drug_names):
        names = set(drug_names) - {""} - _NOT_DRUG_NAMES
        names = sorted((n for n in names if len(n) >= 4), key=len, reverse=True)
        self._drug_re = [(n, re.compile(rf"\b{re.escape(n)}\b")) for n in names]

    def focus(self, expanded: str) -> Focus:
        text = expanded.lower()
        drugs = {n for n, rx in self._drug_re if rx.search(text)}
        # In a follow-up ("how should I take it? (about: side effects of metformin)") the section the
        # user asks about now wins over the one asked about before.
        sections = _sections(expanded.split(FOLLOW_UP_MARK)[0]) or _sections(expanded)
        return Focus(drugs, sections)

    def has_own_topic(self, text: str) -> bool:
        expanded = expand(text)
        return expanded != text or bool(self.focus(expanded).drugs)


_CROSS_REF = re.compile(
    r"\[\s*see [^\]]*\]"                                  # [see Warnings and Precautions (5.1)]
    r"|(?-i:\[\s*see\s+(?:(?:(?!For\b)[A-Z][a-z]*|and)\s+)+)"  # unclosed "[ see Warnings and Precautions "
    r"|\(\s*\d+(\.\d+)*\s*\)|\b\d+\.\d+\s+(?=[A-Z])"      # (2.1), "8.1 Pregnancy"
    r"|To report SUSPECTED ADVERSE REACTIONS.*?(?:www\.\S+|$)",  # US manufacturer / MedWatch boilerplate
    re.I,
)
_HEADING_LEFTOVER = re.compile(r"^[&\s]*(?:[A-Z&]{2,}\s+)+(?=[A-Z][a-z])")


def clean_passage(full: str) -> str:
    """Drop label cross-references ("[see Warnings and Precautions (5.1)]", "8.1 ") and heading leftovers
    ("& USAGE") that the reader tends to copy. `full` is "Title. passage text"."""
    title, sep, body = full.partition(". ")
    if not sep:
        title, body = "", full
    body = _HEADING_LEFTOVER.sub("", _CROSS_REF.sub(" ", body).strip())
    return re.sub(r"\s{2,}", " ", f"{title}{sep}{body}").strip()
