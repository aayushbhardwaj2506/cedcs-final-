"""What a hospital's NAME says about whether it can take an emergency (used for real hospitals, whose capabilities are unknown).

A deliberately small, conservative rule set: obvious non-emergency facilities (eye, dental, skin, ayurveda, clinics...) are
excluded from every emergency, and specialty hospitals (children's, maternity, cardiac...) are only kept when the case is in
their field. This is an inference from a name, not a fact about the hospital, and it is always reported as such. It can be
wrong (a big hospital could have "eye" in its name), which is why the reason names the word that triggered it.
"""

from __future__ import annotations

import re
from typing import Optional

# never an emergency destination for the cases this system handles
_EXCLUDE = {
    r"\beye\b|ophthal|netralaya|nethralaya": "eye hospital",
    r"\bdental\b|dentist|\bteeth\b": "dental clinic",
    r"\bskin\b|derma|cosmetic|\bhair\b|aesthetic": "skin/cosmetic clinic",
    r"\bent\b|ear nose|otolaryn": "ENT clinic",
    r"ayurved|homeo|siddha|unani|naturopath|yoga|acupunct": "alternative-medicine facility",
    r"physio|rehab|fertility|\bivf\b|infertil|test tube": "rehab/fertility facility",
    r"\bclinic\b|polyclinic|dispensary|health ?centre|health center|\bphc\b": "clinic (not an emergency department)",
    r"diagnostic|scan cent|imaging cent|laborator|\blab\b|blood bank": "diagnostic/lab facility",
    r"veterinar|animal|\bpet\b": "veterinary",
    r"cancer|oncolog|radiother": "oncology centre",
    r"de-?addiction|deaddiction": "de-addiction centre",
}
# specialty hospitals: fine when the case is in their field (triage category), otherwise not
_SPECIALTY = {
    r"child|paediatric|pediatric|\bkids?\b|neonat|baby|\bnicu\b": ("children's hospital", {"PAEDIATRIC"}),
    r"matern|women|gynae|gynec|obstet|birth|\bmother": ("maternity/women's hospital", {"OBSTETRIC"}),
    r"cardiac|cardio|\bheart\b": ("cardiac hospital", {"CARDIAC"}),
    r"neuro(?!\s*psych)|brain|spine": ("neuro hospital", {"NEUROLOGICAL"}),
    r"ortho|bone|joint|trauma|accident": ("orthopaedic/trauma hospital", {"TRAUMA"}),
    r"psychiatr|mental|\bmind\b|neuro ?psych": ("psychiatric hospital", {"PSYCHIATRIC"}),
    r"kidney|renal|dialysis|nephro|urolog": ("kidney/urology hospital", {"RENAL"}),
    r"burn": ("burns hospital", {"BURNS"}),
}


def classify(name: str) -> Optional[dict]:
    """None for an ordinary/unknown hospital; otherwise {kind: 'excluded'|'specialty', label, word, categories}."""
    low = f" {name.lower()} "
    for pat, label in _EXCLUDE.items():
        m = re.search(pat, low)
        if m:
            return {"kind": "excluded", "label": label, "word": m.group(0).strip(), "categories": set()}
    for pat, (label, cats) in _SPECIALTY.items():
        m = re.search(pat, low)
        if m:
            return {"kind": "specialty", "label": label, "word": m.group(0).strip(), "categories": set(cats)}
    return None


def rejection_reason(name: str, case_categories: set) -> Optional[str]:
    """Why this hospital should not be recommended for this case, judging by its name only; None if it may stay."""
    hint = classify(name)
    if hint is None:
        return None
    if hint["kind"] == "excluded":
        return f"looks like a {hint['label']} (\"{hint['word']}\" in its name): not an emergency destination"
    if hint["categories"] & set(case_categories):
        return None
    return f"looks like a {hint['label']} (\"{hint['word']}\" in its name), but this case is not in that field"
