"""
SRHN Assembly — Medical Triage Assistant
=========================================
Real-world application: Emergency department triage support.

What it does:
  - Accepts patient symptoms as free text
  - Returns triage category (immediate/urgent/non-urgent/refer)
  - Suggests relevant investigations and red flags
  - Learns from clinician corrections via feedback
  - Stores approved protocols in persistent memory
  - Tracks failure patterns (what it got wrong) to improve

NOT a diagnostic tool. NOT a replacement for clinical judgment.
Designed as a decision SUPPORT layer for trained clinicians.

Usage:
  from apps.medical import MedicalAssistant
  app = MedicalAssistant(engine)
  result = app.triage("65yo male, chest pain radiating to jaw, diaphoresis, onset 20min")
  print(result.category)      # IMMEDIATE
  print(result.red_flags)     # ['ACS', 'STEMI']
  print(result.investigations) # ['12-lead ECG', 'Troponin', 'CXR']
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.engine import SRHNEngine


# ── Triage categories ────────────────────────────────────────────────────────
IMMEDIATE   = "IMMEDIATE"    # life-threatening, act now
URGENT      = "URGENT"       # serious, act within 30min
SEMI_URGENT = "SEMI-URGENT"  # moderate, act within 2hr
NON_URGENT  = "NON-URGENT"   # minor, act within 4hr
REFER       = "REFER"        # not ED, refer to GP/specialist

# ── Red-flag keywords ────────────────────────────────────────────────────────
_RED_FLAGS = {
    IMMEDIATE: [
        "chest pain", "cardiac arrest", "unresponsive", "not breathing",
        "severe bleeding", "anaphylaxis", "stroke", "severe head injury",
        "airway obstruction", "septic shock", "eclampsia", "major trauma",
        "unconscious", "seizing", "diaphoresis", "jaw pain", "stemi",
        "aortic", "dissection",
    ],
    URGENT: [
        "difficulty breathing", "shortness of breath", "altered consciousness",
        "severe abdominal pain", "high fever", "fracture", "severe pain",
        "vomiting blood", "rectal bleeding", "acute confusion",
        "suspected dvt", "pulmonary embolism",
    ],
    SEMI_URGENT: [
        "moderate pain", "urinary symptoms", "ear pain", "mild fever",
        "rash", "sutures", "sprain", "wound",
    ],
}


@dataclass
class TriageResult:
    symptoms:       str
    category:       str
    urgency_score:  int          # 1=immediate 5=non-urgent
    red_flags:      list[str]
    investigations: list[str]
    advice:         str
    confidence:     float
    query_id:       int
    elapsed_ms:     float
    memories_used:  int

    def display(self) -> str:
        urgency_icons = {IMMEDIATE:"🔴", URGENT:"🟡", SEMI_URGENT:"🟢", NON_URGENT:"⚪", REFER:"🔵"}
        icon = urgency_icons.get(self.category, "⚪")
        lines = [
            f"\n  {icon}  TRIAGE: {self.category}  (confidence {self.confidence:.0%})",
            f"  {'─'*50}",
            f"  Symptoms: {self.symptoms[:100]}",
        ]
        if self.red_flags:
            lines.append(f"  Red flags: {', '.join(self.red_flags)}")
        if self.investigations:
            lines.append(f"  Investigations: {', '.join(self.investigations)}")
        lines.append(f"\n  Assessment:\n  {self.advice}")
        lines.append(f"\n  [mem={self.memories_used} | {self.elapsed_ms:.0f}ms | qid={self.query_id}]")
        return "\n".join(lines)


class MedicalAssistant:
    """
    Emergency triage decision support backed by SRHN memory.
    Learns from every correction. Remembers protocols permanently.
    """

    _SYSTEM = """You are a clinical decision support system for emergency triage.
You have access to relevant clinical protocols and past cases from memory.

For each patient presentation:
1. Assign triage category: IMMEDIATE / URGENT / SEMI-URGENT / NON-URGENT / REFER
2. List specific red flags present
3. Recommend immediate investigations (ECG, bloods, imaging)
4. Give brief clinical rationale (2-3 sentences)

Format response as:
TRIAGE: [CATEGORY]
RED FLAGS: [comma-separated list or "none"]
INVESTIGATIONS: [comma-separated list]
RATIONALE: [2-3 sentences]

Base decisions on: ABCDE approach, NEWS2 scoring principles, NICE guidelines.
If uncertain, escalate — when in doubt, triage up."""

    def __init__(self, engine: SRHNEngine):
        self.engine = engine
        self._seed_protocols()

    def _seed_protocols(self):
        """Inject core clinical protocols into SRHN memory on first run."""
        facts = [
            # ACS
            ("Chest pain with radiation to jaw/arm, diaphoresis, and ST elevation = STEMI — activate cath lab immediately. Investigations: 12-lead ECG, Troponin, CXR, FBC, U&E.", "medical"),
            ("HEART score ≥5 = high risk ACS. Admit for observation and serial troponins.", "medical"),
            # Sepsis
            ("Sepsis-6 bundle: blood cultures, IV antibiotics, IV fluids, urine output, lactate, O2. Must be completed within 1 hour of recognition.", "medical"),
            ("qSOFA criteria: RR≥22, altered mentation, SBP≤100. ≥2 = high risk sepsis — escalate immediately.", "medical"),
            # Stroke
            ("FAST: Face drooping, Arm weakness, Speech difficulty, Time to call. Stroke = immediate — CT head, thrombolysis window is 4.5 hours.", "medical"),
            ("Posterior circulation stroke: vertigo, diplopia, ataxia, dysphagia. High NIHSS = transfer to stroke unit.", "medical"),
            # Respiratory
            ("Severe asthma: PEFR <33% predicted, SpO2 <92%, silent chest, cyanosis, bradycardia = life-threatening. Immediate nebulisation, IV magnesium, HDU.", "medical"),
            ("Tension pneumothorax: tracheal deviation, absent breath sounds, hypotension. Emergency needle decompression — do not wait for CXR.", "medical"),
            # Paediatric
            ("Meningococcal septicaemia: non-blanching petechial rash, fever, photophobia = immediate. IV ceftriaxone, admission.", "medical"),
            # Triage
            ("NEWS2 score ≥7 = immediate emergency response. Score 5-6 = urgent review within 30 minutes.", "medical"),
            ("Anaphylaxis triad: urticaria, bronchospasm, hypotension. Treatment: adrenaline 0.5mg IM, IV fluids, antihistamine, steroid.", "medical"),
        ]
        for fact, domain in facts:
            self.engine.learn_fact(fact, domain=domain, confidence=0.99,
                                   source="NICE/JRCALC/RCEM Guidelines")

    def triage(self, symptoms: str, age: int = 0,
               vitals: dict = None) -> TriageResult:
        """
        Triage a patient presentation.

        symptoms: Free-text description of the presenting complaint
        age:      Patient age (0 = unknown)
        vitals:   Dict with keys: hr, bp_sys, bp_dia, rr, spo2, temp, gcs
        """
        t0 = time.perf_counter()

        # Build structured query
        vitals_str = ""
        if vitals:
            parts = []
            if vitals.get("hr"):    parts.append(f"HR {vitals['hr']}")
            if vitals.get("bp_sys"):parts.append(f"BP {vitals['bp_sys']}/{vitals.get('bp_dia','?')}")
            if vitals.get("rr"):    parts.append(f"RR {vitals['rr']}")
            if vitals.get("spo2"):  parts.append(f"SpO2 {vitals['spo2']}%")
            if vitals.get("temp"):  parts.append(f"Temp {vitals['temp']}°C")
            if vitals.get("gcs"):   parts.append(f"GCS {vitals['gcs']}")
            vitals_str = "  Vitals: " + ", ".join(parts) + "\n" if parts else ""

        age_str = f"Age: {age}yo. " if age > 0 else ""
        query = f"{age_str}Presenting complaint: {symptoms}\n{vitals_str}Triage this patient."

        result = self.engine.query(query, top_k=8)
        response = result.get("response", "")

        # Parse structured response
        category      = self._parse_field(response, "TRIAGE",         URGENT)
        red_flags_raw = self._parse_field(response, "RED FLAGS",       "")
        investig_raw  = self._parse_field(response, "INVESTIGATIONS",  "")
        rationale     = self._parse_field(response, "RATIONALE",       response[:300])

        red_flags    = [f.strip() for f in red_flags_raw.split(",") if f.strip() and f.strip().lower() != "none"]
        investigations = [i.strip() for i in investig_raw.split(",") if i.strip()]

        # Validate: keyword-based safety override
        category = self._safety_override(symptoms, category)

        urgency_map = {IMMEDIATE:1, URGENT:2, SEMI_URGENT:3, NON_URGENT:4, REFER:5}
        score = urgency_map.get(category, 3)

        return TriageResult(
            symptoms=symptoms,
            category=category,
            urgency_score=score,
            red_flags=red_flags,
            investigations=investigations,
            advice=rationale,
            confidence=result.get("confidence", 0.5),
            query_id=result.get("query_id", 0),
            elapsed_ms=round((time.perf_counter()-t0)*1000, 1),
            memories_used=result.get("memories_used", 0),
        )

    def _parse_field(self, text: str, field: str, default: str) -> str:
        for line in text.splitlines():
            if line.upper().startswith(field + ":"):
                return line[len(field)+1:].strip()
        return default

    def _safety_override(self, symptoms: str, category: str) -> str:
        """Never downgrade a presentation that contains immediate red flags."""
        s = symptoms.lower()
        for flag in _RED_FLAGS[IMMEDIATE]:
            if flag in s:
                return IMMEDIATE
        if category == NON_URGENT:
            for flag in _RED_FLAGS[URGENT]:
                if flag in s:
                    return URGENT
        return category

    def correct(self, query_id: int, actual_category: str,
                reason: str = ""):
        """
        Clinician correction: the triage was wrong.
        Sends negative feedback, stores failure pattern.
        """
        self.engine.feedback(query_id, reward=-1.0,
                             note=f"Correct category was {actual_category}. {reason}")

    def confirm(self, query_id: int, note: str = ""):
        """Clinician confirms triage was correct. Reinforces decision."""
        self.engine.feedback(query_id, reward=+1.0, note=note)

    def add_protocol(self, protocol: str, source: str = ""):
        """Add a clinical guideline or local protocol to permanent memory."""
        self.engine.learn_fact(protocol, domain="medical",
                               confidence=0.98, source=source)

    def add_local_pathway(self, name: str, steps: list[str]):
        """Add a local care pathway (e.g. hospital-specific sepsis pathway)."""
        self.engine.learn_workflow(name, steps, domain="medical")
