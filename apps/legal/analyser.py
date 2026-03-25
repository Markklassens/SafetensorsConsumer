"""
SRHN Assembly — Legal Document Analyser
=========================================
Real-world application: Contract review, clause extraction, compliance checking.

What it does:
  - Analyses contract clauses for risk flags
  - Checks against loaded regulation knowledge (GDPR, employment law, etc.)
  - Learns from lawyer corrections
  - Stores precedent decisions in memory
  - Classifies clause risk: HIGH / MEDIUM / LOW / ACCEPTABLE

Usage:
  from apps.legal import LegalAnalyser
  app = LegalAnalyser(engine)
  result = app.analyse_clause("The Company may terminate employment without notice.")
  print(result.risk_level)   # HIGH
  print(result.issues)       # ['No notice period violates ERA 1996']

  # Analyse a full contract (list of clauses)
  report = app.analyse_contract(clauses, contract_name="Supply Agreement")
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.engine import SRHNEngine


HIGH       = "HIGH"
MEDIUM     = "MEDIUM"
LOW        = "LOW"
ACCEPTABLE = "ACCEPTABLE"


@dataclass
class ClauseResult:
    clause:      str
    risk_level:  str
    issues:      list[str]
    suggestions: list[str]
    explanation:   str
    confidence:    float
    query_id:      int
    elapsed_ms:    float
    memories_used: int = 0

    def display(self) -> str:
        icons = {HIGH:"🔴", MEDIUM:"🟡", LOW:"🟢", ACCEPTABLE:"✅"}
        icon  = icons.get(self.risk_level, "⚪")
        lines = [
            f"  {icon}  {self.risk_level} RISK  (conf {self.confidence:.0%})",
            f"  Clause: {self.clause[:120]}...",
        ]
        if self.issues:
            lines.append("  Issues:")
            for i in self.issues: lines.append(f"    • {i}")
        if self.suggestions:
            lines.append("  Suggestions:")
            for s in self.suggestions: lines.append(f"    → {s}")
        lines.append(f"  {self.explanation}")
        return "\n".join(lines)


@dataclass
class ContractReport:
    contract_name:  str
    clauses:        list[ClauseResult]
    high_count:     int = 0
    medium_count:   int = 0
    low_count:      int = 0
    acceptable_count: int = 0
    overall_risk:   str = ACCEPTABLE
    summary:        str = ""

    def display(self) -> str:
        lines = [
            f"\n  Contract: {self.contract_name}",
            f"  Overall risk: {self.overall_risk}",
            f"  Clauses: {len(self.clauses)} analysed — "
            f"🔴 {self.high_count} HIGH  🟡 {self.medium_count} MEDIUM  "
            f"🟢 {self.low_count} LOW  ✅ {self.acceptable_count} OK",
            f"\n  {self.summary}",
            f"\n  {'─'*60}",
        ]
        for i, c in enumerate(self.clauses):
            lines.append(f"\n  Clause {i+1}:")
            lines.append(c.display())
        return "\n".join(lines)


class LegalAnalyser:
    """Contract review and compliance checking backed by SRHN memory."""

    _SYSTEM = """You are a legal risk analyst reviewing contract clauses and documents.
You have access to legal precedents, regulations, and prior decisions from memory.

For each clause, analyse:
1. Risk level: HIGH / MEDIUM / LOW / ACCEPTABLE
2. Specific legal issues (cite relevant law where possible)
3. Suggested amendments to reduce risk
4. Brief explanation

Format response EXACTLY as:
RISK: [level]
ISSUES: [semicolon-separated issues, or "none"]
SUGGESTIONS: [semicolon-separated amendments, or "none"]
EXPLANATION: [2-3 sentences]

Consider: employment law, GDPR, contract formation, liability, IP, termination rights."""

    def __init__(self, engine: SRHNEngine):
        self.engine = engine
        self._seed_legal_knowledge()

    def _seed_legal_knowledge(self):
        regulations = [
            # Employment
            ("UK Employment Rights Act 1996: employees with 2+ years service have right to written statement of reasons for dismissal. Instant dismissal without notice requires gross misconduct.", "legal"),
            ("ACAS Code of Practice: disciplinary and grievance procedures must be followed before dismissal. Failure to follow = uplift of up to 25% on tribunal awards.", "legal"),
            ("National Minimum Wage Act 1998: employers must pay NMW for all working time including travel between assignments for mobile workers.", "legal"),
            # GDPR / Data
            ("GDPR Article 13: data subjects must be informed at collection point about: data controller identity, purposes, legal basis, retention period, data subject rights.", "legal"),
            ("GDPR Article 17 Right to Erasure: must be fulfilled within 1 month. Erasure is not required where data is necessary for legal claims or public interest.", "legal"),
            ("GDPR Article 28: data processing agreements with processors are mandatory. Must include subject matter, duration, nature, purpose, type of data, obligations.", "legal"),
            # Contract
            ("Unfair Contract Terms Act 1977: exclusion clauses for death/personal injury caused by negligence are void. Other exclusion clauses subject to reasonableness test.", "legal"),
            ("Consumer Rights Act 2015: terms must be transparent and prominent. Unfair terms not binding on consumers. Core terms exempt if transparent.", "legal"),
            # IP
            ("Copyright Designs and Patents Act 1988: employer owns copyright in works created by employees in course of employment. Contractor IP remains with contractor unless assigned.", "legal"),
            # Payments
            ("Late Payment of Commercial Debts Act 1998: statutory interest of 8% above base rate on late B2B payments. Debt recovery costs also claimable.", "legal"),
        ]
        for fact, domain in regulations:
            self.engine.learn_fact(fact, domain=domain, confidence=0.97, source="UK Law")

        # Add standard workflows
        self.engine.learn_workflow(
            "Contract risk review checklist",
            steps=[
                "Check parties: full legal names, registered addresses, capacity to contract",
                "Verify consideration: is there valid, sufficient consideration on both sides?",
                "Review termination: notice periods, grounds for immediate termination, survival clauses",
                "Check liability caps: are they reasonable? Is personal injury excluded?",
                "Review IP ownership: who owns work product? Are assignments explicit?",
                "Check governing law and jurisdiction: is this enforceable?",
                "Review data processing: GDPR compliance, DPA if applicable",
                "Check non-compete / restraint of trade: geographic scope, duration (18mo max typical)",
            ],
            domain="legal")

    def analyse_clause(self, clause: str,
                       jurisdiction: str = "UK") -> ClauseResult:
        """Analyse a single contract clause for legal risk."""
        t0    = time.perf_counter()
        query = (f"Jurisdiction: {jurisdiction}. "
                 f"Analyse this contract clause for legal risk:\n\n\"{clause}\"")
        result   = self.engine.query(query, top_k=8)
        response = result.get("response", "")

        risk_level   = self._parse(response, "RISK",        MEDIUM)
        issues_raw   = self._parse(response, "ISSUES",      "")
        suggest_raw  = self._parse(response, "SUGGESTIONS", "")
        explanation  = self._parse(response, "EXPLANATION", response[:300])

        issues      = [i.strip() for i in issues_raw.split(";") if i.strip() and i.strip().lower() != "none"]
        suggestions = [s.strip() for s in suggest_raw.split(";") if s.strip() and s.strip().lower() != "none"]

        return ClauseResult(
            clause=clause,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            explanation=explanation,
            confidence=result.get("confidence", 0.5),
            query_id=result.get("query_id", 0),
            elapsed_ms=round((time.perf_counter()-t0)*1000, 1),
            memories_used=result.get("memories_used", 0),
        )

    def analyse_contract(self, clauses: list[str],
                         contract_name: str = "Contract",
                         jurisdiction: str = "UK") -> ContractReport:
        """Analyse a full contract (list of clauses)."""
        results  = [self.analyse_clause(c, jurisdiction) for c in clauses]
        counts   = {HIGH:0, MEDIUM:0, LOW:0, ACCEPTABLE:0}
        for r in results:
            counts[r.risk_level] = counts.get(r.risk_level, 0) + 1

        overall = (HIGH if counts[HIGH] > 0 else
                   MEDIUM if counts[MEDIUM] > 1 else
                   LOW if counts[LOW] > 0 else ACCEPTABLE)

        high_issues = [r.issues[0] for r in results
                       if r.risk_level == HIGH and r.issues]
        summary = (f"{counts[HIGH]} high-risk clauses require immediate attention. "
                   f"Key issues: {'; '.join(high_issues[:3])}."
                   if counts[HIGH] > 0 else
                   f"No high-risk clauses found. {counts[MEDIUM]} medium-risk items to review.")

        return ContractReport(
            contract_name=contract_name,
            clauses=results,
            high_count=counts[HIGH],
            medium_count=counts[MEDIUM],
            low_count=counts[LOW],
            acceptable_count=counts[ACCEPTABLE],
            overall_risk=overall,
            summary=summary,
        )

    def _parse(self, text: str, field: str, default: str) -> str:
        for line in text.splitlines():
            if line.upper().startswith(field + ":"):
                return line[len(field)+1:].strip()
        return default

    def correct(self, query_id: int, actual_risk: str, reason: str = ""):
        self.engine.feedback(query_id, reward=-1.0,
                             note=f"Actual risk: {actual_risk}. {reason}")

    def confirm(self, query_id: int):
        self.engine.feedback(query_id, reward=+1.0)

    def add_precedent(self, case: str, decision: str, source: str = ""):
        self.engine.learn_fact(f"PRECEDENT — {case}: {decision}",
                               domain="legal", confidence=0.95, source=source)

    def add_regulation(self, regulation: str, jurisdiction: str = "UK"):
        self.engine.learn_fact(regulation, domain="legal",
                               confidence=0.97, source=jurisdiction)
