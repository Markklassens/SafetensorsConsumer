"""
SRHN Assembly — Sports Analytics Assistant
============================================
Real-world application: Match analysis, player stats QA, betting/fantasy insights.

What it does:
  - Answers questions about loaded team/player/match data
  - Stores season statistics in SRHN memory
  - Analyses form, head-to-head, injury impact
  - Learns from user corrections (e.g. wrong team named)
  - Supports: football, cricket, basketball, tennis

Usage:
  from apps.sports import SportsAnalyst
  analyst = SportsAnalyst(engine)
  analyst.load_season("Premier League 2024-25", teams_data)
  result = analyst.query("Who is the top scorer in the Premier League?")
  result = analyst.analyse_match("Arsenal vs Man City", "2025-04-12")
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.engine import SRHNEngine


@dataclass
class AnalysisResult:
    query:        str
    answer:       str
    confidence:   float
    query_id:     int
    elapsed_ms:   float
    memories_used: int

    def display(self) -> str:
        return (f"\n  Q: {self.query}\n"
                f"  A: {self.answer}\n"
                f"  [conf={self.confidence:.0%} | mem={self.memories_used} | {self.elapsed_ms:.0f}ms]")


class SportsAnalyst:
    """Sports analytics assistant backed by SRHN memory."""

    _SYSTEM = """You are a sports analyst with deep knowledge of football, cricket, basketball, and tennis.
You have access to season statistics, match results, and player data from memory.
Answer questions accurately and concisely. Cite statistics when available.
If data is not in memory, say so clearly rather than guessing."""

    def __init__(self, engine: SRHNEngine):
        self.engine = engine
        self._seed_sports_knowledge()

    def _seed_sports_knowledge(self):
        facts = [
            # Premier League
            ("Premier League 2024-25: 20 clubs, 380 matches per season. Season runs Aug-May.", "sports"),
            ("Premier League promotion/relegation: bottom 3 relegated to Championship.", "sports"),
            ("Champions League qualification: top 4 Premier League clubs qualify automatically.", "sports"),
            ("Offside rule: a player is offside if any part of their body they can score with is closer to the goal line than both the ball and the second-last defender.", "sports"),
            # Cricket
            ("Test cricket: 5 days, 2 innings per team, played in whites. T20: 20 overs per side, ~3 hours.", "sports"),
            ("Duckworth-Lewis-Stern (DLS) method used to set revised targets in rain-affected limited overs matches.", "sports"),
            # Basketball
            ("NBA: 82 regular season games, 30 teams, 5 on court per side. Shot clock: 24 seconds.", "sports"),
            ("Triple-double: double figures in 3 stats (points, rebounds, assists) in one game.", "sports"),
            # Tennis
            ("Grand Slams: Australian Open (Jan), French Open (May-Jun), Wimbledon (Jul), US Open (Aug-Sep).", "sports"),
            ("ATP ranking points: 2000 for Grand Slam win, 1000 for Masters 1000, 500 for ATP 500.", "sports"),
        ]
        for fact, domain in facts:
            self.engine.learn_fact(fact, domain=domain, confidence=0.95,
                                   source="Sports Knowledge Base")

    def load_season(self, season_name: str, data: list[str]):
        """
        Load season statistics into SRHN memory.
        data: list of fact strings e.g. ["Arsenal: 28W 6D 4L, 91pts", ...]
        """
        for item in data:
            self.engine.learn_fact(
                f"{season_name}: {item}", domain="sports",
                confidence=0.97, source=season_name)

    def load_player_stats(self, player: str, stats: dict):
        """Load individual player statistics."""
        stats_str = ", ".join(f"{k}: {v}" for k, v in stats.items())
        self.engine.learn_fact(
            f"{player} stats: {stats_str}",
            domain="sports", confidence=0.97)

    def load_match_result(self, match: str, result: str,
                          scorers: list[str] = None):
        """Store a match result."""
        fact = f"Match result: {match} — {result}"
        if scorers:
            fact += f". Scorers: {', '.join(scorers)}"
        self.engine.learn_fact(fact, domain="sports", confidence=0.99)

    def query(self, question: str) -> AnalysisResult:
        """Answer any sports question using loaded data."""
        t0     = time.perf_counter()
        result = self.engine.query(question, top_k=8)
        return AnalysisResult(
            query=question,
            answer=result.get("response", ""),
            confidence=result.get("confidence", 0.5),
            query_id=result.get("query_id", 0),
            elapsed_ms=round((time.perf_counter()-t0)*1000, 1),
            memories_used=result.get("memories_used", 0),
        )

    def analyse_match(self, teams: str, date: str = "") -> AnalysisResult:
        """Analyse an upcoming or recent match."""
        q = f"Analyse the match: {teams}"
        if date: q += f" on {date}"
        q += ". Consider recent form, head-to-head record, key players, and likely outcome."
        return self.query(q)

    def correct(self, query_id: int, correct_answer: str):
        self.engine.feedback(query_id, reward=-1.0, note=f"Correct: {correct_answer}")

    def confirm(self, query_id: int):
        self.engine.feedback(query_id, reward=+1.0)
