"""
SRHN Assembly — DevOps Runbook Agent
======================================
Real-world application: Incident response, on-call support, runbook execution.

What it does:
  - Diagnoses infrastructure alerts using memory of past incidents
  - Suggests runbook steps based on alert type
  - Learns which fixes actually worked (feedback from SRE)
  - Tracks failure patterns (what made things worse)
  - Accumulates institutional knowledge from post-mortems
  - Severity scoring: P1/P2/P3/P4

Usage:
  from apps.devops import RunbookAgent
  agent = RunbookAgent(engine)
  result = agent.diagnose("CPU 98% on prod-api-03, response times >10s, OOM errors in logs")
  print(result.severity)     # P1
  print(result.steps)        # ['Check top processes', 'Review memory usage'...]
  print(result.runbook)      # name of matching runbook
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.engine import SRHNEngine


P1 = "P1"   # Complete outage, customer impact
P2 = "P2"   # Degraded service, partial impact
P3 = "P3"   # Minor issue, workaround available
P4 = "P4"   # Informational, no immediate action


@dataclass
class IncidentResult:
    alert:        str
    severity:     str
    runbook:      str
    steps:        list[str]
    root_cause:   str
    immediate_actions: list[str]
    escalate_to:  str
    confidence:   float
    query_id:     int
    elapsed_ms:   float
    memories_used: int

    def display(self) -> str:
        sev_icons = {P1:"🔴", P2:"🟠", P3:"🟡", P4:"🟢"}
        icon = sev_icons.get(self.severity, "⚪")
        lines = [
            f"\n  {icon}  SEVERITY: {self.severity}  Runbook: {self.runbook}",
            f"  {'─'*55}",
            f"  Alert: {self.alert[:120]}",
            f"\n  Root Cause: {self.root_cause}",
        ]
        if self.immediate_actions:
            lines.append("\n  Immediate actions:")
            for a in self.immediate_actions: lines.append(f"    1. {a}")
        if self.steps:
            lines.append("\n  Investigation steps:")
            for i, s in enumerate(self.steps): lines.append(f"    {i+1}. {s}")
        if self.escalate_to:
            lines.append(f"\n  Escalate to: {self.escalate_to}")
        lines.append(f"\n  [mem={self.memories_used} | {self.elapsed_ms:.0f}ms | conf={self.confidence:.0%}]")
        return "\n".join(lines)


class RunbookAgent:
    """Incident response and runbook agent backed by SRHN institutional memory."""

    _SYSTEM = """You are a senior SRE / DevOps engineer responding to infrastructure incidents.
You have access to past incidents, runbooks, and post-mortem learnings from memory.

For each alert:
1. Assign severity: P1 (outage) / P2 (degraded) / P3 (minor) / P4 (info)
2. Name the matching runbook (or "custom" if none)
3. List immediate actions (do NOW, < 5 minutes)
4. List investigation steps (methodical diagnosis)
5. Identify likely root cause category
6. Specify who to escalate to

Format response EXACTLY as:
SEVERITY: [P1/P2/P3/P4]
RUNBOOK: [runbook name]
IMMEDIATE: [semicolon-separated immediate actions]
STEPS: [semicolon-separated investigation steps]
ROOT_CAUSE: [category: memory | cpu | disk | network | database | deployment | external]
ESCALATE: [team or person, or "none"]

Be specific. "Check metrics" is not useful. "Run: kubectl top pods -n production | sort -k3 -rn | head -20" is useful."""

    def __init__(self, engine: SRHNEngine):
        self.engine = engine
        self._seed_runbooks()

    def _seed_runbooks(self):
        """Pre-load standard runbooks into SRHN memory."""
        runbooks = [
            # High CPU
            ("HIGH CPU RUNBOOK: 1) top -b -n1 | head -20 to identify process. "
             "2) If app process: check for runaway query, thread dump, heap dump. "
             "3) If kernel: check for crypto/disk I/O wait. "
             "4) kubectl top pods if k8s. 5) Scale horizontally if load spike. "
             "6) Consider graceful restart if memory leak.", "devops"),
            # OOM / Memory
            ("OOM RUNBOOK: 1) dmesg | grep -i oom to confirm OOM killer. "
             "2) Check /proc/meminfo, free -h. "
             "3) Identify pid from oom killer log. "
             "4) Heap dump: jmap -dump:format=b,file=heap.hprof <pid> (JVM). "
             "5) Check for memory leak pattern: gradual increase over 24h. "
             "6) Immediate: restart affected service if P1.", "devops"),
            # Disk full
            ("DISK FULL RUNBOOK: 1) df -h to identify full filesystem. "
             "2) du -sh /* | sort -rh | head to find large dirs. "
             "3) Common culprits: logs (/var/log), cores, tmp files, docker layers. "
             "4) log rotate: logrotate -f /etc/logrotate.conf. "
             "5) docker system prune -f to clear unused images. "
             "6) Do NOT just delete files blindly — check what they are first.", "devops"),
            # Database
            ("DATABASE SLOW QUERY RUNBOOK: 1) SHOW PROCESSLIST (MySQL) or "
             "SELECT * FROM pg_stat_activity WHERE state='active' (PG). "
             "2) EXPLAIN on slow queries. 3) Check index usage: SHOW INDEX. "
             "4) Check connections: SHOW STATUS LIKE 'Threads_connected'. "
             "5) Kill long-running queries > 60s. "
             "6) Check replication lag: SHOW SLAVE STATUS\\G.", "devops"),
            # Kubernetes
            ("K8S POD CRASHLOOP RUNBOOK: 1) kubectl describe pod <name> -n <ns> — check Events. "
             "2) kubectl logs <pod> --previous — get last crash logs. "
             "3) Check resource limits: OOMKilled = increase memory limit. "
             "4) Check liveness probe — may be too aggressive. "
             "5) kubectl get events --sort-by=.metadata.creationTimestamp. "
             "6) Check node status: kubectl get nodes.", "devops"),
            # Deployment
            ("FAILED DEPLOYMENT RUNBOOK: 1) Immediate: rollback if P1/P2. "
             "kubectl rollout undo deployment/<name>. "
             "2) Check deployment logs in CI/CD. "
             "3) Compare config between old and new version. "
             "4) Check health endpoint. "
             "5) Post-rollback: verify metrics return to baseline.", "devops"),
            # Network
            ("NETWORK LATENCY RUNBOOK: 1) ping, traceroute to identify hop. "
             "2) iperf3 for bandwidth test. "
             "3) ss -s for socket statistics. "
             "4) netstat -tulpn for open connections. "
             "5) Check DNS: dig, nslookup. "
             "6) Check security groups / firewall rules if cloud.", "devops"),
        ]
        for fact, domain in runbooks:
            self.engine.learn_fact(fact, domain=domain, confidence=0.96,
                                   source="SRE Runbook Library")

        # Common post-mortem learnings
        learnings = [
            ("POST-MORTEM LEARNING: Deploying during peak traffic (9am-5pm) caused 3 P1s. "
             "Deploy window: weekdays 11pm-6am, weekends 8am-10am only.", "devops"),
            ("POST-MORTEM LEARNING: Database connection pool exhaustion caused by "
             "long-running analytics queries on production DB. Solution: read replica.", "devops"),
            ("POST-MORTEM LEARNING: Docker image size >2GB causes k8s pod scheduling delays >5min. "
             "Use multi-stage builds, target <500MB.", "devops"),
        ]
        for fact, domain in learnings:
            self.engine.learn_fact(fact, domain=domain, confidence=0.9,
                                   source="Post-Mortem Database")

    def diagnose(self, alert: str, service: str = "",
                 env: str = "production") -> IncidentResult:
        """Diagnose an infrastructure alert and return runbook guidance."""
        t0    = time.perf_counter()
        ctx   = f"Service: {service}. " if service else ""
        query = (f"{ctx}Environment: {env}.\n"
                 f"Alert: {alert}\n"
                 f"Diagnose and provide runbook steps.")

        result   = self.engine.query(query, top_k=10)
        response = result.get("response", "")

        severity = self._parse(response, "SEVERITY",   P2)
        runbook  = self._parse(response, "RUNBOOK",    "custom")
        imm_raw  = self._parse(response, "IMMEDIATE",  "")
        step_raw = self._parse(response, "STEPS",      "")
        root     = self._parse(response, "ROOT_CAUSE", "unknown")
        escalate = self._parse(response, "ESCALATE",   "")

        immediate = [s.strip() for s in imm_raw.split(";") if s.strip()]
        steps     = [s.strip() for s in step_raw.split(";") if s.strip()]

        return IncidentResult(
            alert=alert,
            severity=severity,
            runbook=runbook,
            steps=steps,
            root_cause=root,
            immediate_actions=immediate,
            escalate_to=escalate,
            confidence=result.get("confidence", 0.5),
            query_id=result.get("query_id", 0),
            elapsed_ms=round((time.perf_counter()-t0)*1000, 1),
            memories_used=result.get("memories_used", 0),
        )

    def _parse(self, text: str, field: str, default: str) -> str:
        for line in text.splitlines():
            if line.upper().startswith(field + ":"):
                return line[len(field)+1:].strip()
        return default

    def resolved(self, query_id: int, what_fixed_it: str):
        """Mark an incident as resolved and store what fixed it."""
        self.engine.feedback(query_id, reward=+1.0, note=f"Fixed by: {what_fixed_it}")
        self.engine.learn_fact(
            f"RESOLUTION PATTERN: {what_fixed_it}",
            domain="devops", confidence=0.88, source="incident-resolution")

    def made_worse(self, query_id: int, what_failed: str):
        """This advice made things worse — learn from it."""
        self.engine.feedback(query_id, reward=-1.0, note=f"Made worse by: {what_failed}")

    def add_runbook(self, name: str, steps: list[str]):
        """Add a custom runbook from your environment."""
        self.engine.learn_workflow(name, steps, domain="devops")

    def add_post_mortem(self, incident: str, cause: str, fix: str):
        """Store a post-mortem finding for future incident response."""
        fact = f"POST-MORTEM: {incident} — Root cause: {cause} — Fix: {fix}"
        self.engine.learn_fact(fact, domain="devops", confidence=0.92,
                               source="post-mortem")
