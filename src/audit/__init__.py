"""Regulatory audit finding generator.

Modules
-------
variable_mapper  — maps regulation concepts to DB columns (run once per SAS version)
dqc_engine       — stores and executes approved DQC SQL queries
finding_generator — converts DQC violations to structured audit findings
"""

from .variable_mapper import build_mapping, load_mapping, MappingEntry

__all__ = ["build_mapping", "load_mapping", "MappingEntry"]
