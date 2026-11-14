"""
Phase 4: Validation & Reporting
Adds similarity confidence intervals and permutation-test p-values to detected
research frontiers, and writes the HTML gap report.
"""

from .hypothesis_test import StatisticalValidator
from .report import write_html_report

__all__ = ["StatisticalValidator", "write_html_report"]
