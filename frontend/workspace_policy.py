"""UI-specific limits and choices shared by Lattice 272 workspaces.

These are interactive editor limits, NOT pricing-engine or model restrictions.
Importing this module never imports Streamlit, builds lattices, or prices options.
"""
from __future__ import annotations

MAX_VISUAL_STEPS = 100
MAX_BOUNDARY_STEPS = 200

COMPARISON_THRESHOLDS = (0.0, 1e-18, 1e-15, 1e-12)
EXPLORER_THRESHOLDS = (1e-18, 1e-9, 1e-5, 1e-3)
ANALYTICS_THRESHOLDS = (0.0, 1e-18, 1e-15, 1e-12, 1e-5)
