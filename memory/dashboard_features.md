---
name: Dashboard feature set (April 2026 rewrite)
description: All tabs, charts, and features in the major dashboard rewrite — helps avoid duplication
type: project
---

Major rewrite of app/dashboard.py (April 2026). The app is now organized into 5 tabs.

**Tab 1 — Assessment**: manual/Fitbit/facial input → risk badge, probability bar chart, radar chart (new), warning flags, AI suggestions, signal contributions bar chart, raw signal table with traffic-light status column.

**Tab 2 — Department Overview**: aggregated hospital stats (total nurses, avg score, risk counts), dept bar chart with threshold lines, at-risk nurse table.

**Tab 3 — Nurse Profiles & Forecast**: 20-nurse mock profile database (get_nurse_profiles()), trend + 4-week forecast chart, dept percentile comparison.

**Tab 4 — Fitbit Live Feed**: simulated real-time sensor metrics, auto-refresh toggle, live burnout estimate, 24-hour HRV/HR dual-panel trend chart.

**Tab 5 — Color Chaos Exercise**: 10-trial Stroop test (mismatched color word/font), accuracy + RT tracking, maps result to Color-Pattern Chaos signal value.

**Key fixes:**
- Low range bug fixed: dashboard now uses score_to_label(composite_score) with thresholds 0.33 / 0.55 instead of classifier argmax.
- generator.py label thresholds changed from (0.30, 0.50) to (0.33, 0.55) — Low class now ~37% of training data.

**Color palette:** teal primary (#0d9488), green/amber/red for risk, indigo for facial signals, slate for muted text.

**Why:** user requested all features in one session (April 9 2026).
**How to apply:** when adding new features, keep the 5-tab structure; do not revert score_to_label approach.
