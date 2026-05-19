# QuietSignals — CLAUDE.md

## Project Overview
QuietSignals is a Streamlit-based nurse burnout intelligence dashboard. It predicts burnout risk (Low / Moderate / High) using a RandomForestClassifier trained on synthetic data structured after the TILES-2018 USC Keck Hospital dataset. The main app is `app/dashboard.py`.

## Key Files
- `app/dashboard.py` — single-file Streamlit app (all UI logic)
- `model/signals.py` — signal definitions, weights, palette constants
- `model/generator.py` — synthetic training data (1500 rows)
- `model/train.py` — trains RandomForestClassifier (300 trees)
- `model/predict.py` — composite score + RF predict_proba
- `model/fitbit.py` — simulates TILES-2018 Fitbit data
- `model/grayscale.py` — optional DeepFace facial emotion pipeline
- `requirements.txt` — numpy, pandas, matplotlib, scikit-learn

## Architecture Rules
- Do NOT add new Python files unless absolutely necessary — work within existing files.
- Do NOT add new dependencies to `requirements.txt` without client approval.
- All nurse data is synthetic (seeded random). No real patient data exists.
- `deepface`/`opencv` are intentionally excluded from `requirements.txt` for Streamlit Cloud compatibility.

## Client Priorities (approved feedback)
1. **Remove Color Chaos tab** — the exercise is done offline; its signal value gets entered in the Assessment tab as a normal slider input.
2. **Nurse ID-first profile view** — Tab 3 should let users look up a nurse by ID and see all their stats on one page (reference: clinical-black.vercel.app style).
3. **Signal-level burnout breakdown** — in the nurse profile, show what score the nurse placed in each individual signal metric (make the burnout score concrete).
4. **Reduce whitespace** — tighten vertical spacing and padding throughout.
5. **Department Overview** — client is happy with it; preserve as-is.
6. **Cross-session data persistence** — assessments submitted for a nurse must be saved and retrievable in future sessions. Data should accumulate over time to reflect how burnout develops (not just a single-session snapshot). Scope: single organization for now.

## Persistence Architecture
- Storage: SQLite (`data/quietsignals.db`) — lightweight, no extra dependencies beyond what's already in requirements.txt (Python stdlib `sqlite3`).
- On every Assessment submission, save: nurse_id, nurse_name, dept, shift, timestamp, all 10 signal values, composite_score, risk_level.
- The Nurse Profiles tab reads saved session history for the selected nurse.
- **Streamlit Cloud caveat**: the filesystem resets on each dyno restart; this is acceptable for the current demo/prototype phase. If permanent cloud persistence is needed later, migrate to an external DB (Supabase, etc.).

## Do Not Change Without Asking
- Color palette (`C` dict in dashboard.py) — client approved current colors
- Signal weights in `model/signals.py`
- The ML model architecture (RandomForest, 300 trees, 75/25 split)
- `.github/workflows/run_demo.yml`
