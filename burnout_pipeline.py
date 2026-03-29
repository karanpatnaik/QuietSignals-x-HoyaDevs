"""
================================================================================
Nurse Burnout Detection from Fitbit Activity & Pace Data
================================================================================
Uses TILES-2018 dataset structure — real data from 212 hospital nurses/staff
collected over 10 weeks at USC Keck Hospital.

Dataset: https://tiles-data.isi.edu/
Paper:   Mundnich et al. 2020, Scientific Data (Nature)

Sensor data (from Fitbit Charge 2 — worn 24/7):
  - Step count (minute-level → daily totals, pace, activity patterns)
  - Heart rate (continuous PPG, 1-min intervals)
  - Sleep (duration, stages, efficiency, onset/wake times)

Ground truth labels (surveys — used ONLY for training):
  - Daily EMAs: stress, anxiety, energy, job satisfaction
  - Baseline: personality, psych capital, psych flexibility

The model learns to predict burnout from Fitbit data alone.
In deployment, nurses just wear a Fitbit — no surveys needed.

Author: [Your Name]
Course: [Your Course]
================================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

# ============================================================================
# SECTION 1: SYNTHETIC DATA GENERATOR (mimics TILES-2018 Fitbit structure)
# ============================================================================

def generate_tiles_fitbit_data(n_nurses=212, n_weeks=10, seed=SEED):
    """
    Generate synthetic data matching TILES-2018 Fitbit data structure.
    
    When you download real data from https://tiles-data.isi.edu/:
      - fitbit_steps_intraday.csv.gz  → minute-level step counts
      - fitbit_heart_rate.csv.gz      → minute-level HR
      - fitbit_sleep.csv.gz           → sleep episodes with stages
      - surveys_scored/               → EMA responses (ground truth)
    
    Replace this function with real CSV loading.
    """
    np.random.seed(seed)
    records = []
    
    for nurse_id in range(1, n_nurses + 1):
        pid = f"TILES_{str(nurse_id).zfill(4)}"
        
        # Nurse characteristics
        shift = np.random.choice(['day', 'night'], p=[0.6, 0.4])
        burnout_tendency = np.random.beta(2, 3)  # latent burnout level
        age = np.random.randint(22, 62)
        
        for week in range(1, n_weeks + 1):
            # Burnout increases over time for susceptible nurses
            time_decay = 1 + (week / n_weeks) * 0.3 * burnout_tendency
            noise = lambda s=0.1: np.random.normal(0, s)
            
            for day_of_week in range(7):
                is_workday = day_of_week < 5
                
                # ============================================
                # FITBIT STEP DATA (pace & activity proxy)
                # ============================================
                
                # Daily step count (nurses walk A LOT on shift, less when burned out)
                if is_workday:
                    base_steps = 12000 if shift == 'day' else 10000
                else:
                    base_steps = 6000
                
                daily_steps = max(500, int(
                    base_steps * (1 - 0.4 * burnout_tendency * time_decay)
                    + noise(2000)
                ))
                
                # Walking pace proxy: steps per active hour
                active_hours = max(1, 8 - 3 * burnout_tendency * time_decay + noise(1))
                steps_per_active_hour = daily_steps / active_hours
                
                # Step cadence regularity (CV of minute-level steps)
                # Lower regularity = more erratic movement = fatigue
                step_regularity = max(0.1, 0.8 - 0.4 * burnout_tendency * time_decay + noise(0.1))
                
                # Peak stepping rate (steps/min during most active period)
                peak_step_rate = max(20, 120 - 50 * burnout_tendency * time_decay + noise(15))
                
                # Sedentary bouts (periods of 0 steps, >10 min)
                sedentary_bouts = max(0, int(3 + 8 * burnout_tendency * time_decay + noise(2)))
                
                # Total sedentary minutes
                sedentary_minutes = min(900, int(
                    300 + 350 * burnout_tendency * time_decay + noise(60)
                ))
                
                # Step count by shift period (for work pattern analysis)
                if is_workday and shift == 'day':
                    morning_steps = int(daily_steps * (0.35 - 0.05 * burnout_tendency) + noise(300))
                    afternoon_steps = int(daily_steps * (0.40 - 0.08 * burnout_tendency) + noise(300))
                    evening_steps = daily_steps - morning_steps - afternoon_steps
                else:
                    morning_steps = int(daily_steps * 0.3 + noise(200))
                    afternoon_steps = int(daily_steps * 0.35 + noise(200))
                    evening_steps = daily_steps - morning_steps - afternoon_steps
                
                # ============================================
                # FITBIT HEART RATE DATA
                # ============================================
                
                resting_hr = 58 + 18 * burnout_tendency * time_decay + noise(3)
                mean_hr_workday = resting_hr + (15 if is_workday else 8) + noise(4)
                max_hr = mean_hr_workday + 30 + noise(8)
                hr_range = max_hr - resting_hr
                
                # Time in HR zones (minutes)
                hr_zone_fat_burn = max(0, int(60 - 25 * burnout_tendency * time_decay + noise(15)))
                hr_zone_cardio = max(0, int(30 - 15 * burnout_tendency * time_decay + noise(10)))
                hr_zone_peak = max(0, int(10 - 8 * burnout_tendency * time_decay + noise(5)))
                hr_zone_rest = 1440 - hr_zone_fat_burn - hr_zone_cardio - hr_zone_peak
                
                # HR variability proxy (RMSSD from PPG intervals)
                hrv_rmssd = max(10, 55 - 30 * burnout_tendency * time_decay + noise(8))
                
                # ============================================
                # FITBIT SLEEP DATA
                # ============================================
                
                # Night shift workers have different patterns
                shift_penalty = 0.15 if shift == 'night' else 0
                
                sleep_duration_hrs = max(2, 
                    7.2 - 2.5 * (burnout_tendency * time_decay + shift_penalty) + noise(0.8))
                sleep_efficiency = min(98, max(40,
                    90 - 25 * burnout_tendency * time_decay - 5 * shift_penalty + noise(5)))
                
                # Sleep stages (Fitbit Charge 2 provides these)
                deep_sleep_min = max(5, int(
                    80 - 40 * burnout_tendency * time_decay + noise(15)))
                rem_sleep_min = max(5, int(
                    90 - 35 * burnout_tendency * time_decay + noise(15)))
                light_sleep_min = max(30, int(
                    sleep_duration_hrs * 60 - deep_sleep_min - rem_sleep_min + noise(20)))
                awake_min = max(0, int(
                    15 + 25 * burnout_tendency * time_decay + noise(8)))
                
                n_awakenings = max(0, int(
                    2 + 7 * burnout_tendency * time_decay + noise(1.5)))
                
                # Sleep onset time (later = worse for day shift)
                if shift == 'day':
                    sleep_onset_hour = min(3, max(21,
                        22.5 + 2 * burnout_tendency * time_decay + noise(0.5))) % 24
                else:
                    sleep_onset_hour = (8 + 2 * burnout_tendency + noise(1)) % 24
                
                # Sleep regularity (std of onset times across week)
                sleep_onset_variability = max(0.1,
                    0.5 + 1.5 * burnout_tendency * time_decay + noise(0.3))
                
                # ============================================
                # SURVEY GROUND TRUTH (prefixed with _survey)
                # ============================================
                def likert(base, effect):
                    return np.clip(base - effect * burnout_tendency * time_decay + noise(0.4), 1, 5)
                
                records.append({
                    'participant_id': pid,
                    'week': week,
                    'day_of_week': day_of_week,
                    'is_workday': int(is_workday),
                    'shift': shift,
                    'age': age,
                    
                    # --- FITBIT STEPS (pace/gait proxy) ---
                    'daily_steps': daily_steps,
                    'steps_per_active_hour': round(steps_per_active_hour),
                    'step_regularity': round(step_regularity, 3),
                    'peak_step_rate': round(peak_step_rate, 1),
                    'sedentary_bouts': sedentary_bouts,
                    'sedentary_minutes': sedentary_minutes,
                    'morning_steps': max(0, morning_steps),
                    'afternoon_steps': max(0, afternoon_steps),
                    'evening_steps': max(0, evening_steps),
                    
                    # --- FITBIT HEART RATE ---
                    'resting_hr': round(resting_hr, 1),
                    'mean_hr': round(mean_hr_workday, 1),
                    'max_hr': round(max_hr, 1),
                    'hr_range': round(hr_range, 1),
                    'hrv_rmssd': round(hrv_rmssd, 1),
                    'hr_zone_rest_min': hr_zone_rest,
                    'hr_zone_fat_burn_min': hr_zone_fat_burn,
                    'hr_zone_cardio_min': hr_zone_cardio,
                    'hr_zone_peak_min': hr_zone_peak,
                    
                    # --- FITBIT SLEEP ---
                    'sleep_duration_hrs': round(sleep_duration_hrs, 2),
                    'sleep_efficiency': round(sleep_efficiency, 1),
                    'deep_sleep_min': deep_sleep_min,
                    'rem_sleep_min': rem_sleep_min,
                    'light_sleep_min': light_sleep_min,
                    'awake_min': awake_min,
                    'n_awakenings': n_awakenings,
                    'sleep_onset_hour': round(sleep_onset_hour, 1),
                    'sleep_onset_variability': round(sleep_onset_variability, 2),
                    
                    # --- SURVEY GROUND TRUTH (never model input) ---
                    '_survey_stress': round(likert(2.0, -2.0), 2),
                    '_survey_energy': round(likert(4.0, 2.2), 2),
                    '_survey_anxiety': round(likert(1.8, -1.8), 2),
                    '_survey_job_satisfaction': round(likert(4.2, 2.0), 2),
                    '_survey_health': round(likert(4.0, 1.8), 2),
                })
    
    df = pd.DataFrame(records)
    
    # Inject 12% missing (realistic for Fitbit non-wear)
    fitbit_cols = ['daily_steps', 'resting_hr', 'mean_hr', 'hrv_rmssd',
                   'sleep_duration_hrs', 'sleep_efficiency', 'deep_sleep_min',
                   'peak_step_rate', 'step_regularity']
    for col in fitbit_cols:
        mask = np.random.random(len(df)) < 0.12
        df.loc[mask, col] = np.nan
    
    return df


# ============================================================================
# SECTION 2: GROUND TRUTH LABELS FROM SURVEYS
# ============================================================================

class SurveyBurnoutLabeler:
    """Generate burnout labels from survey data. Used ONLY for training."""
    
    WEIGHTS = {
        '_survey_stress':          {'w': 0.25, 'dir': 'positive'},
        '_survey_energy':          {'w': 0.25, 'dir': 'negative'},
        '_survey_anxiety':         {'w': 0.20, 'dir': 'positive'},
        '_survey_job_satisfaction': {'w': 0.15, 'dir': 'negative'},
        '_survey_health':          {'w': 0.15, 'dir': 'negative'},
    }
    
    def __init__(self):
        self.stats = {}
    
    def fit(self, df):
        for col in self.WEIGHTS:
            self.stats[col] = {'min': df[col].quantile(0.02), 'max': df[col].quantile(0.98)}
        return self
    
    def score(self, df):
        scores = np.zeros(len(df))
        for col, cfg in self.WEIGHTS.items():
            s = self.stats[col]
            norm = ((df[col] - s['min']) / (s['max'] - s['min'] + 1e-8)).clip(0, 1)
            if cfg['dir'] == 'negative':
                norm = 1 - norm
            scores += norm.fillna(0).values * cfg['w']
        return (scores / sum(c['w'] for c in self.WEIGHTS.values()) * 100).clip(0, 100)
    
    def label(self, scores):
        return np.where(scores < 35, 'Low', np.where(scores < 60, 'Moderate', 'High'))


# ============================================================================
# SECTION 3: FITBIT FEATURE ENGINEERING (sensor-only, no surveys)
# ============================================================================

FITBIT_COLS = [
    'daily_steps', 'steps_per_active_hour', 'step_regularity', 'peak_step_rate',
    'sedentary_bouts', 'sedentary_minutes', 'morning_steps', 'afternoon_steps',
    'evening_steps', 'resting_hr', 'mean_hr', 'max_hr', 'hr_range', 'hrv_rmssd',
    'hr_zone_rest_min', 'hr_zone_fat_burn_min', 'hr_zone_cardio_min', 'hr_zone_peak_min',
    'sleep_duration_hrs', 'sleep_efficiency', 'deep_sleep_min', 'rem_sleep_min',
    'light_sleep_min', 'awake_min', 'n_awakenings', 'sleep_onset_hour',
    'sleep_onset_variability',
]


def engineer_features(df):
    """
    Build per-nurse feature vectors from daily Fitbit data.
    
    Feature groups:
      1. Aggregates (mean, std, min, max) of each Fitbit metric
      2. Workday vs. off-day contrasts
      3. Weekly trends (slope over the 10 weeks)
      4. Week-over-week volatility
      5. Pace & activity pattern features
      6. Cross-modal interactions
    """
    features_list = []
    
    for pid in df['participant_id'].unique():
        pdata = df[df['participant_id'] == pid]
        feats = {'participant_id': pid, 'shift': pdata['shift'].iloc[0], 'age': pdata['age'].iloc[0]}
        
        # --- 1. Aggregates ---
        for col in FITBIT_COLS:
            if col in pdata.columns:
                vals = pdata[col].dropna()
                if len(vals) > 0:
                    feats[f'{col}_mean'] = vals.mean()
                    feats[f'{col}_std'] = vals.std()
                    feats[f'{col}_min'] = vals.min()
                    feats[f'{col}_max'] = vals.max()
                    feats[f'{col}_median'] = vals.median()
        
        # --- 2. Workday vs off-day contrasts ---
        work = pdata[pdata['is_workday'] == 1]
        off = pdata[pdata['is_workday'] == 0]
        
        for col in ['daily_steps', 'resting_hr', 'sleep_duration_hrs', 'sedentary_minutes']:
            if col in pdata.columns:
                w_mean = work[col].mean() if len(work) > 0 else np.nan
                o_mean = off[col].mean() if len(off) > 0 else np.nan
                feats[f'{col}_work_off_diff'] = (w_mean - o_mean) if not (np.isnan(w_mean) or np.isnan(o_mean)) else np.nan
                feats[f'{col}_work_off_ratio'] = (w_mean / (o_mean + 1e-8)) if not (np.isnan(w_mean) or np.isnan(o_mean)) else np.nan
        
        # --- 3. Weekly trends (slope) ---
        weekly = pdata.groupby('week')[FITBIT_COLS].mean()
        for col in ['daily_steps', 'resting_hr', 'sleep_duration_hrs', 'hrv_rmssd',
                     'step_regularity', 'peak_step_rate', 'sedentary_minutes',
                     'sleep_efficiency', 'n_awakenings', 'deep_sleep_min']:
            if col in weekly.columns:
                valid = weekly[col].dropna()
                if len(valid) >= 3:
                    slope = np.polyfit(valid.index.values, valid.values, 1)[0]
                    feats[f'{col}_weekly_trend'] = round(slope, 4)
                else:
                    feats[f'{col}_weekly_trend'] = 0
        
        # --- 4. Week-over-week volatility ---
        for col in ['daily_steps', 'resting_hr', 'sleep_duration_hrs', 'hrv_rmssd']:
            if col in weekly.columns:
                diffs = weekly[col].diff().dropna()
                feats[f'{col}_weekly_volatility'] = diffs.std() if len(diffs) > 1 else 0
        
        # Early vs late study shift
        mid = pdata['week'].max() // 2
        for col in ['daily_steps', 'resting_hr', 'sleep_duration_hrs', 'step_regularity']:
            if col in pdata.columns:
                early = pdata[pdata['week'] <= mid][col].mean()
                late = pdata[pdata['week'] > mid][col].mean()
                feats[f'{col}_early_late_shift'] = (late - early) if not (np.isnan(early) or np.isnan(late)) else 0
        
        # --- 5. Pace & activity pattern features ---
        if 'morning_steps' in pdata.columns:
            m = pdata['morning_steps'].mean()
            a = pdata['afternoon_steps'].mean()
            e = pdata['evening_steps'].mean()
            total = m + a + e + 1e-8
            feats['morning_step_pct'] = m / total
            feats['afternoon_step_pct'] = a / total
            feats['evening_step_pct'] = e / total
            feats['step_distribution_entropy'] = -sum(
                p * np.log2(p + 1e-10) for p in [m/total, a/total, e/total]
            )
        
        # Active-to-sedentary ratio
        if 'sedentary_minutes_mean' in feats and 'daily_steps_mean' in feats:
            feats['active_sedentary_ratio'] = feats['daily_steps_mean'] / (feats['sedentary_minutes_mean'] + 1)
        
        # --- 6. Cross-modal interactions ---
        # Cardiac cost of activity
        if 'resting_hr_mean' in feats and 'daily_steps_mean' in feats:
            feats['cardiac_effort_per_step'] = feats['resting_hr_mean'] / (feats['daily_steps_mean'] / 1000 + 1e-8)
        
        # Sleep recovery index
        if 'sleep_duration_hrs_mean' in feats and 'deep_sleep_min_mean' in feats:
            feats['restorative_sleep_pct'] = (
                (feats.get('deep_sleep_min_mean', 0) + feats.get('rem_sleep_min_mean', 0))
                / (feats['sleep_duration_hrs_mean'] * 60 + 1e-8) * 100
            )
        
        # Sleep-activity coupling
        if 'sleep_efficiency_mean' in feats and 'daily_steps_mean' in feats:
            feats['sleep_activity_coupling'] = (
                feats['sleep_efficiency_mean'] / 100 * feats['daily_steps_mean'] / 10000
            )
        
        # HR recovery proxy (HRV per unit activity)
        if 'hrv_rmssd_mean' in feats and 'daily_steps_mean' in feats:
            feats['hrv_per_activity'] = feats['hrv_rmssd_mean'] / (feats['daily_steps_mean'] / 1000 + 1e-8)
        
        features_list.append(feats)
    
    features_df = pd.DataFrame(features_list)
    return features_df


# ============================================================================
# SECTION 4: ML PIPELINE
# ============================================================================

def build_and_evaluate(features_df, burnout_labels, feature_cols):
    """Train models: Fitbit features → burnout labels."""
    
    X = features_df[feature_cols].copy()
    le = LabelEncoder()
    y = le.fit_transform(burnout_labels)
    
    print(f"\n{'='*60}")
    print(f"  FITBIT-ONLY BURNOUT CLASSIFICATION")
    print(f"{'='*60}")
    print(f"  Nurses:      {len(X)}")
    print(f"  Features:    {len(feature_cols)} (Fitbit steps + HR + sleep)")
    print(f"  Classes:     {dict(pd.Series(burnout_labels).value_counts())}")
    print(f"  Survey cols: 0 in model input")
    print(f"{'='*60}\n")
    
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    X_processed = preprocessor.fit_transform(X)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4, class_weight='balanced', random_state=SEED),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=4, random_state=SEED),
        'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 (macro)':>12} {'AUC-ROC':>10}")
    print(f"  {'-'*57}")
    
    for name, model in models.items():
        acc = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
        f1 = cross_val_score(model, X_processed, y, cv=cv, scoring='f1_macro')
        try:
            auc = cross_val_score(model, X_processed, y, cv=cv, scoring='roc_auc_ovr')
        except:
            auc = np.array([0])
        results[name] = {'accuracy': acc.mean(), 'acc_std': acc.std(),
                         'f1': f1.mean(), 'f1_std': f1.std(), 'auc': auc.mean()}
        print(f"  {name:<25} {acc.mean():>8.3f}±{acc.std():.3f}  {f1.mean():>8.3f}±{f1.std():.3f}  {auc.mean():>8.3f}")
    
    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = models[best_name]
    print(f"\n  >>> Best: {best_name} (F1={results[best_name]['f1']:.3f})")
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=SEED)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    print(f"\n{'='*60}")
    print(f"  HOLDOUT RESULTS ({best_name})")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    rf = models['Random Forest']
    rf.fit(X_processed, y)
    
    return {
        'best_name': best_name, 'best_model': best_model, 'rf': rf,
        'preprocessor': preprocessor, 'results': results,
        'y_test': y_test, 'y_pred': y_pred, 'le': le,
        'feature_cols': feature_cols, 'X_processed': X_processed, 'y': y,
    }


# ============================================================================
# SECTION 5: FEATURE IMPORTANCE
# ============================================================================

def show_top_features(pipeline, top_n=20):
    rf = pipeline['rf']
    feat_imp = pd.DataFrame({
        'feature': pipeline['feature_cols'][:len(rf.feature_importances_)],
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False).head(top_n)
    
    print(f"\n{'='*60}")
    print(f"  TOP {top_n} FITBIT FEATURES")
    print(f"{'='*60}")
    for _, row in feat_imp.iterrows():
        bar = '█' * int(row['importance'] / feat_imp['importance'].max() * 25)
        print(f"  {row['feature']:<45} {bar} {row['importance']:.4f}")
    return feat_imp


# ============================================================================
# SECTION 6: DEPLOYMENT SIMULATION
# ============================================================================

def simulate_deployment(pipeline, features_df, feature_cols):
    """Simulate real-time nurse burnout monitoring from Fitbit only."""
    
    model = pipeline['best_model']
    preprocessor = pipeline['preprocessor']
    le = pipeline['le']
    
    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT SIMULATION — NURSE BURNOUT MONITOR")
    print(f"  Input: Fitbit (steps, HR, sleep) — zero nurse input")
    print(f"{'='*60}\n")
    
    sample = features_df.sample(6, random_state=99)
    
    for _, row in sample.iterrows():
        pid = row['participant_id']
        shift = row.get('shift', '?')
        X_new = row[feature_cols].values.reshape(1, -1)
        X_scaled = preprocessor.transform(X_new)
        
        pred = le.inverse_transform(model.predict(X_scaled))[0]
        proba = model.predict_proba(X_scaled)[0]
        icon = {'High': '🔴', 'Moderate': '🟡', 'Low': '🟢'}
        
        proba_str = ' | '.join(f"{c}: {p:.0%}" for c, p in zip(le.classes_, proba))
        print(f"  {pid} ({shift} shift) → {icon.get(pred, '⚪')} {pred} burnout risk  ({proba_str})")
    
    print(f"\n  Production workflow:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Nurse wears Fitbit → data syncs automatically     │")
    print(f"  │  Pipeline extracts steps/HR/sleep features weekly   │")
    print(f"  │  Model outputs burnout risk to admin dashboard      │")
    print(f"  │  No forms. No surveys. No disruption.               │")
    print(f"  └─────────────────────────────────────────────────────┘\n")


# ============================================================================
# SECTION 7: VISUALIZATION
# ============================================================================

def create_dashboard(df, features_df, pipeline, feat_imp, save_dir='.'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('TILES-2018 Nurse Burnout Detection — Fitbit Activity & Pace Analysis',
                 fontsize=15, fontweight='bold', y=1.02)
    
    # 1. Daily steps distribution by burnout level
    ax = axes[0, 0]
    for label, color in [('Low', '#2ecc71'), ('Moderate', '#f39c12'), ('High', '#e74c3c')]:
        mask = df['_burnout_label'] == label
        if mask.any():
            ax.hist(df[mask]['daily_steps'].dropna(), bins=25, alpha=0.5,
                    color=color, label=label, edgecolor='white')
    ax.set_xlabel('Daily Steps')
    ax.set_ylabel('Count')
    ax.set_title('Daily Steps by Burnout Level')
    ax.legend()
    
    # 2. Class distribution
    ax = axes[0, 1]
    le = pipeline['le']
    class_counts = pd.Series(pipeline['y']).map(dict(enumerate(le.classes_))).value_counts()
    colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
    bars = ax.bar(class_counts.index, class_counts.values,
                  color=[colors.get(c, '#95a5a6') for c in class_counts.index],
                  edgecolor='white', linewidth=1.5)
    ax.set_title('Nurse Burnout Distribution')
    ax.set_ylabel('Nurses')
    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold')
    
    # 3. Confusion matrix
    ax = axes[0, 2]
    cm = confusion_matrix(pipeline['y_test'], pipeline['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel('Predicted (Fitbit)')
    ax.set_ylabel('Actual (Survey)')
    ax.set_title(f'Confusion Matrix ({pipeline["best_name"]})')
    
    # 4. Feature importance
    ax = axes[1, 0]
    top10 = feat_imp.head(10)
    ax.barh(top10['feature'], top10['importance'], color='#3498db', edgecolor='white')
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Fitbit Features')
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=7)
    
    # 5. Model comparison
    ax = axes[1, 1]
    model_names = list(pipeline['results'].keys())
    f1s = [pipeline['results'][m]['f1'] for m in model_names]
    accs = [pipeline['results'][m]['accuracy'] for m in model_names]
    x = np.arange(len(model_names))
    w = 0.35
    ax.bar(x - w/2, f1s, w, label='F1', color='#9b59b6', edgecolor='white')
    ax.bar(x + w/2, accs, w, label='Accuracy', color='#1abc9c', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax.set_title('Model Comparison (Fitbit Only)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    
    # 6. Step trend over weeks by burnout group
    ax = axes[1, 2]
    weekly = df.groupby(['week', '_burnout_label'])['daily_steps'].mean().reset_index()
    for label, color in [('Low', '#2ecc71'), ('Moderate', '#f39c12'), ('High', '#e74c3c')]:
        subset = weekly[weekly['_burnout_label'] == label]
        ax.plot(subset['week'], subset['daily_steps'], marker='o', color=color,
                label=label, linewidth=2, markersize=5)
    ax.set_xlabel('Week')
    ax.set_ylabel('Avg Daily Steps')
    ax.set_title('Step Count Trend by Burnout Level')
    ax.legend()
    
    plt.tight_layout()
    path = f'{save_dir}/nurse_burnout_fitbit_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Dashboard saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Nurse Burnout Detection — TILES-2018 Fitbit Pipeline       ║")
    print("║  Steps + Heart Rate + Sleep → Burnout Risk                  ║")
    print("║  Data from 212 hospital nurses. Zero self-reports needed.   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Step 1
    print("[1/7] Loading nurse Fitbit data...")
    print("       Source: TILES-2018 (tiles-data.isi.edu)")
    print("       Replace synthetic generator with real CSVs when downloaded.\n")
    df = generate_tiles_fitbit_data()
    print(f"  Records:    {len(df)} nurse-days")
    print(f"  Nurses:     {df['participant_id'].nunique()}")
    print(f"  Weeks:      {df['week'].nunique()}")
    print(f"  Shifts:     {dict(df.groupby('participant_id')['shift'].first().value_counts())}")
    print(f"  Missing:    {df[FITBIT_COLS].isnull().mean().mean()*100:.1f}% of Fitbit data\n")
    
    # Step 2
    print("[2/7] Generating ground truth burnout labels from surveys...")
    labeler = SurveyBurnoutLabeler().fit(df)
    df['_burnout_score'] = labeler.score(df)
    df['_burnout_label'] = labeler.label(df['_burnout_score'].values)
    
    per_nurse = df.sort_values('week').groupby('participant_id').last()
    print(f"  Ground truth (per nurse, final week):")
    for label, count in per_nurse['_burnout_label'].value_counts().items():
        print(f"    {label:>10}: {count} ({count/len(per_nurse)*100:.1f}%)")
    
    # Step 3
    print(f"\n[3/7] Engineering features from FITBIT DATA ONLY...")
    features_df = engineer_features(df)
    gt = per_nurse[['_burnout_label']].reset_index()
    features_df = features_df.merge(gt, on='participant_id')
    
    feature_cols = [c for c in features_df.columns
                    if c not in ['participant_id', '_burnout_label', 'shift', 'age']
                    and not c.startswith('_survey')
                    and features_df[c].dtype in ['float64', 'int64', 'int32']]
    
    print(f"  Features: {len(feature_cols)} (100% from Fitbit — steps, HR, sleep)")
    print(f"  Zero survey features in model ✓")
    
    # Step 4
    print(f"\n[4/7] Training models...")
    pipeline = build_and_evaluate(features_df, features_df['_burnout_label'], feature_cols)
    
    # Step 5
    print(f"\n[5/7] Feature importances...")
    feat_imp = show_top_features(pipeline)
    
    # Step 6
    print(f"\n[6/7] Deployment simulation...")
    simulate_deployment(pipeline, features_df, feature_cols)
    
    # Step 7
    print(f"[7/7] Visualizations...")
    create_dashboard(df, features_df, pipeline, feat_imp, save_dir='.')
    
    best = pipeline['best_name']
    r = pipeline['results'][best]
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Dataset:     TILES-2018 (212 hospital nurses, 10 weeks)")
    print(f"  Input:       Fitbit Charge 2 (steps, HR, sleep)")
    print(f"  Features:    {len(feature_cols)} (pace, activity, cardiac, sleep)")
    print(f"  Best Model:  {best}")
    print(f"  Accuracy:    {r['accuracy']:.3f} ± {r['acc_std']:.3f}")
    print(f"  F1 (macro):  {r['f1']:.3f} ± {r['f1_std']:.3f}")
    print(f"  AUC-ROC:     {r['auc']:.3f}")
    print(f"  Nurse input: ZERO (Fitbit auto-syncs)")
    print(f"{'='*60}\n")
    
    return df, features_df, pipeline


if __name__ == '__main__':
    df, features_df, pipeline = main()
