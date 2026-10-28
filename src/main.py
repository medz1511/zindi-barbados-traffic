"""
Barbados Traffic Analysis Challenge (Zindi) - "Traffic Engineer" model
Input : input/all_features_EXPERT.csv, input/Train.csv, input/SampleSubmission.csv
Output: output/submission_PHYSICS_ENGINEER.csv
Usage : python src/main.py
"""
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CELLULE 90 : L'INGÉNIEUR TRAFIC (PHYSIQUE & CYCLES) ---
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler

print("👷 DÉMARRAGE DE L'INGÉNIEUR TRAFIC (PHYSICS BOOST)...")

# ============================================================
# 1. PRÉPARATION & NETTOYAGE
# ============================================================
try:
    df_features = pd.read_csv(os.path.join(INPUT_DIR, 'all_features_EXPERT.csv'))
    train_original = pd.read_csv(os.path.join(INPUT_DIR, 'Train.csv'))
    ss = pd.read_csv(os.path.join(INPUT_DIR, 'SampleSubmission.csv'))
except:
    base_path = INPUT_DIR
    df_features = pd.read_csv(os.path.join(base_path, 'all_features_EXPERT.csv'))
    train_original = pd.read_csv(os.path.join(INPUT_DIR, 'Train.csv'))
    ss = pd.read_csv(os.path.join(INPUT_DIR, 'SampleSubmission.csv'))

def get_clean_id(x):
    try: return int(str(x).split('_')[-5])
    except: return int(x)
df_features['clean_seg_id'] = df_features.apply(lambda r: get_clean_id(r.get('time_segment_id', r.get('videos_path'))), axis=1)
df_features.loc[df_features['num_cars'] == 0, ['density', 'congestion_index']] = 0

# GLOBAL STATS (La base solide)
global_stats = df_features.groupby('clean_seg_id').agg({
    'density': ['mean', 'sum'], 'avg_speed': 'mean', 'congestion_index': 'max'
}).reset_index()
global_stats.columns = ['clean_seg_id', 'glob_dens_mean', 'glob_dens_sum', 'glob_spd_mean', 'glob_cong_max']
df_features = df_features.merge(global_stats, on='clean_seg_id', how='left')

# ============================================================
# 2. FEATURE ENGINEERING : LA PHYSIQUE (NOUVEAU !)
# ============================================================
print("   🧪 Injection des formules physiques...")

# A. FLUX & INTENSITÉ (Physique des fluides)
# On ajoute +1 au diviseur pour éviter la division par zéro
df_features['phys_flow'] = df_features['glob_dens_mean'] * df_features['glob_spd_mean'] # Débit
df_features['phys_jam'] = df_features['glob_dens_mean'] / (df_features['glob_spd_mean'] + 1) # Intensité du bouchon

# B. CYCLICITÉ TEMPORELLE (Mathématiques)
# Transforme l'heure en cercle (23h touche 0h)
df_features['video_time'] = pd.to_datetime(df_features['video_time'])
df_features['hour'] = df_features['video_time'].dt.hour
df_features['minute'] = df_features['video_time'].dt.minute

df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour']/24)
df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour']/24)

# On garde le tri temporel pour les lags
df_features = df_features.sort_values(['view_label', 'clean_seg_id'])

# LAGS (On inclut les nouvelles features physiques !)
cols_to_lag = ['num_cars', 'avg_speed', 'density', 'phys_flow', 'phys_jam', 'glob_dens_mean', 'glob_spd_mean']
lag_dfs = []
for lag in [1, 2, 3]:
    shifted = df_features.groupby('view_label')[cols_to_lag].shift(lag)
    shifted.columns = [f'{col}_lag{lag}' for col in cols_to_lag]
    lag_dfs.append(shifted)
df_features = pd.concat([df_features] + lag_dfs, axis=1).fillna(0)

# Normalisation
scaler = RobustScaler()
# On sélectionne toutes les colonnes numériques utiles
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
exclude_cols = ['clean_seg_id', 'hour', 'minute'] # On ne normalise pas les IDs ou le temps brut
cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]
df_features[cols_to_scale] = scaler.fit_transform(df_features[cols_to_scale])

# ============================================================
# 3. PRÉPARATION TRAIN/TEST
# ============================================================
train_labels = pd.read_csv(os.path.join(INPUT_DIR, 'Train.csv'))
train_full = df_features[df_features['dataset'] == 'train'].merge(
    train_labels[['time_segment_id', 'view_label', 'congestion_enter_rating']],
    left_on=['clean_seg_id', 'view_label'], right_on=['time_segment_id', 'view_label'], how='inner'
)
train_full = train_full.sort_values(['view_label', 'clean_seg_id'])
train_full['target_future'] = train_full.groupby('view_label')['congestion_enter_rating'].shift(-3)
train_ready = train_full.dropna(subset=['target_future']).copy()
test_ready = df_features[df_features['dataset'] == 'test'].copy()
test_ready = test_ready.sort_values(['view_label', 'clean_seg_id'])

drop_cols = ['dataset', 'videos_path', 'video_time', 'date', 'congestion_enter_rating', 'target_future', 'y_target', 'signaling', 'view_label']
feature_cols = [c for c in train_ready.columns if c not in drop_cols and c in test_ready.columns and train_ready[c].dtype in [np.float64, np.int64]]

# ============================================================
# 4. ENTRAÎNEMENT (POIDS FINAL CUT)
# ============================================================
# On garde tes poids gagnants : Light=4.2, Moderate=3.1
weights_final = {'free flowing': 1.0, 'light delay': 4.2, 'moderate delay': 3.1, 'heavy delay': 4.5}
print(f"🚀 Entraînement Physique ({len(feature_cols)} features)...")

params = {
    'n_estimators': 3500, 'learning_rate': 0.012, 'num_leaves': 35,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'class_weight': weights_final,
    'n_jobs': -1, 'verbose': -1
}

X = train_ready[feature_cols]
y = train_ready['target_future']
X_test = test_ready[feature_cols]

SEEDS = [42, 2025, 777, 101, 888, 999, 123, 456, 1000, 555] # 10 seeds standard
final_probs = np.zeros((len(test_ready), 4))
classes_order = None

for i, seed in enumerate(SEEDS):
    print(f"   🌱 Seed {i+1}/10...", end='\r')
    params['random_state'] = seed
    clf = LGBMClassifier(**params)
    clf.fit(X, y)
    if classes_order is None: classes_order = clf.classes_
    final_probs += clf.predict_proba(X_test) / len(SEEDS)

# ============================================================
# 5. LISSAGE TEMPOREL (CELUI QUI A DONNÉ 0.626)
# ============================================================
print("\n🌊 Lissage Temporel...")
test_ready['prob_0'] = final_probs[:, 0]
test_ready['prob_1'] = final_probs[:, 1]
test_ready['prob_2'] = final_probs[:, 2]
test_ready['prob_3'] = final_probs[:, 3]

# Rolling mean (Window 3)
rolling_probs = test_ready.groupby('view_label')[['prob_0', 'prob_1', 'prob_2', 'prob_3']].transform(lambda x: x.rolling(window=3, center=True, min_periods=1).mean())

# Mix 60% Brut / 40% Lissée
final_probs_smoothed = (test_ready[['prob_0', 'prob_1', 'prob_2', 'prob_3']].values * 0.6) + (rolling_probs.values * 0.4)

# ============================================================
# 6. SOUMISSION
# ============================================================
print("🔧 Calibration...")
# Ordre : Free, Heavy, Light, Moderate
calib_dict = {'free flowing': 0.95, 'light delay': 1.15, 'moderate delay': 1.05, 'heavy delay': 1.0}
calibration_vector = np.array([calib_dict[c] for c in classes_order])
final_probs_calibrated = final_probs_smoothed * calibration_vector

final_preds_idx = np.argmax(final_probs_calibrated, axis=1)
final_preds_str = [classes_order[i] for i in final_preds_idx]

test_ready['final_pred'] = final_preds_str
submission_map = dict(zip(test_ready['clean_seg_id'].astype(str) + '_' + test_ready['view_label'], test_ready['final_pred']))

knowledge_base = {}
for (idx, row), pred in zip(test_ready.iterrows(), final_preds_str):
    knowledge_base.setdefault(row['view_label'], {})[row['clean_seg_id']] = pred

submission_rows = []
for _, row in ss.iterrows():
    row_id = row['ID']
    val = "free flowing"
    try:
        parts = row_id.split('_')
        target_id = int(parts[2])
        view = next((v for v in ["Norman Niles #1", "Norman Niles #2", "Norman Niles #3", "Norman Niles #4"] if v in row_id), "Unknown")
        key = str(target_id) + '_' + view
        if key in submission_map: val = submission_map[key]
        else:
            if view in knowledge_base:
                candidates = [pid for pid in knowledge_base[view].keys() if pid <= target_id]
                if candidates:
                    best_source = max(candidates)
                    if (target_id - best_source) < 15:
                        val = knowledge_base[view][best_source]
    except: pass
    submission_rows.append({'ID': row_id, 'Target': val, 'Target_Accuracy': val})

sub_df = pd.DataFrame(submission_rows)
filename = os.path.join(OUTPUT_DIR, 'submission_PHYSICS_ENGINEER.csv')
sub_df.to_csv(filename, index=False)

print(f"\n📊 RÉPARTITION FINALE ({filename}) :")
print(sub_df['Target'].value_counts())
