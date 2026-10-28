"""
Barbados Traffic Analysis Challenge (Zindi) - Video feature extraction
Input : input/Train.csv, input/TestInputSegments.csv + videos from GCS bucket `brb-traffic`
Output: input/all_features_EXPERT.csv (appended row by row, resumable via input/processed_log.txt)
Prerequisite: `gcloud auth application-default login`
Usage : python src/extract_features.py   (run again until "0 Tests | 0 Trains")
"""
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CELLULE 1 : CONFIGURATION ET FONCTION D'EXTRACTION EXPERTE ---
import os
import torch
import cv2
import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image

# 1. Chargement du Modèle (GPU si dispo)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Utilisation du device : {device}")

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
model.eval()

# Classes COCO pertinentes
COCO_TO_CUSTOM = {1: 'person', 3: 'car', 8: 'truck', 6: 'bus'}

# 2. Fonction d'Extraction Optimisée (Vitesse + Features Expertes)
def extract_features_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened(): return None

    vehicle_counts = {'car': 0, 'truck': 0, 'bus': 0, 'person': 0}
    flow_speeds = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 300 # Sécurité

    # STRATÉGIE SNAPSHOT : On ne regarde que 5 images réparties sur la vidéo
    frame_indices = np.linspace(0, total_frames - 2, 5, dtype=int)

    prev_gray = None

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break

        # A. Optimisation Vitesse : Redimensionner en 320x180
        frame_small = cv2.resize(frame, (320, 180))

        # B. Détection (FasterRCNN)
        img = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
        img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor)[0]

        for label, score in zip(preds['labels'], preds['scores']):
            # Seuil 0.4 suffisant pour basse résolution
            if score > 0.4 and label.item() in COCO_TO_CUSTOM:
                vehicle_counts[COCO_TO_CUSTOM[label.item()]] += 1

        # C. Flux Optique (Vitesse)
        ret2, frame_next = cap.read()
        if ret2:
            frame_next_small = cv2.resize(frame_next, (320, 180))
            gray_curr = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray_next = cv2.cvtColor(frame_next_small, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(gray_curr, gray_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_speeds.append(np.mean(magnitudes))

    cap.release()

    divider = len(frame_indices) if len(frame_indices) > 0 else 1

    # --- CALCUL DES FEATURES ---
    avg_speed = np.mean(flow_speeds) if flow_speeds else 0
    total_vehicles = sum(vehicle_counts.values()) / divider

    # Expert 1 : Variation de vitesse (Indique stop & go)
    std_speed = np.std(flow_speeds) if flow_speeds else 0

    # Expert 2 : Indice de Congestion (Densité / Vitesse)
    congestion_index = total_vehicles / (avg_speed + 0.1)

    return {
        'num_cars': vehicle_counts['car'] / divider,
        'num_buses': vehicle_counts['bus'] / divider,
        'num_trucks': vehicle_counts['truck'] / divider,
        'num_peds': vehicle_counts['person'] / divider,
        'avg_speed': avg_speed,
        'density': total_vehicles,
        'queue_length_proxy': max(vehicle_counts['car'], vehicle_counts['bus']) / divider,
        'std_speed': std_speed,
        'congestion_index': congestion_index
    }
print("✅ Configuration terminée.")


# --- CELLULE 2 : TRAITEMENT PRIORITAIRE (TEST INPUT FIRST) ---
import os
import pandas as pd
from google.cloud import storage

# --- CONFIGURATION ---
client = storage.Client(project="brb-traffic")
bucket_name = 'brb-traffic'
drive_save_path = INPUT_DIR
os.makedirs(drive_save_path, exist_ok=True)

features_file = os.path.join(drive_save_path, 'all_features_EXPERT.csv') # Fichier EXPERT
processed_log = os.path.join(drive_save_path, 'processed_log.txt')
local_video_folder = 'videos_temp'
os.makedirs(local_video_folder, exist_ok=True)

# --- CHARGEMENT ROBUSTE ---
try:
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'Train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'TestInputSegments.csv'))
except:
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('TestInputSegments.csv')

def extract_path(v): return f"{v.split('/')[-1].split('_')[0]}/{v.split('/')[-1]}"
train_df['videos_path'] = train_df['videos'].apply(extract_path)
test_df['videos_path'] = test_df['videos'].apply(extract_path)
train_df['dataset'] = 'train'; test_df['dataset'] = 'test'
full_df = pd.concat([train_df, test_df], ignore_index=True)

# --- FILTRAGE ET PRIORISATION ---
def get_done():
    if not os.path.exists(processed_log): return set()
    with open(processed_log, 'r') as f: return set(l.strip() for l in f)

processed = get_done()
remaining = full_df[~full_df['videos_path'].isin(processed)]

# On sépare Test et Train
test_remaining = remaining[remaining['dataset'] == 'test']
train_remaining = remaining[remaining['dataset'] == 'train']

print(f"📊 Reste à traiter : {len(test_remaining)} Tests (URGENT) | {len(train_remaining)} Trains")

# STRATÉGIE : Si il reste du Test, on ne fait QUE ça. Sinon, on fait du Train.
if len(test_remaining) > 0:
    print("⚠️ MODE PRIORITÉ : Traitement du Test Input en cours...")
    videos_to_process = test_remaining # Pas de mélange, on veut finir les séries
else:
    print("✅ Test Input fini ! On traite le Train pour améliorer le modèle.")
    videos_to_process = train_remaining.sample(frac=1, random_state=42)

BATCH_LIMIT = 5000
count = 0

print(f"🚀 Démarrage du batch ({len(videos_to_process)} vidéos)...")

for index, row in tqdm(videos_to_process.iterrows(), total=len(videos_to_process)):
    if count >= BATCH_LIMIT: break

    path = row['videos_path']
    local = os.path.join(local_video_folder, os.path.basename(path))

    try:
        # Téléchargement
        blob = client.bucket(bucket_name).blob(path)
        blob.download_to_filename(local)

        # Extraction
        feats = extract_features_from_video(local)

        if feats:
            feats.update({
                'videos_path': path, 'time_segment_id': row['time_segment_id'],
                'view_label': row['view_label'], 'date': row['date'],
                'signaling': row.get('signaling', 'unknown'),
                'video_time': row.get('video_time', ''), 'dataset': row['dataset']
            })
            # Sauvegarde directe
            pd.DataFrame([feats]).to_csv(features_file, mode='a', header=not os.path.exists(features_file), index=False)

        # Marquage comme fait
        with open(processed_log, 'a') as f: f.write(f"{path}\n")
        if os.path.exists(local): os.remove(local)
        count += 1

    except Exception as e:
        # On marque quand même comme fait (pour éviter boucle 404)
        with open(processed_log, 'a') as f: f.write(f"{path}\n")
        if os.path.exists(local): os.remove(local)
        continue

print("🏁 Batch terminé.")
