![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![PyTorch](https://img.shields.io/badge/Vision-Faster%20R--CNN-red)
![Zindi](https://img.shields.io/badge/Zindi-Private%20LB%200.694-green)

[🇬🇧 English](README.md) | 🇫🇷 Français

# Barbados Traffic Analysis Challenge — Solution « L'Ingénieur Trafic »

**Auteur :** Medz1511
**ID de soumission :** oxBsY8BS
**Fichier final :** `submission_PHYSICS_ENGINEER.csv`
**Score privé :** 0.693830175 · **Score public :** 0.60507212

---

## 1. Présentation et objectifs
Le Ministère des Transports de la Barbade veut anticiper la congestion du **rond-point Norman Niles**. Quatre caméras filment ses entrées et sorties ; chaque segment d'environ 1 minute est étiqueté `free flowing`, `light delay`, `moderate delay` ou `heavy delay`.

L'objectif : **prédire la congestion 5 minutes dans le futur** sous contrainte temps réel — 15 minutes de vidéo en entrée, 2 minutes d'embargo de traitement, puis une fenêtre de prédiction de 5 minutes. Aucune donnée future ne peut servir à prédire le passé.

Mon approche transforme la vidéo brute en **features de physique du trafic** (comptages, vitesse, densité, débit, intensité de bouchon), traitées par un LightGBM moyenné sur plusieurs seeds, puis par un lissage temporel et une calibration des probabilités.

## 2. Schéma d'architecture

```text
[Bucket GCS brb-traffic — 18 716 vidéos, ~134 Go]
       |
       v
[Extraction des features vidéo]  --> src/extract_features.py (GPU)
   |-- Faster R-CNN ResNet50-FPN v2 (COCO) sur 5 images en 320x180
   |-- Flux optique Farneback (vitesse)
       |
       v
[input/all_features_EXPERT.csv]  (une ligne par vidéo)
       |
       v
[Feature Engineering]  --> src/main.py (CPU)
   (nettoyage, stats multi-caméras, physique du trafic, heure cyclique, lags, normalisation)
       |
       v
[LightGBM x 10 seeds]  --> probabilités moyennées
       |
       v
[Post-traitement]  --> lissage temporel (60/40) + calibration
       |
       v
[output/submission_PHYSICS_ENGINEER.csv]
```

## 3. Processus ETL (Extract, Transform, Load)

### Extract
- **Labels et segments :** `Train.csv` (16 076 lignes), `TestInputSegments.csv`, `SampleSubmission.csv` (Zindi).
- **Vidéos :** téléchargées une par une depuis le bucket GCS `brb-traffic` (version ré-encodée, ~134 Go), puis supprimées après traitement.
- Le bucket complet (`brb-traffic-full`, >500 Go) contient **les mêmes 18 716 vidéos** en qualité d'origine. Il n'a volontairement pas été utilisé : les images sont réduites en 320x180 avant la détection, le gain de qualité aurait été perdu.

### Transform (par vidéo)
- 5 images réparties sur la vidéo, réduites en 320x180 (compromis vitesse / précision sur GPU Colab).
- **Détection :** Faster R-CNN (pré-entraîné COCO, seuil 0.4) → voitures, bus, camions, piétons.
- **Vitesse :** flux optique Farneback entre images consécutives → moyenne et écart-type du mouvement.
- **Dérivés :** densité, proxy de longueur de file, indice de congestion = véhicules / (vitesse + 0.1).

### Load
- Les features de chaque vidéo sont **ajoutées immédiatement** à `input/all_features_EXPERT.csv` ; les vidéos traitées sont notées dans `processed_log.txt`, ce qui permet de **reprendre** l'extraction après une déconnexion de Colab.
- Les vidéos de test sont traitées **en priorité**, puis celles d'entraînement.
- Couverture : **18 532 / 18 716** vidéos (99 %).

## 4. Modélisation

### Feature Engineering (39 features)
- **Nettoyage :** « densités fantômes » — densité et indice de congestion mis à 0 quand aucune voiture n'est détectée.
- **Stats multi-caméras** par segment (les 4 caméras) : densité moyenne/totale, vitesse moyenne, congestion max.
- **Physique du trafic :**
  - `phys_flow` = densité moyenne × vitesse moyenne (débit)
  - `phys_jam` = densité moyenne / (vitesse moyenne + 1) (intensité du bouchon)
- **Heure cyclique :** `hour_sin`, `hour_cos`.
- **Historique :** lags 1 à 3 par caméra sur les comptages, la vitesse, la densité et les features physiques.
- **Normalisation :** `RobustScaler` (robuste aux valeurs aberrantes du détecteur).

### Cible
`congestion_enter_rating` **décalée de −3 segments** par caméra : les features à l'instant *t* prédisent la congestion à *t+3*.

### Modèle
- **Algorithme :** `LGBMClassifier`, moyenné sur **10 seeds** (42, 2025, 777, 101, 888, 999, 123, 456, 1000, 555).
- **Hyperparamètres :** `n_estimators=3500`, `learning_rate=0.012`, `num_leaves=35`, `colsample_bytree=0.7`.
- **Poids de classes :** free flowing 1.0 · light delay 4.2 · moderate delay 3.1 · heavy delay 4.5 — la métrique est à 70 % un Macro-F1, les classes minoritaires comptent autant que la majoritaire.

## 5. Stratégie d'inférence
1. **Moyenne des seeds :** probabilités moyennées sur les 10 modèles.
2. **Lissage temporel :** `0.6 × brut + 0.4 × moyenne glissante centrée (fenêtre 3)` par caméra, pour atténuer les pics isolés.
3. **Calibration :** probabilités multipliées par free 0.95 · light 1.15 · moderate 1.05 · heavy 1.0, puis argmax.
4. **Correspondance avec les ID de soumission :** correspondance directe sur `(segment, caméra)`, sinon la prédiction précédente la plus proche de la même caméra (15 segments max), sinon `free flowing`.

## 6. Temps d'exécution
| Étape | Matériel | Durée |
|---|---|---|
| `src/extract_features.py` | GPU Colab | plusieurs heures (~1–2 s par vidéo, reprise possible) |
| `src/main.py` | CPU standard | quelques minutes |

## 7. Performances
- **Métrique :** 70 % Macro-F1 + 30 % Accuracy.
- **Leaderboard public :** 0.60507212
- **Leaderboard privé :** 0.693830175

**Leçon retenue.** Cette soumission ne faisait que 0.605 sur le leaderboard public (50 % du test), moins bien que d'autres de mes versions (jusqu'à 0.638) : je ne l'avais donc pas sélectionnée pour le classement final. Sur le leaderboard privé, c'était pourtant ma meilleure (0.694). Depuis, je choisis mes modèles finaux sur la robustesse de l'approche, pas sur le score public.

## 8. Gestion des erreurs
- **Extraction avec reprise :** chaque vidéo traitée est journalisée ; les téléchargements en échec (404) aussi, pour éviter les boucles infinies.
- **Prédictions manquantes :** repli sur la prédiction précédente la plus proche de la même caméra, puis sur `free flowing`.
- **Division par zéro :** `+1` / `+0.1` ajoutés à tous les dénominateurs de vitesse.

## 9. Structure du projet

```text
zindi-barbados-traffic/
├── input/
│   ├── Train.csv
│   ├── TestInputSegments.csv
│   ├── SampleSubmission.csv
│   └── all_features_EXPERT.csv      # features vidéo extraites
├── output/                          # la soumission est écrite ici
├── notebooks/
│   └── PHYSICS_ENGINEER_pipeline.ipynb
├── src/
│   ├── extract_features.py          # vidéos -> all_features_EXPERT.csv
│   └── main.py                      # features -> soumission
├── requirements.txt
├── README.md
└── README.fr.md
```

## 10. Exécution

```bash
pip install -r requirements.txt

# Optionnel : ré-extraire les features depuis les vidéos (GPU + accès GCS)
gcloud auth application-default login
python src/extract_features.py      # relancer jusqu'à "0 Tests | 0 Trains"

# Entraînement et prédiction
python src/main.py
```
La soumission est écrite dans `output/submission_PHYSICS_ENGINEER.csv`.
Répartition attendue (880 lignes) : free flowing 550 · moderate delay 165 · heavy delay 105 · light delay 60.

## 11. Conclusion
Les techniques lourdes — stacking (LightGBM / XGBoost / CatBoost), SMOTE, pseudo-labeling, Optuna — n'ont jamais battu un LightGBM bien alimenté. Ce qui a fait progresser le score, c'est la **connaissance métier** : nettoyer les artefacts du détecteur, encoder la physique du trafic (débit, intensité de bouchon), croiser l'information des 4 caméras et lisser les prédictions dans le temps.

**Pistes :** traiter les 184 vidéos du dossier `added/`, utiliser du tracking multi-objets (ex. ByteTrack) pour compter les entrées/sorties réelles, et prédire `congestion_exit_rating`.

## Licence
Code : [MIT](https://choosealicense.com/licenses/mit/) · Données : Zindi, [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
