![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![PyTorch](https://img.shields.io/badge/Vision-Faster%20R--CNN-red)
![Zindi](https://img.shields.io/badge/Zindi-Private%20LB%200.694-green)

🇬🇧 English | [🇫🇷 Français](README.fr.md)

# Barbados Traffic Analysis Challenge — "Traffic Engineer" Solution

**Author:** Medz1511
**Submission ID:** oxBsY8BS
**Final Filename:** `submission_PHYSICS_ENGINEER.csv`
**Private Leaderboard Score:** 0.693830175 · **Public Leaderboard Score:** 0.60507212

---

## 1. Overview and Objectives
The Barbados Ministry of Transport and Works wants to anticipate congestion at the **Norman Niles roundabout**. Four cameras film its entrances and exits; each ~1-minute segment is labelled `free flowing`, `light delay`, `moderate delay` or `heavy delay`.

The task: **predict congestion 5 minutes into the future** under real-time constraints — 15 minutes of video as input, a 2-minute processing embargo, then a 5-minute prediction window. No future data may be used to predict the past.

My approach turns raw video into **traffic-physics features** (vehicle counts, speed, density, flow, jam intensity) and feeds them to a seed-averaged LightGBM, followed by temporal smoothing and probability calibration.

## 2. Architecture Diagram

```text
[GCS bucket brb-traffic — 18,716 videos, ~134 GB]
       |
       v
[Video Feature Extraction]  --> src/extract_features.py (GPU)
   |-- Faster R-CNN ResNet50-FPN v2 (COCO) on 5 frames @ 320x180
   |-- Farneback optical flow (speed)
       |
       v
[input/all_features_EXPERT.csv]  (one row per video)
       |
       v
[Feature Engineering]  --> src/main.py (CPU)
   (cleaning, cross-camera stats, traffic physics, cyclic time, lags, scaling)
       |
       v
[LightGBM x 10 seeds]  --> averaged class probabilities
       |
       v
[Post-Processing]  --> temporal smoothing (60/40) + calibration
       |
       v
[output/submission_PHYSICS_ENGINEER.csv]
```

## 3. ETL Process (Extract, Transform, Load)

### Extract
- **Labels & segments:** `Train.csv` (16,076 rows), `TestInputSegments.csv`, `SampleSubmission.csv` (Zindi).
- **Videos:** downloaded one by one from the `brb-traffic` GCS bucket (re-encoded, ~134 GB) and deleted right after processing.
- The full >500 GB bucket (`brb-traffic-full`) contains the **same 18,716 videos** in original quality. It was deliberately not used: frames are resized to 320x180 before detection, so the extra quality would be lost.

### Transform (per video)
- 5 frames evenly spaced over the video, resized to 320x180 (speed/accuracy trade-off on a Colab GPU).
- **Detection:** Faster R-CNN (COCO pretrained, threshold 0.4) → cars, buses, trucks, pedestrians.
- **Speed:** Farneback optical flow between consecutive frames → mean and std of motion magnitude.
- **Derived:** density, queue-length proxy, congestion index = vehicles / (speed + 0.1).

### Load
- Each video's features are **appended immediately** to `input/all_features_EXPERT.csv`; processed paths are logged in `processed_log.txt`, so extraction **resumes** after a Colab disconnection.
- Test videos are processed **first** (priority), then training videos.
- Coverage: **18,532 / 18,716** videos (99%).

## 4. Data Modeling

### Feature Engineering (39 features)
- **Cleaning:** "ghost densities" — density and congestion index forced to 0 when no car is detected.
- **Cross-camera stats** per time segment (all 4 cameras): mean/sum density, mean speed, max congestion.
- **Traffic physics:**
  - `phys_flow` = mean density × mean speed (flow rate)
  - `phys_jam` = mean density / (mean speed + 1) (jam intensity)
- **Cyclic time:** `hour_sin`, `hour_cos`.
- **History:** lags 1–3 per camera on counts, speed, density and physics features.
- **Scaling:** `RobustScaler` (robust to detection outliers).

### Target
`congestion_enter_rating` **shifted by −3 segments** per camera: features at time *t* predict congestion at *t+3*.

### Model Details
- **Algorithm:** `LGBMClassifier`, averaged over **10 seeds** (42, 2025, 777, 101, 888, 999, 123, 456, 1000, 555).
- **Hyperparameters:** `n_estimators=3500`, `learning_rate=0.012`, `num_leaves=35`, `colsample_bytree=0.7`.
- **Class weights:** free flowing 1.0 · light delay 4.2 · moderate delay 3.1 · heavy delay 4.5 — the metric is 70% Macro-F1, so minority classes matter as much as the majority.

## 5. Inference Strategy
1. **Seed averaging:** mean of the 10 models' probabilities.
2. **Temporal smoothing:** `0.6 × raw + 0.4 × centred rolling mean (window 3)` per camera, so isolated spikes are dampened.
3. **Calibration:** probabilities multiplied by free 0.95 · light 1.15 · moderate 1.05 · heavy 1.0, then argmax.
4. **Mapping to submission IDs:** direct match on `(segment, camera)`, otherwise the closest previous prediction from the same camera (max 15 segments back), otherwise `free flowing`.

## 6. Run Time
| Step | Hardware | Time |
|---|---|---|
| `src/extract_features.py` | Colab GPU | several hours (~1–2 s per video, resumable) |
| `src/main.py` | Standard CPU | a few minutes |

## 7. Performance Metrics
- **Metric:** 70% Macro-F1 + 30% Accuracy.
- **Public Leaderboard:** 0.60507212
- **Private Leaderboard:** 0.693830175

**Lesson learned.** This submission scored only 0.605 on the public leaderboard (50% of the test set), below other versions of mine (up to 0.638), so I did not select it for final ranking. On the private leaderboard it turned out to be my best (0.694). Since then I select final models on the robustness of the approach, not on the public score.

## 8. Error Handling
- **Resumable extraction:** every processed video is logged; failed downloads (404) are logged too, to avoid infinite retries.
- **Missing predictions:** fallback to the closest previous prediction of the same camera, then to `free flowing`.
- **Division by zero:** `+1` / `+0.1` added to all speed denominators.

## 9. Project Structure

```text
zindi-barbados-traffic/
├── input/
│   ├── Train.csv
│   ├── TestInputSegments.csv
│   ├── SampleSubmission.csv
│   └── all_features_EXPERT.csv      # extracted video features
├── output/                          # submission is written here
├── notebooks/
│   └── PHYSICS_ENGINEER_pipeline.ipynb
├── src/
│   ├── extract_features.py          # videos -> all_features_EXPERT.csv
│   └── main.py                      # features -> submission
├── requirements.txt
├── README.md
└── README.fr.md
```

## 10. How to Run

```bash
pip install -r requirements.txt

# Optional: re-extract features from videos (GPU + GCS access required)
gcloud auth application-default login
python src/extract_features.py      # run again until "0 Tests | 0 Trains"

# Train and predict
python src/main.py
```
The submission is written to `output/submission_PHYSICS_ENGINEER.csv`.
Expected distribution (880 rows): free flowing 550 · moderate delay 165 · heavy delay 105 · light delay 60.

## 11. Conclusion
Heavy techniques — stacking (LightGBM / XGBoost / CatBoost), SMOTE, pseudo-labeling, Optuna — never beat a well-fed LightGBM. What moved the score was **domain knowledge**: cleaning detector artefacts, encoding traffic physics (flow, jam intensity), sharing information across the 4 cameras, and smoothing predictions over time.

**Next steps:** process the 184 videos of the `added/` folder, use multi-object tracking (e.g. ByteTrack) to count real entries/exits, and predict `congestion_exit_rating`.

## License
Code: [MIT](https://choosealicense.com/licenses/mit/) · Data: Zindi, [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
