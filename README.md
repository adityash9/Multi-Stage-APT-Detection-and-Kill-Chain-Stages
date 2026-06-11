# Multi-Stage-APT-Detection-and-Kill-Chain-Stages
# APT Kill Chain Detection System


### Project Structure
```
APT_Final/
├── datasets/
│   ├── dapt2020.csv          ← DAPT2020 dataset (1000 rows)
│   └── unraveled_apt.csv     ← Unraveled APT dataset (400 rows)
├── models/                   ← Created after training
│   ├── model_plain.pkl
│   ├── model_ht.pkl
│   ├── model_pso.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── ...
├── train_models.py           ← Step 1: Run this
├── apt_dashboard.py          ← Step 2: Run this
├── requirements.txt
└── README.md
```

---

### Kill Chain Stages Detected
| Class | Stage | Description |
|-------|-------|-------------|
| 0 | Reconnaissance | Port scanning, network mapping |
| 1 | Initial Access | Login attempts, exploits |
| 2 | Command & Control | C2 beacons, malware comms |
| 3 | Data Exfiltration | Outbound data transfers |
| 4 | Benign | Normal traffic |

---

### 3 Models Compared
| Model | Expected Accuracy | Method |
|-------|------------------|--------|
| No Tuning | Low (~60-70%) | Fixed small network, high LR |
| HyperParam Tuning | Medium (~75-82%) | Randomized search, 15 combos |
| PSO Optimized | High (~85-92%) | Particle Swarm Optimization |

### PSO Configuration
- **Particles**: 10
- **Iterations**: 15
- **c1 = 1.5** (cognitive acceleration coefficient)
- **c2 = 1.5** (social acceleration coefficient)
- **Inertia w**: 0.9 → 0.4 (linear decay)
- Velocity: `v(t+1) = w·v(t) + c1·r1·(pbest-x) + c2·r2·(gbest-x)`

---

### Step-by-Step Setup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train All 3 Models
```bash
python train_models.py
```
Training takes ~5-15 minutes. You will see:
```
FINAL RESULTS SUMMARY
  Model                            Acc    Prec     Rec      F1
  1. No Tuning    (Low)         0.6XXX  0.6XXX  0.6XXX  0.6XXX
  2. HyperParam   (Medium)      0.7XXX  0.7XXX  0.7XXX  0.7XXX
  3. PSO Optimized (High)       0.8XXX  0.8XXX  0.8XXX  0.8XXX
```

#### 3. Launch Dashboard
```bash
python apt_dashboard.py
```
Open browser: **http://127.0.0.1:7860**

---

### Dashboard Tabs
1. **Model Comparison** — Click "Load All Charts" to see metrics table, bar chart, and per-class F1 heatmap
2. **Traffic Analysis** — Upload any CSV of network flows (or click "Generate Sample CSV")
3. **Confusion Matrix** — After uploading a labeled CSV in Tab 2, click here to see confusion matrices

---

### Datasets Used
- **DAPT2020**: 1000 network flow records, 80 features
- **Unraveled APT**: 400 network flow records, 81 features
- Combined training set: 1400 samples on 74 shared features
- Labels assigned automatically using domain-knowledge rules based on:
  - `suspicious_ports_count`, `failed_login_attempts`, `blacklist_lookups`
  - `c2_communication_score`, `data_exfiltration_indicators`, `malware_signature_hits`
 
## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- Gradio
- Particle Swarm Optimization (PSO)
- Principal Component Analysis (PCA)
- Machine Learning
- Cybersecurity
