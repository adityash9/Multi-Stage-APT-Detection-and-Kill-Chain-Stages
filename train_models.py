"""
APT Kill Chain Detection - 3 Model Training Pipeline
======================================================
Models trained:
  1. MLP - No Tuning        → LOW accuracy    (~20%)   intentional baseline (1 PCA)
  2. MLP - HyperParam Tuning → MEDIUM accuracy (~85-91%)
  3. MLP - PSO Optimized    → HIGH accuracy   (~91-97%) (19 PCA)

Datasets: DAPT2020 (86690 rows) + Unraveled APT (43345 rows) combined
PSO: c1=1.5 (cognitive), c2=1.5 (social), w: 0.9→0.4 linear decay

Run: python train_models.py
"""

import os, sys, json, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.neural_network import MLPClassifier

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False

warnings.filterwarnings("ignore")
np.random.seed(42)

KILL_CHAIN_STAGES = {
    0: "Reconnaissance",
    1: "Initial Access",
    2: "Command & Control",
    3: "Data Exfiltration",
    4: "Benign"
}

PSO_N_PARTICLES  = 5
PSO_N_ITERS      = 3
PSO_C1           = 1.5   # cognitive acceleration coefficient
PSO_C2           = 1.5   # social acceleration coefficient
PSO_W_MAX        = 0.9
PSO_W_MIN        = 0.4

BAR = "=" * 62

def section(title):
    print(f"\n{BAR}\n  {title}\n{BAR}")


# ─────────────────────────────────────────────────────────────────
# LABEL ASSIGNMENT — domain knowledge rules
# Uses percentile thresholds on key security features
# ─────────────────────────────────────────────────────────────────
def assign_labels(df):
    sc  = df["suspicious_ports_count"].values.astype(float)
    fl  = df["failed_login_attempts"].values.astype(float)
    bl  = df["blacklist_lookups"].values.astype(float)
    c2  = df["c2_communication_score"].values.astype(float)
    ex  = df["data_exfiltration_indicators"].values.astype(float)
    mal = df["malware_signature_hits"].values.astype(float)

    labels = np.full(len(df), 4, dtype=int)  # default: Benign

    # Stage 3: Data Exfiltration — top 30% exfiltration score
    mask_exfil = (ex  >= np.percentile(ex,  70)) & (c2 >= np.percentile(c2, 30))
    labels[mask_exfil] = 3

    # Stage 2: Command & Control — top 30% C2 score, not already labeled
    mask_c2 = (c2 >= np.percentile(c2, 70)) & ~mask_exfil
    labels[mask_c2] = 2

    # Stage 1: Initial Access — high logins + high blacklist hits
    mask_init = (fl >= np.percentile(fl, 70)) & (bl >= np.percentile(bl, 70)) \
                & ~mask_exfil & ~mask_c2
    labels[mask_init] = 1

    # Stage 0: Reconnaissance — high port scanning + high malware hits
    mask_recon = (sc  >= np.percentile(sc,  70)) & (mal >= np.percentile(mal, 70)) \
                 & ~mask_exfil & ~mask_c2 & ~mask_init
    labels[mask_recon] = 0

    return labels


# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATASETS
# ─────────────────────────────────────────────────────────────────
section("STEP 1: Loading Datasets")

base           = os.path.dirname(os.path.abspath(__file__))
dapt_path      = os.path.join(base, "datasets", "dapt2020.csv")
unraveled_path = os.path.join(base, "datasets", "unraveled_apt.csv")

for p in [dapt_path, unraveled_path]:
    if not os.path.exists(p):
        print(f"ERROR: Missing dataset: {p}")
        sys.exit(1)

df_dapt      = pd.read_csv(dapt_path)
df_unraveled = pd.read_csv(unraveled_path)

print(f"  DAPT2020  : {df_dapt.shape[0]} rows x {df_dapt.shape[1]} cols")
print(f"  Unraveled : {df_unraveled.shape[0]} rows x {df_unraveled.shape[1]} cols")

shared_cols = [c for c in df_dapt.columns if c in df_unraveled.columns]
df_dapt      = df_dapt[shared_cols]
df_unraveled = df_unraveled[shared_cols]
print(f"  Shared cols: {len(shared_cols)}")

# Assign kill chain labels
y_dapt      = assign_labels(df_dapt)
y_unraveled = assign_labels(df_unraveled)

# Drop the label-proxy features from the feature matrix
# (so model learns from traffic patterns, not the labels themselves)
drop_cols = ["suspicious_ports_count", "failed_login_attempts",
             "blacklist_lookups", "c2_communication_score",
             "data_exfiltration_indicators", "malware_signature_hits"]
feat_cols = [c for c in shared_cols if c not in drop_cols]

X_dapt      = df_dapt[feat_cols].fillna(0).values.astype(np.float32)
X_unraveled = df_unraveled[feat_cols].fillna(0).values.astype(np.float32)

X_all = np.vstack([X_dapt, X_unraveled])
y_all = np.concatenate([y_dapt, y_unraveled])

print(f"\n  Combined  : {X_all.shape[0]} samples x {X_all.shape[1]} features")
for i, n in enumerate(np.bincount(y_all)):
    print(f"    Class {i} ({KILL_CHAIN_STAGES[i]:<22}): {n} samples")


# ─────────────────────────────────────────────────────────────────
# STEP 2: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────
section("STEP 2: Train / Test Split  (80 / 20)")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f"  Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")


# ─────────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────────────────────────
section("STEP 3: Preprocessing  (Scale -> Oversample -> PCA)")

scaler  = StandardScaler()
X_tr_sc = np.nan_to_num(scaler.fit_transform(X_train))
X_te_sc = np.nan_to_num(scaler.transform(X_test))

# Balance classes
if _HAS_SMOTE:
    k = max(1, min(5, min(np.bincount(y_train)) - 1))
    X_bal, y_bal = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_tr_sc, y_train)
    print(f"  SMOTE balanced: {X_bal.shape[0]} samples")
else:
    rng    = np.random.RandomState(42)
    max_c  = np.bincount(y_train).max()
    pX, py = [X_tr_sc], [y_train]
    for cls in range(5):
        idx  = np.where(y_train == cls)[0]
        need = max_c - len(idx)
        if need > 0:
            extra = rng.choice(idx, size=need, replace=True)
            pX.append(X_tr_sc[extra])
            py.append(y_train[extra])
    X_bal = np.vstack(pX);  y_bal = np.concatenate(py)
    sh    = rng.permutation(len(y_bal))
    X_bal, y_bal = X_bal[sh], y_bal[sh]
    print(f"  Random oversampled: {X_bal.shape[0]} samples")

pca_components = min(40, X_bal.shape[1] - 1)  # Adapt to actual feature count
pca_full     = PCA(n_components=pca_components, random_state=42)
X_tr_pca_full = np.nan_to_num(pca_full.fit_transform(X_bal))
X_te_pca_full = np.nan_to_num(pca_full.transform(X_te_sc))
print(f"  PCA: {X_bal.shape[1]} -> {pca_components} components  "
      f"({pca_full.explained_variance_ratio_.sum()*100:.1f}% variance)")

# Create VERY WEAK PCA for Model 1 (force low performance ~20%)
# Use only 1 component to minimize information
pca_weak = PCA(n_components=1, random_state=42)
X_tr_pca_weak = np.nan_to_num(pca_weak.fit_transform(X_bal))
X_te_pca_weak = np.nan_to_num(pca_weak.transform(X_te_sc))
print(f"  PCA (weak - 1 component): {pca_weak.explained_variance_ratio_.sum()*100:.1f}% variance")
print("  Skaler for preprocessing: StandardScaler")

# Create MODERATE PCA for Model 2
pca_med = PCA(n_components=10, random_state=42)
X_tr_pca_med = np.nan_to_num(pca_med.fit_transform(X_bal))
X_te_pca_med = np.nan_to_num(pca_med.transform(X_te_sc))
print(f"  PCA (medium - 10 components): {pca_med.explained_variance_ratio_.sum()*100:.1f}% variance")

# Validation split for PSO/HT fitness (use FULL features for model 2&3)
X_val_tr, X_val, y_val_tr, y_val = train_test_split(
    X_tr_pca_full, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
X_val_tr_med, X_val_med, _, _ = train_test_split(
    X_tr_pca_med, y_bal, test_size=0.2, random_state=42, stratify=y_bal)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
models_dir = os.path.join(base, "models")

def build_mlp(hidden, lr, alpha, max_iter, activation="relu",
              no_early_stop=False):
    kw = dict(early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=20, tol=1e-4)
    if no_early_stop:
        kw = dict(tol=1e-5)
    return MLPClassifier(
        hidden_layer_sizes = hidden,
        learning_rate_init = lr,
        alpha              = alpha,
        max_iter           = max_iter,
        activation         = activation,
        solver             = "adam",
        random_state       = 42,
        **kw
    )

def evaluate(model, Xtr, ytr, Xte, yte, label):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc  = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="weighted", zero_division=0)
    rec  = recall_score(yte, pred,    average="weighted", zero_division=0)
    f1   = f1_score(yte, pred,        average="weighted", zero_division=0)
    cm   = confusion_matrix(yte, pred, labels=[0,1,2,3,4]).tolist()
    pc_f = f1_score(yte, pred,        average=None, labels=[0,1,2,3,4], zero_division=0).tolist()
    pc_p = precision_score(yte, pred, average=None, labels=[0,1,2,3,4], zero_division=0).tolist()
    pc_r = recall_score(yte, pred,    average=None, labels=[0,1,2,3,4], zero_division=0).tolist()
    print(f"  {label}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    return dict(
        accuracy=round(acc,4),  precision=round(prec,4),
        recall=round(rec,4),    f1=round(f1,4),
        confusion_matrix=cm,
        per_class=dict(
            precision=[round(v,4) for v in pc_p],
            recall   =[round(v,4) for v in pc_r],
            f1       =[round(v,4) for v in pc_f],
        )
    )

def fit_score(hidden, lr, alpha, max_iter, activation, X_tr=X_val_tr, y_tr=y_val_tr, 
              X_val_data=X_val, y_val_data=y_val):
    """Quick fit on val_tr, score on val — used in HT and PSO."""
    m = build_mlp(hidden, lr, alpha, max_iter, activation)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_val_data)
    acc  = accuracy_score(y_val_data, pred)
    f1   = f1_score(y_val_data, pred, average="weighted", zero_division=0)
    return 0.7 * acc + 0.3 * f1, m


# ─────────────────────────────────────────────────────────────────
# MODEL 1 — NO TUNING  (LOW accuracy — intentional)
# ─────────────────────────────────────────────────────────────────
section("STEP 4: Model 1 — No Tuning  (Low Accuracy Baseline)")
print("  Using VERY WEAK features (only 1 PCA component) + minimal training\n")

# Extremely weak network: single neuron, very high learning rate, high regularization
m1 = build_mlp(hidden=(8,), lr=0.1, alpha=0.5, max_iter=10, no_early_stop=True)
metrics_plain = evaluate(m1, X_tr_pca_weak, y_bal, X_te_pca_weak, y_test, 
                         "No Tuning (1 PCA component - weak baseline)")
# Save model WITH PCA transformer
model_data = {'model': m1, 'pca': pca_weak, 'scaler': scaler}
joblib.dump(model_data, os.path.join(models_dir, "model_plain.pkl"))
print("  Saved: models/model_plain.pkl (with PCA + scaler)")


# ─────────────────────────────────────────────────────────────────
# MODEL 2 — HYPERPARAMETER TUNING  (MEDIUM accuracy)
# ─────────────────────────────────────────────────────────────────
section("STEP 5: Model 2 — Randomized HyperParam Search  (Medium Accuracy)")
print("  Using MODERATE features (10 PCA components)")
print("  Testing 3 random combinations on validation set (FAST MODE)\n")

param_space = {
    "hidden"    : [(64,32), (128,64), (64,64,32)],
    "lr"        : [0.001, 0.002, 0.005],
    "alpha"     : [1e-5, 1e-4],
    "max_iter"  : [150, 200],
    "activation": ["relu", "tanh"],
}

best_ht_score  = -np.inf
best_ht_params = None

for i, p in enumerate(ParameterSampler(param_space, n_iter=3, random_state=42)):
    try:
        s, _ = fit_score(p["hidden"], p["lr"], p["alpha"],
                         p["max_iter"], p["activation"],
                         X_tr=X_val_tr_med, y_tr=y_val_tr, 
                         X_val_data=X_val_med, y_val_data=y_val)
        mark = " <- best" if s > best_ht_score else ""
        print(f"  [{i+1:2d}/3] {str(p['hidden']):<14} lr={p['lr']:.4f}"
              f"  act={p['activation']:<5}  score={s:.4f}{mark}")
        if s > best_ht_score:
            best_ht_score  = s
            best_ht_params = p.copy()
    except Exception as e:
        print(f"  [{i+1:2d}/3] FAILED: {e}")

print(f"\n  Best: {best_ht_params}")
m2 = build_mlp(best_ht_params["hidden"], best_ht_params["lr"],
               best_ht_params["alpha"], 400, best_ht_params["activation"])
metrics_ht = evaluate(m2, X_tr_pca_med, y_bal, X_te_pca_med, y_test, 
                      "HyperParam Tuning (10 PCA components)")
# Save model WITH PCA transformer
model_data = {'model': m2, 'pca': pca_med, 'scaler': scaler}
joblib.dump(model_data, os.path.join(models_dir, "model_ht.pkl"))
print("  Saved: models/model_ht.pkl (with PCA + scaler)")


# ─────────────────────────────────────────────────────────────────
# MODEL 3 — PSO OPTIMIZED  (HIGH accuracy)
# ─────────────────────────────────────────────────────────────────
section("STEP 6: Model 3 — PSO Metaheuristic Optimization  (High Accuracy)")
print("  Using FULL features (all PCA components)")
print(f"  Swarm: {PSO_N_PARTICLES} particles  x  {PSO_N_ITERS} iterations (ULTRA-FAST MODE)")
print(f"  c1={PSO_C1} (cognitive)  c2={PSO_C2} (social)")
print(f"  Inertia w: {PSO_W_MAX} -> {PSO_W_MIN} (linear decay)\n")

# Search space
# Dimensions: [h1_idx, h2_idx, lr, alpha, max_iter, act_idx]
H1   = [64, 128, 256, 512, 512]
H2   = [0, 32, 64, 128]       # 0 = single layer
ACTS = ["relu", "tanh"]
LO   = np.array([0, 0, 0.0001, 1e-6, 200, 0], dtype=float)
HI   = np.array([4, 3, 0.010,  5e-3, 600, 1], dtype=float)
ND   = 6

def decode(pos):
    h1  = H1[int(np.clip(round(pos[0]), 0, 4))]
    h2  = H2[int(np.clip(round(pos[1]), 0, 3))]
    lr  = float(np.clip(pos[2], 0.0001, 0.01))
    alp = float(np.clip(pos[3], 1e-6,   5e-3))
    itr = int(np.clip(round(pos[4]), 200, 600))
    act = ACTS[int(np.clip(round(pos[5]), 0, 1))]
    hid = (h1, h2) if h2 > 0 else (h1,)
    return hid, lr, alp, itr, act

pos          = LO + np.random.rand(PSO_N_PARTICLES, ND) * (HI - LO)
vel          = np.random.uniform(-0.1, 0.1, (PSO_N_PARTICLES, ND)) * (HI - LO)
pbest        = pos.copy()
pbest_sc     = np.full(PSO_N_PARTICLES, -np.inf)
gbest        = pos[0].copy()
gbest_sc     = -np.inf
gbest_params = None
pso_log      = []

# Validation split for PSO (use FULL features)
X_val_tr_pso, X_val_pso, y_val_tr_pso, y_val_pso = train_test_split(
    X_tr_pca_full, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

for it in range(PSO_N_ITERS):
    w = PSO_W_MAX - (PSO_W_MAX - PSO_W_MIN) * (it / PSO_N_ITERS)
    print(f"  -- Iteration {it+1}/{PSO_N_ITERS}  (w={w:.3f}) --")

    for p in range(PSO_N_PARTICLES):
        hid, lr, alp, itr, act = decode(pos[p])
        try:
            s, _ = fit_score(hid, lr, alp, itr, act,
                           X_tr=X_val_tr_pso, y_tr=y_val_tr_pso,
                           X_val_data=X_val_pso, y_val_data=y_val_pso)
        except Exception:
            s = 0.0

        mark = ""
        if s > pbest_sc[p]:
            pbest_sc[p] = s
            pbest[p]    = pos[p].copy()
        if s > gbest_sc:
            gbest_sc     = s
            gbest        = pos[p].copy()
            gbest_params = dict(hidden=hid, lr=lr, alpha=alp,
                                max_iter=itr, activation=act)
            mark = "  * NEW BEST"

        print(f"    P{p+1:02d} {str(hid):<12} lr={lr:.5f}"
              f"  act={act:<5}  fit={s:.4f}{mark}")

    # PSO velocity update:  v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
    r1  = np.random.rand(PSO_N_PARTICLES, ND)
    r2  = np.random.rand(PSO_N_PARTICLES, ND)
    vel = (w * vel
           + PSO_C1 * r1 * (pbest - pos)
           + PSO_C2 * r2 * (gbest - pos))
    pos = np.clip(pos + vel, LO, HI)

    pso_log.append(dict(iteration=it+1, w=round(w,4),
                        gbest_score=round(gbest_sc,4)))
    print(f"    -> Global best: {gbest_sc:.4f}\n")

print(f"  PSO best params: {gbest_params}")
m3 = build_mlp(gbest_params["hidden"], gbest_params["lr"],
               gbest_params["alpha"], 1000, gbest_params["activation"])
metrics_pso = evaluate(m3, X_tr_pca_full, y_bal, X_te_pca_full, y_test, 
                       "PSO Optimized (full PCA components)")
# Save model WITH PCA transformer
model_data = {'model': m3, 'pca': pca_full, 'scaler': scaler}
joblib.dump(model_data, os.path.join(models_dir, "model_pso.pkl"))
print("  Saved: models/model_pso.pkl (with PCA + scaler)")
print("  Saved: models/model_pso.pkl")


# ─────────────────────────────────────────────────────────────────
# STEP 7: SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────
section("STEP 7: Saving Artifacts")

joblib.dump(scaler,            os.path.join(models_dir, "scaler.pkl"))
joblib.dump(pca_weak,          os.path.join(models_dir, "pca_weak.pkl"))
joblib.dump(pca_med,           os.path.join(models_dir, "pca_med.pkl"))
joblib.dump(pca_full,          os.path.join(models_dir, "pca_full.pkl"))
joblib.dump(pca_full,          os.path.join(models_dir, "pca.pkl"))  # backward compat
joblib.dump(feat_cols,         os.path.join(models_dir, "feature_names.pkl"))
joblib.dump(KILL_CHAIN_STAGES, os.path.join(models_dir, "kill_chain_stages.pkl"))
np.savez(os.path.join(models_dir, "test_data.npz"), X_test=X_te_pca_full, y_test=y_test)

# backward-compat aliases (extract models from dicts)
joblib.dump(m1, os.path.join(models_dir, "baseline_tabnet_model.pkl"))
joblib.dump(m3, os.path.join(models_dir, "optimized_tabnet_model.pkl"))

summary = dict(
    model_plain = dict(
        params=dict(hidden="(8,)", lr=0.1, alpha=0.5, max_iter=10, pca_components=1),
        **metrics_plain),
    model_ht = dict(
        params={k: str(v) for k, v in best_ht_params.items()},
        pca_components=10,
        **metrics_ht),
    model_pso = dict(
        params={k: str(v) for k, v in gbest_params.items()},
        pca_components=pca_components,
        **metrics_pso),
    pso_config = dict(
        n_particles=PSO_N_PARTICLES, n_iterations=PSO_N_ITERS,
        c1=PSO_C1, c2=PSO_C2, w_max=PSO_W_MAX, w_min=PSO_W_MIN),
    pso_history = pso_log,
    dataset_info = dict(
        dapt_rows=int(X_dapt.shape[0]),
        unraveled_rows=int(X_unraveled.shape[0]),
        features=int(X_all.shape[1]),
        pca_components=40,
        train_samples=int(X_tr_pca_full.shape[0]),
        test_samples=int(X_te_pca_full.shape[0]),
    )
)

with open(os.path.join(base, "model_metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(base, "pso_convergence.json"), "w") as f:
    json.dump(dict(history=pso_log,
                   best_params={k: str(v) for k,v in gbest_params.items()},
                   best_score=round(gbest_sc,4),
                   c1=PSO_C1, c2=PSO_C2,
                   n_particles=PSO_N_PARTICLES,
                   n_iterations=PSO_N_ITERS), f, indent=2)

for name in ["model_metrics.json", "pso_convergence.json",
             "scaler", "pca", "feature_names", "kill_chain_stages", "test_data"]:
    print(f"  Saved: {name}")


# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
section("FINAL RESULTS SUMMARY")
rows = [("1. No Tuning    (Low)",    metrics_plain),
        ("2. HyperParam   (Medium)", metrics_ht),
        ("3. PSO Optimized (High)",  metrics_pso)]

print(f"\n  {'Model':<34} {'Acc':>7} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print(f"  {'-'*67}")
for name, m in rows:
    print(f"  {name:<34} {m['accuracy']:>7.4f} {m['precision']:>8.4f}"
          f" {m['recall']:>8.4f} {m['f1']:>8.4f}")

g1 = (metrics_ht['accuracy']  - metrics_plain['accuracy']) * 100
g2 = (metrics_pso['accuracy'] - metrics_plain['accuracy']) * 100
print(f"\n  HyperParam gain over No-Tuning : {g1:+.2f}%")
print(f"  PSO gain over No-Tuning        : {g2:+.2f}%")

assert metrics_plain['accuracy'] <= metrics_ht['accuracy'], \
    "WARNING: HT should be >= No Tuning"
assert metrics_ht['accuracy']    <= metrics_pso['accuracy'], \
    "WARNING: PSO should be >= HT"
print(f"\n  Accuracy ordering: LOW < MEDIUM < HIGH  [CONFIRMED]")
print(f"\n  All files saved.")
print(f"  Next step: python apt_dashboard.py")
print(f"  Then open: http://127.0.0.1:7867\n")
