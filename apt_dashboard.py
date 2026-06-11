"""
APT Kill Chain Detection Dashboard (Clean Version)
====================================================
Analyzes network traffic and classifies APT stages using trained ML models.

Models: 3 Neural Networks (Plain, HyperParam Tuning, PSO Optimized)
Input: CSV with network traffic features (20 columns)
Output: Kill chain stage predictions and visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gradio as gr
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# CONSTANTS
# ============================================================

KILL_CHAIN_STAGES = {
    0: "Reconnaissance",
    1: "Initial Access",
    2: "Command & Control",
    3: "Data Exfiltration",
    4: "Benign"
}

# These 6 columns are used ONLY for label assignment in training
# They should NOT be passed to the models
LABEL_PROXY_COLS = [
    'suspicious_ports_count',
    'failed_login_attempts',
    'blacklist_lookups',
    'c2_communication_score',
    'data_exfiltration_indicators',
    'malware_signature_hits'
]

# Original 20 features that models were trained on
ORIGINAL_FEATURES = [
    'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
    'Total Length of Bwd Packet', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Protocol', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'Down/Up Ratio',
    'Average Packet Size'
]

# Global variables for storing uploaded data
uploaded_data = None
uploaded_info = None

# ============================================================
# LABEL ASSIGNMENT (for visualization only)
# ============================================================

def assign_labels(df):
    """Assign kill chain labels based on proxy columns with improved distribution"""
    sc = df["suspicious_ports_count"].values.astype(float)
    fl = df["failed_login_attempts"].values.astype(float)
    bl = df["blacklist_lookups"].values.astype(float)
    c2 = df["c2_communication_score"].values.astype(float)
    ex = df["data_exfiltration_indicators"].values.astype(float)
    mal = df["malware_signature_hits"].values.astype(float)

    labels = np.full(len(df), 4, dtype=int)  # default: Benign

    # Use threshold-based classification for better balance
    
    # Stage 3: Data Exfiltration (highest priority) - high exfil and moderate C2
    mask_exfil = (ex >= 70) & (c2 >= 40)
    labels[mask_exfil] = 3

    # Stage 2: Command & Control - very high C2 score (not already exfil)
    mask_c2 = (c2 >= 75) & ~mask_exfil
    labels[mask_c2] = 2

    # Stage 1: Initial Access - high failed logins AND blacklist lookups (not already classified)
    mask_init = (fl >= 70) & (bl >= 70) & ~mask_exfil & ~mask_c2
    labels[mask_init] = 1

    # Stage 0: Reconnaissance - high suspicious ports AND malware signatures (not already classified)
    mask_recon = (sc >= 70) & (mal >= 70) & ~mask_exfil & ~mask_c2 & ~mask_init
    labels[mask_recon] = 0

    return labels


def add_proxy_columns(df):
    """Add proxy columns if missing (for label assignment visualization)"""
    df = df.copy()
    
    if 'label' in df.columns:
        # If labels already exist, derive proxy columns from them for consistency
        labels = df['label'].values
        
        for col in LABEL_PROXY_COLS:
            if col not in df.columns:
                # Generate proxy columns based on actual labels for better consistency
                if col == 'suspicious_ports_count':
                    df[col] = np.where(labels == 0, np.random.uniform(85, 100, len(df)),
                                     np.where(labels == 4, np.random.uniform(0, 20, len(df)),
                                             np.random.uniform(20, 60, len(df))))
                elif col == 'failed_login_attempts':
                    df[col] = np.where(labels == 1, np.random.uniform(80, 100, len(df)),
                                     np.where(labels == 4, np.random.uniform(0, 10, len(df)),
                                             np.random.uniform(10, 40, len(df))))
                elif col == 'blacklist_lookups':
                    df[col] = np.where(labels == 1, np.random.uniform(80, 100, len(df)),
                                     np.where(labels == 4, np.random.uniform(0, 15, len(df)),
                                             np.random.uniform(15, 50, len(df))))
                elif col == 'c2_communication_score':
                    df[col] = np.where(labels == 2, np.random.uniform(85, 100, len(df)),
                                     np.where(labels == 3, np.random.uniform(40, 85, len(df)),
                                             np.where(labels == 4, np.random.uniform(0, 10, len(df)),
                                                     np.random.uniform(10, 50, len(df)))))
                elif col == 'data_exfiltration_indicators':
                    df[col] = np.where(labels == 3, np.random.uniform(85, 100, len(df)),
                                     np.where(labels == 2, np.random.uniform(20, 50, len(df)),
                                             np.where(labels == 4, np.random.uniform(0, 5, len(df)),
                                                     np.random.uniform(5, 40, len(df)))))
                elif col == 'malware_signature_hits':
                    df[col] = np.where(labels == 0, np.random.uniform(75, 100, len(df)),
                                     np.where(labels == 3, np.random.uniform(60, 90, len(df)),
                                             np.where(labels == 4, np.random.uniform(0, 10, len(df)),
                                                     np.random.uniform(20, 70, len(df)))))
    else:
        # Original behavior for data without labels
        for col in LABEL_PROXY_COLS:
            if col not in df.columns:
                df[col] = np.random.uniform(0, 100, len(df))
    
    return df


# ============================================================
# PREPROCESSING & PREDICTION
# ============================================================

def load_models():
    """Load the 3 MLP models with PCA transformers and scalers"""
    models_data = {}
    
    model_defs = [
        ('models/model_plain.pkl', 'Model 1: No Hyperparameters'),
        ('models/model_ht.pkl', 'Model 2: HyperParam Tuning'),
        ('models/model_pso.pkl', 'Model 3: PSO Optimized'),
    ]
    
    for filepath, name in model_defs:
        try:
            if os.path.exists(filepath):
                loaded_data = joblib.load(filepath)
                
                # Handle both dict format (with PCA/scaler) and old format (model only)
                if isinstance(loaded_data, dict):
                    models_data[name] = {
                        'model': loaded_data.get('model'),
                        'pca': loaded_data.get('pca'),
                        'scaler': loaded_data.get('scaler')
                    }
                else:
                    # Fallback: try to load PCA separately
                    pca_path = filepath.replace('model_', 'pca_').replace('.pkl', '_pca.pkl')
                    if os.path.exists(pca_path):
                        pca = joblib.load(pca_path)
                    else:
                        pca = None
                    models_data[name] = {'model': loaded_data, 'pca': pca, 'scaler': None}
                    
                print(f"✓ Loaded: {name}")
            else:
                print(f"✗ Missing: {filepath}")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
    
    if not models_data:
        print("ERROR: No models found!")
    
    return models_data


def preprocess_features(X, pca_model, scaler=None):
    """Apply scaling + PCA correctly to match training features"""
    try:
        # Step 1: Scaling
        if scaler is not None:
            X_scaled = scaler.transform(X)
        elif os.path.exists('models/scaler.pkl'):
            scaler = joblib.load('models/scaler.pkl')
            X_scaled = scaler.transform(X)
        else:
            print("WARNING: No scaler found, using raw features")
            X_scaled = X

        # Step 2: PCA (THIS FIXES FEATURE MISMATCH)
        if pca_model is not None:
            X_pca = pca_model.transform(X_scaled)
            return X_pca
        else:
            return X_scaled

    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        return None

def get_original_features(df):
    """Ensure input matches training features EXACTLY"""

    feature_path = os.path.join("models", "feature_names.pkl")

    if not os.path.exists(feature_path):
        raise ValueError("feature_names.pkl NOT FOUND in models folder")

    feat_cols = joblib.load(feature_path)

    print("✅ Expected features:", len(feat_cols))
    print("📥 Input columns:", len(df.columns))

    # Add missing columns
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns in correct order
    df = df[feat_cols]

    X = df.fillna(0).values.astype(np.float32)

    print("🚀 Final shape to model:", X.shape)

    return X


# ============================================================
# TAB 1: ANALYZE UPLOADED CSV
# ============================================================

def upload_and_analyze(csv_file):
    """Tab 1: Upload CSV and show predictions"""
    global uploaded_data, uploaded_info
    
    try:
        if csv_file is None:
            return "ERROR: No file uploaded", None
        
        # Load CSV
        df = pd.read_csv(csv_file)
        
        # Add proxy columns for label assignment
        df = add_proxy_columns(df)
        
        # PRESERVE existing labels from CSV, only assign if missing
        if 'label' not in df.columns:
            df['label'] = assign_labels(df)
        
        # Store for other tabs
        uploaded_data = df.copy()
        uploaded_info = f"Loaded: {len(df)} samples, {df.shape[1]} columns"
        
        # Extract original 20 features
        try:
            X = get_original_features(df)
        except ValueError as e:
            return f"ERROR: {str(e)}", None
        
        # Load models (each with its own PCA)
        models_data = load_models()
        if not models_data:
            return "ERROR: No models found", None
        
        # Make predictions with each model using its own PCA
        predictions = {}
        for name, data in models_data.items():
            try:
                X_pca = preprocess_features(X, data['pca'], data.get('scaler'))
                if X_pca is None:
                    return f"ERROR: Preprocessing failed for {name}", None
                predictions[name] = data['model'].predict(X_pca)
            except Exception as e:
                return f"ERROR in {name}: {str(e)}", None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 7))
        
        stage_names = [KILL_CHAIN_STAGES[i] for i in range(5)]
        x = np.arange(len(stage_names))
        width = 0.25
        colors = ['steelblue', 'cyan', 'lime']
        
        for idx, (name, preds) in enumerate(predictions.items()):
            counts = np.array([np.sum(preds == i) for i in range(5)])
            bars = ax.bar(x + idx*width, counts, width, label=name,
                         color=colors[idx], edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Kill Chain Stages', fontweight='bold', fontsize=14)
        ax.set_ylabel('Count', fontweight='bold', fontsize=14)
        ax.set_title(f'Kill Chain Stage Distribution ({len(df)} traffic flows)', fontweight='bold', fontsize=16)
        ax.set_xticks(x + width)
        ax.set_xticklabels(stage_names, fontsize=12)
        ax.legend(fontsize=12, frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        
        msg = f"SUCCESS: Analyzed {len(df)} flows for {len(models_data)} models."
        return msg, fig
    
    except Exception as e:
        return f"ERROR: {str(e)}", None




# ============================================================
# TAB 2: CONFUSION MATRICES
# ============================================================

def show_confusion_matrices():
    """Tab 2: Generate confusion matrices using uploaded data with actual labels"""
    plt.close('all')
    
    try:
        # Use uploaded data
        if uploaded_data is None:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "ERROR: No uploaded data. Please upload a CSV first.", ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        df_test = uploaded_data.copy()
        
        # Use existing label column if available, otherwise generate labels
        if 'label' not in df_test.columns:
            df_test = add_proxy_columns(df_test)
            df_test['label'] = assign_labels(df_test)
        
        # Balance the uploaded data by selecting equal samples from each class
        balanced_dfs = []
        for class_label in range(5):
            class_samples = df_test[df_test['label'] == class_label]
            if len(class_samples) > 0:
                # Take up to 80 samples from this class
                sampled = class_samples.sample(n=min(80, len(class_samples)), random_state=42)
                balanced_dfs.append(sampled)
        
        if balanced_dfs:
            df_test = pd.concat(balanced_dfs, ignore_index=True)
        
        try:
            X_test = get_original_features(df_test)
        except ValueError as e:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"ERROR: {str(e)}", ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        y_test = df_test['label'].values
        
        # Load models (each with its own PCA)
        models_data = load_models()
        if not models_data:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "ERROR: No models found", ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            return fig
        
        # Generate confusion matrices with clean, professional design
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Confusion Matrices', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        for ax, (name, data) in zip(axes, models_data.items()):
            X_pca = preprocess_features(X_test, data['pca'], data.get('scaler'))
            if X_pca is None:
                ax.text(0.5, 0.5, f"ERROR: Preprocessing failed", ha='center', va='center', 
                       fontsize=12, color='red', fontweight='bold')
                ax.axis('off')
                continue
            
            y_pred = data['model'].predict(X_pca)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
            
            # Professional heatmap with muted colors and small text
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax,
                       cbar=True, cbar_kws={'label': 'Count', 'shrink': 0.9, 'pad': 0.02},
                       xticklabels=[0, 1, 2, 3, 4],
                       yticklabels=[0, 1, 2, 3, 4],
                       annot_kws={'size': 13, 'weight': 'normal', 'color': 'black'},
                       linewidths=0.5, linecolor='gray', square=True,
                       vmin=0)
            
            # Simple title without accuracy
            ax.set_title(f'{name}', fontweight='normal', fontsize=14, pad=10)
            
            ax.set_xlabel('Predicted', fontweight='normal', fontsize=11, labelpad=5)
            ax.set_ylabel('True', fontweight='normal', fontsize=11, labelpad=5)
            ax.tick_params(axis='both', labelsize=10, length=4, width=1)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"ERROR: {str(e)}", ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig


def show_confusion_matrices_button():
    return show_confusion_matrices()


# ============================================================
# TAB 3: METRICS TABLE
# ============================================================

def show_metrics():
    """Tab 3: Display metrics table for uploaded data"""
    try:
        # Use uploaded data with original labels preserved
        if uploaded_data is None:
            return pd.DataFrame({'Status': ["ERROR: No uploaded data."]})
        
        df = uploaded_data.copy()
        # Preserve labels from CSV, don't reassign
        if 'label' not in df.columns:
            df = add_proxy_columns(df)
            df['label'] = assign_labels(df)
        
        try:
            X = get_original_features(df)
        except ValueError as e:
            return pd.DataFrame({'Status': [f"ERROR: {str(e)}"]})
        
        y = df['label'].values
        
        # Load models (each with its own PCA)
        models_data = load_models()
        if not models_data:
            return pd.DataFrame({'Status': ["ERROR: No models found"]})
        
        # Compute metrics
        metrics_list = []
        
        for name, data in models_data.items():
            X_pca = preprocess_features(X, data['pca'], data.get('scaler'))
            if X_pca is None:
                metrics_list.append({
                    'Model': name,
                    'Accuracy': 'ERROR',
                    'Precision': 'ERROR',
                    'Recall': 'ERROR',
                    'F1 Score': 'ERROR'
                })
                continue
            
            y_pred = data['model'].predict(X_pca)
            
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            metrics_list.append({
                'Model': name,
                'Accuracy': f'{acc:.4f}',
                'Precision': f'{prec:.4f}',
                'Recall': f'{rec:.4f}',
                'F1 Score': f'{f1:.4f}'
            })
        
        return pd.DataFrame(metrics_list)
    
    except Exception as e:
        return pd.DataFrame({'Status': [f"ERROR: {str(e)}"]})


def show_metrics_button():
    return show_metrics()


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="APT Kill Chain Detection Dashboard") as demo:
    gr.Markdown("# 🛡️ APT Kill Chain Detection Dashboard")
    gr.Markdown("Analyze network traffic and detect Advanced Persistent Threat stages")
    
    with gr.Tabs():
        # TAB 1: UPLOAD & ANALYZE
        with gr.Tab("📊 Upload & Analyze"):
            with gr.Row():
                with gr.Column():
                    csv_input = gr.File(label="Upload CSV", file_types=['.csv'])
                    analyze_btn = gr.Button("Analyze", scale=1)
            
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                chart_output = gr.Plot(label="Kill Chain Distribution")
            
            analyze_btn.click(
                fn=upload_and_analyze,
                inputs=[csv_input],
                outputs=[status_output, chart_output]
            )
        
        # TAB 2: CONFUSION MATRICES
        with gr.Tab("🎯 Confusion Matrices"):
            with gr.Row():
                matrices_btn = gr.Button("Show Confusion Matrices")
            
            with gr.Row():
                matrices_output = gr.Plot(label="Confusion Matrices")
            
            matrices_btn.click(
                fn=show_confusion_matrices_button,
                outputs=matrices_output
            )
        
        # TAB 3: METRICS
        with gr.Tab("📈 Metrics"):
            with gr.Row():
                metrics_btn = gr.Button("Show Model Metrics")
            
            with gr.Row():
                metrics_output = gr.Dataframe(label="Model Metrics")
            
            metrics_btn.click(
                fn=show_metrics_button,
                outputs=metrics_output
            )


if __name__ == "__main__":
    print("Starting APT Kill Chain Detection Dashboard...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7867,
        show_error=True,
        share=False,
        quiet=False
    )
