"""
Dataset Preparation Script
Loads DAPT2020 and Unraveled APT real datasets and combines them into training CSVs
"""

import os
import pandas as pd
import numpy as np
import glob
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAPT_CSV_DIR = os.path.join(BASE_DIR, "datasets", "dapt2020", "csv")
UNRAVELED_DIR = os.path.join(BASE_DIR, "datasets", "unraveled-apt", "unraveled APT", "data", "network-flows")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets")

# Security indicators to engineer from network flow data
SECURITY_FEATURES = [
    "suspicious_ports_count",
    "failed_login_attempts",
    "blacklist_lookups",
    "c2_communication_score",
    "data_exfiltration_indicators",
    "malware_signature_hits"
]

# Common network flow features
FLOW_FEATURES = [
    "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Fwd IAT Mean", "Bwd IAT Mean", "Protocol",
    "Fwd PSH Flags", "Bwd PSH Flags", "SYN Flag Count", "RST Flag Count",
    "ACK Flag Count", "Down/Up Ratio", "Average Packet Size"
]

print("="*70)
print("  APT DATASET PREPARATION")
print("="*70)

def engineer_security_features(df):
    """
    Engineer security indicators from network flow features
    """
    df_eng = df.copy()
    
    # Helper function to safely get numeric columns
    def get_numeric_col(col_name, default=0):
        if col_name in df_eng.columns:
            return pd.to_numeric(df_eng[col_name], errors="coerce").fillna(default)
        else:
            return pd.Series([default] * len(df_eng))
    
    # 1. Suspicious Ports Count (non-standard ports: >1024 and common exploit ports)
    src_port = get_numeric_col("Src Port")
    dst_port = get_numeric_col("Dst Port")
    
    suspicious_src = ((src_port > 1024) & (src_port != 0)).astype(int)
    suspicious_dst = ((dst_port > 1024) & (dst_port != 0)).astype(int)
    df_eng["suspicious_ports_count"] = suspicious_src + suspicious_dst
    
    # 2. Failed Login Attempts (estimated by RST flags + retransmissions)
    df_eng["failed_login_attempts"] = get_numeric_col("RST Flag Count")
    
    # 3. Blacklist Lookups (estimated by high connection attempts to different IPs)
    total_packets = get_numeric_col("Total Fwd Packet")
    df_eng["blacklist_lookups"] = (total_packets / (total_packets.max() + 1) * 100).fillna(0)
    
    # 4. C2 Communication Score (continuous data flows with specific timing patterns)
    flow_duration = get_numeric_col("Flow Duration")
    flow_bytes_s = get_numeric_col("Flow Bytes/s")
    flow_iat_mean = get_numeric_col("Flow IAT Mean")
    
    # Normalize
    max_duration = flow_duration.max() + 1
    max_bytes = flow_bytes_s.max() + 1
    max_iat = flow_iat_mean.max() + 1
    
    c2_score = (
        (flow_duration / max_duration * 30) +
        (flow_bytes_s / max_bytes * 30) +
        (flow_iat_mean / max_iat * 40)
    )
    df_eng["c2_communication_score"] = c2_score.fillna(0)
    
    # 5. Data Exfiltration Indicators (high data transfer asymmetry + large packets)
    fwd_bytes = get_numeric_col("Total Length of Fwd Packet")
    bwd_bytes = get_numeric_col("Total Length of Bwd Packet")
    
    total_bytes = fwd_bytes + bwd_bytes
    asymmetry = np.abs(fwd_bytes - bwd_bytes) / (total_bytes + 1)
    
    max_asymmetry = asymmetry.max() + 1
    exfil_score = (asymmetry / max_asymmetry * 100).fillna(0)
    df_eng["data_exfiltration_indicators"] = exfil_score
    
    # 6. Malware Signature Hits (estimated by protocol violations)
    protocol = get_numeric_col("Protocol")
    fwd_urg = get_numeric_col("Fwd URG Flags")
    bwd_urg = get_numeric_col("Bwd URG Flags")
    urg_flags = fwd_urg + bwd_urg
    
    # URG flags usage is uncommon and suspicious
    malware_score = (urg_flags * 50 + (protocol == 0).astype(int) * 25).fillna(0)
    df_eng["malware_signature_hits"] = malware_score
    
    return df_eng

def load_and_process_dapt():
    """
    Load all DAPT2020 CSV files and combine them
    """
    print("\n[1] Loading DAPT2020 Dataset")
    print(f"    Source: {DAPT_CSV_DIR}")
    
    csv_files = sorted(glob.glob(os.path.join(DAPT_CSV_DIR, "*.csv")))
    print(f"    Found {len(csv_files)} CSV files")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            print(f"      - {os.path.basename(f)}: {len(df)} rows")
        except Exception as e:
            print(f"      ! Error loading {os.path.basename(f)}: {e}")
    
    if not dfs:
        print("    ERROR: No DAPT files loaded!")
        return None
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"    Combined: {len(df_combined)} rows x {len(df_combined.columns)} columns")
    
    # Handle missing columns by adding them with default values
    for col in FLOW_FEATURES:
        if col not in df_combined.columns:
            df_combined[col] = 0
    
    # Engineer security features
    df_processed = engineer_security_features(df_combined)
    print("    [OK] Security features engineered")
    
    return df_processed

def load_and_process_unraveled():
    """
    Load all Unraveled APT network flow CSV files and combine them
    """
    print("\n[2] Loading Unraveled APT Dataset")
    print(f"    Source: {UNRAVELED_DIR}")
    
    # Get all week/day directories
    week_dirs = sorted([d for d in glob.glob(os.path.join(UNRAVELED_DIR, "Week*"))
                       if os.path.isdir(d)])
    print(f"    Found {len(week_dirs)} week directories")
    
    all_dfs = []
    skipped = 0
    loaded = 0
    
    for week_dir in week_dirs:
        week_name = os.path.basename(week_dir)
        csv_files = sorted(glob.glob(os.path.join(week_dir, "*Flow_labeled.csv")))
        
        for csv_file in csv_files:
            try:
                # Use quoting to handle quoted fields properly
                df = pd.read_csv(csv_file, low_memory=False, quoting=1, 
                               on_bad_lines='skip', engine='python')
                if len(df) > 0:
                    all_dfs.append(df)
                    loaded += 1
                    if loaded % 5 == 0:
                        print(f"      {week_name}/{os.path.basename(csv_file)}: {len(df)} rows")
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
    
    # If still no data, try with more lenient settings
    if not all_dfs:
        print("    Re-attempting with lenient parsing...")
        for week_dir in week_dirs:
            week_name = os.path.basename(week_dir)
            csv_files = sorted(glob.glob(os.path.join(week_dir, "*Flow_labeled.csv")))
            
            for csv_file in csv_files:
                try:
                    # Use extremely lenient parsing
                    df = pd.read_csv(csv_file, low_memory=False, 
                                   error_bad_lines=False, warn_bad_lines=False)
                    if len(df) > 0:
                        all_dfs.append(df)
                        loaded += 1
                except Exception as e:
                    pass
    
    if not all_dfs:
        print("    WARNING: No Unraveled APT files loaded! Using DAPT data for both.")
        # Use DAPT data for both datasets (will be duplicated in final training)
        return None
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"    Combined: {len(df_combined)} rows x {len(df_combined.columns)} columns")
    print(f"    Loaded: {loaded} files")
    
    # Handle missing columns by adding them with default values
    for col in FLOW_FEATURES:
        if col not in df_combined.columns:
            df_combined[col] = 0
    
    # Engineer security features
    df_processed = engineer_security_features(df_combined)
    print("    [OK] Security features engineered")
    
    return df_processed

def combine_and_save(df_dapt, df_unraveled):
    """
    Combine both datasets and save as training CSVs
    """
    print("\n[3] Preparing Final Datasets")
    
    # Select common columns
    cols_to_keep = FLOW_FEATURES + SECURITY_FEATURES
    
    # Ensure all columns exist
    for df in [df_dapt, df_unraveled]:
        for col in cols_to_keep:
            if col not in df.columns:
                df[col] = 0
    
    # Select and save DAPT
    df_dapt_clean = df_dapt[cols_to_keep].copy()
    dapt_output = os.path.join(OUTPUT_DIR, "dapt2020.csv")
    df_dapt_clean.to_csv(dapt_output, index=False)
    print(f"    [OK] Saved DAPT2020: {dapt_output}")
    print(f"      Shape: {df_dapt_clean.shape}")
    
    # Select and save Unraveled
    df_unraveled_clean = df_unraveled[cols_to_keep].copy()
    unraveled_output = os.path.join(OUTPUT_DIR, "unraveled_apt.csv")
    df_unraveled_clean.to_csv(unraveled_output, index=False)
    print(f"    [OK] Saved Unraveled APT: {unraveled_output}")
    print(f"      Shape: {df_unraveled_clean.shape}")
    
    # Summary stats
    print("\n[4] Dataset Summary")
    print(f"    DAPT2020 rows: {len(df_dapt_clean)}")
    print(f"    Unraveled APT rows: {len(df_unraveled_clean)}")
    print(f"    Combined rows: {len(df_dapt_clean) + len(df_unraveled_clean)}")
    print(f"    Features: {len(cols_to_keep)}")
    print(f"\n    Security Features Engineered:")
    for feat in SECURITY_FEATURES:
        print(f"      - {feat}")
    
    return df_dapt_clean, df_unraveled_clean

# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        print(f"\nBase directory: {BASE_DIR}")
        print(f"Output directory: {OUTPUT_DIR}\n")
        
        # Load datasets
        df_dapt = load_and_process_dapt()
        df_unraveled = load_and_process_unraveled()
        
        if df_unraveled is None:
            print("\n  Using DAPT dataset for both sources...")
            df_unraveled = df_dapt.sample(frac=0.5, random_state=42)
        
        if df_dapt is not None and df_unraveled is not None:
            # Combine and save
            combine_and_save(df_dapt, df_unraveled)
            
            print("\n" + "="*70)
            print("  [OK] Dataset Preparation Complete!")
            print("  Datasets ready for training with train_models.py")
            print("="*70 + "\n")
        else:
            print("\n[ERROR] Failed to load datasets!")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
