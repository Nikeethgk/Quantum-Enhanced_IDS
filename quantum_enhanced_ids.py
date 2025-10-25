"""
Quantum-Enhanced IDS Simulator (single-file Streamlit app)

Save as: quantum_enhanced_ids.py
Run:    streamlit run quantum_enhanced_ids.py

Dependencies:
    pip install streamlit numpy pandas matplotlib

What it does (short):
- Synthesizes network flows with 'true' labels (malicious/benign)
- Generates a classical signature score and a quantum-inspired score
- Combines them into a detection score and classifies by threshold
- Displays metrics and multiple graphs (bar chart of outcomes, histogram,
  ROC-like curve, and timeline of alerts)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

st.set_page_config(page_title="Quantum-Enhanced IDS Simulator", layout="wide")

# ---------------------- Helper functions ----------------------
def quantum_measure(amplitudes):
    """
    Convert amplitudes to probabilities (|a|^2) and sample an index.
    For our scoring, we'll return a continuous 'quantum_score' in [0,1]
    by sampling a distribution derived from amplitudes.
    """
    amps = np.array(amplitudes, dtype=np.complex128)
    probs = np.abs(amps) ** 2
    probs = probs / probs.sum()
    # sample an index
    idx = np.random.choice(len(probs), p=probs)
    # map index to a score between 0 and 1 (uniform within partition)
    return (idx + np.random.random()) / len(probs)

def synthesize_flows(n_flows=500, malicious_ratio=0.2, seed=None):
    """
    Create a DataFrame of synthetic network flows with 'true_label' (1=malicious,0=benign)
    and some features we will use to build classical & quantum scores.
    """
    rng = np.random.default_rng(seed)
    n_mal = int(n_flows * malicious_ratio)
    labels = np.array([1]*n_mal + [0]*(n_flows - n_mal))
    rng.shuffle(labels)

    # Create simple features: packet_count, duration_s, unique_ports, payload_entropy
    packet_count = rng.poisson(20, size=n_flows) + (labels * rng.poisson(40, size=n_flows))
    duration_s = rng.exponential(scale=2.0, size=n_flows) + (labels * rng.exponential(scale=5.0, size=n_flows))
    unique_ports = rng.integers(1, 6, size=n_flows) + (labels * rng.integers(0, 6, size=n_flows))
    payload_entropy = np.clip(rng.normal(4.0, 1.0, size=n_flows) + (labels * rng.normal(1.2, 0.8, size=n_flows)), 0.1, 8.0)

    df = pd.DataFrame({
        'flow_id': np.arange(1, n_flows+1),
        'packet_count': packet_count,
        'duration_s': duration_s,
        'unique_ports': unique_ports,
        'payload_entropy': payload_entropy,
        'true_label': labels
    })

    # Simulated timestamp (seconds) across a 1-hour window
    df['timestamp_s'] = rng.integers(0, 3600, size=n_flows)
    df.sort_values('timestamp_s', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def classical_signature_score(row, signature_strength=0.5, rng=None):
    """
    A toy classical score: if payload_entropy > threshold or packet_count high, signature gives higher score.
    signature_strength scales how strongly signatures detect malicious flows.
    Returns value in [0,1].
    """
    if rng is None:
        rng = np.random
    score = 0.0
    # payload entropy suspiciousness
    score += min(1.0, max(0.0, (row['payload_entropy'] - 4.0) / 4.0))
    # packet_count suspiciousness
    score += min(1.0, (row['packet_count'] / 200.0))
    # unique_ports suspicious
    score += min(1.0, (row['unique_ports'] - 1) / 10.0)
    score = score / 3.0  # normalize to 0..1
    # add randomness and scale by signature strength
    score = np.clip(score * signature_strength + rng.random() * (1.0 - signature_strength) * 0.5, 0.0, 1.0)
    return score

def quantum_score_from_features(row, n_amplitudes=6, noise_scale=0.2):
    """
    Build a small amplitude vector from features and derive a quantum score in [0,1].
    The amplitude components are built so malicious flows tend to concentrate amplitude
    in higher-index entries which map to higher scores.
    """
    # base amplitudes from normalized features
    a1 = max(0.01, row['payload_entropy'] / 8.0)
    a2 = max(0.01, np.clip(row['packet_count'] / 200.0, 0.01, 1.0))
    a3 = max(0.01, np.clip(row['unique_ports'] / 10.0, 0.01, 1.0))
    base = np.array([a1, a2, a3])
    # expand into n_amplitudes sized array with small noise and progressive weighting
    amps = np.zeros(n_amplitudes, dtype=np.float64)
    for i in range(n_amplitudes):
        # favor higher i for higher suspicious features
        weight = 1.0 + (i / n_amplitudes) * (base.mean())
        amps[i] = weight * np.mean(base) + np.random.normal(0, noise_scale)
    amps = np.clip(amps, 0.01, None)
    return quantum_measure(amps)

def evaluate_detection(df, threshold=0.5):
    """
    Given df with 'detection_score', assign predicted label and compute basic confusion metrics.
    Returns dictionary of metrics and augmented df.
    """
    df = df.copy()
    df['pred_label'] = (df['detection_score'] >= threshold).astype(int)
    tp = int(((df['true_label'] == 1) & (df['pred_label'] == 1)).sum())
    tn = int(((df['true_label'] == 0) & (df['pred_label'] == 0)).sum())
    fp = int(((df['true_label'] == 0) & (df['pred_label'] == 1)).sum())
    fn = int(((df['true_label'] == 1) & (df['pred_label'] == 0)).sum())

    detection_rate = tp / max(1, (tp + fn))  # recall
    fpr = fp / max(1, (fp + tn))
    precision = tp / max(1, (tp + fp))

    metrics = {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Detection Rate (Recall)': detection_rate,
        'False Positive Rate': fpr,
        'Precision': precision
    }
    return df, metrics

def compute_roc_points(df, n_points=50):
    """
    Compute TPR/FPR pairs across thresholds for a simple ROC-like curve (no sklearn).
    """
    scores = np.linspace(0, 1, n_points)
    tpr = []
    fpr = []
    for thr in scores:
        _, m = evaluate_detection(df.assign(detection_score=df['detection_score']), threshold=thr)
        tpr.append(m['Detection Rate (Recall)'])
        fpr.append(m['False Positive Rate'])
    return scores, np.array(tpr), np.array(fpr)

# ---------------------- Streamlit UI ----------------------
st.title("⚛️ Quantum-Enhanced IDS Simulator")
st.markdown(
    "Simulate an IDS that combines a classical signature detector with a quantum-inspired "
    "scoring mechanism. Tweak parameters in the sidebar, run the simulation, and inspect graphs."
)

with st.sidebar.expander("Simulation Controls", expanded=True):
    n_flows = st.number_input("Number of flows", min_value=50, max_value=10000, value=1000, step=50)
    malicious_ratio = st.slider("Malicious flow ratio", 0.01, 0.5, 0.1, 0.01)
    signature_strength = st.slider("Classical signature strength", 0.0, 1.0, 0.6, 0.05)
    quantum_noise = st.slider("Quantum noise scale", 0.0, 1.0, 0.2, 0.05)
    combine_weight = st.slider("Quantum weight in final score", 0.0, 1.0, 0.5, 0.05)
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01)
    seed_val = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
    if seed_val == 0:
        seed_val = None

    run_button = st.button("Run simulation")

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:**\n- Increase signature strength to favor classical detection.\n- Increase quantum weight to let quantum score drive detections.\n- Tune threshold for desired tradeoff between TPR and FPR.")

if run_button:
    df = synthesize_flows(n_flows, malicious_ratio, seed=seed_val)
    # compute scores
    q_scores = []
    c_scores = []
    rng = np.random.default_rng(seed_val)
    for idx, row in df.iterrows():
        c = classical_signature_score(row, signature_strength=signature_strength, rng=rng)
        # temporarily set small RNG for quantum score reproducibility per row
        # note: quantum function uses global np.random for randomness; we keep simple
        q = quantum_score_from_features(row, noise_scale=quantum_noise)
        q_scores.append(q)
        c_scores.append(c)
    df['quantum_score'] = q_scores
    df['classical_score'] = c_scores

    # combine into detection_score; allow weighting between quantum and classical
    df['detection_score'] = np.clip(combine_weight * df['quantum_score'] + (1 - combine_weight) * df['classical_score'], 0.0, 1.0)

    # evaluate with threshold
    df_eval, metrics = evaluate_detection(df, threshold=threshold)

    # Display metrics
    st.header("Simulation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flows simulated", f"{len(df_eval)}")
    col2.metric("Malicious (true)", f"{int(df_eval['true_label'].sum())}")
    col3.metric("Detected (predicted)", f"{int(df_eval['pred_label'].sum())}")
    col4.metric("Detection Rate (Recall)", f"{metrics['Detection Rate (Recall)']:.2f}")

    # Confusion bar chart
    st.subheader("Confusion outcomes (TP/TN/FP/FN)")
    counts = {'TP': metrics['TP'], 'FP': metrics['FP'], 'FN': metrics['FN'], 'TN': metrics['TN']}
    fig1, ax1 = plt.subplots(figsize=(6,3))
    ax1.bar(counts.keys(), counts.values())
    ax1.set_ylabel("Flows")
    ax1.set_title("Confusion Matrix Counts")
    st.pyplot(fig1)

    # Detection score histograms for malicious vs benign
    st.subheader("Detection score distribution")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(df_eval[df_eval['true_label']==1]['detection_score'], bins=30, alpha=0.7, label='Malicious')
    ax2.hist(df_eval[df_eval['true_label']==0]['detection_score'], bins=30, alpha=0.7, label='Benign')
    ax2.axvline(threshold, color='red', linestyle='--', label=f"Threshold {threshold:.2f}")
    ax2.set_xlabel("Detection score")
    ax2.set_ylabel("Count")
    ax2.legend()
    st.pyplot(fig2)

    # ROC-like curve
    st.subheader("ROC-like curve (TPR vs FPR)")
    scores, tpr, fpr = compute_roc_points(df_eval, n_points=100)
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.plot(fpr, tpr, marker='.', linewidth=1)
    ax3.plot([0,1],[0,1], linestyle='--', color='grey', linewidth=0.8)
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate (Recall)")
    ax3.set_title("ROC-like curve")
    st.pyplot(fig3)

    # Timeline of alerts (predicted malicious over time)
    st.subheader("Timeline: Predicted malicious alerts per minute")
    df_eval['minute'] = (df_eval['timestamp_s'] // 60).astype(int)
    timeline = df_eval[df_eval['pred_label']==1].groupby('minute').size()
    fig4, ax4 = plt.subplots(figsize=(10,3))
    ax4.plot(timeline.index, timeline.values, marker='o')
    ax4.set_xlabel("Minute (simulated hour)")
    ax4.set_ylabel("Alerts (predicted malicious)")
    ax4.set_title("Alerts per minute")
    st.pyplot(fig4)

    # Show metrics table and allow CSV download
    st.subheader("Metrics & Sample Data")
    st.write(pd.DataFrame([metrics]).T.rename(columns={0:'value'}))
    st.dataframe(df_eval.sample(min(200, len(df_eval))).reset_index(drop=True))

    csv = df_eval.to_csv(index=False)
    st.download_button("Download results (CSV)", data=csv, file_name="quantum_ids_results.csv", mime="text/csv")

    st.success("Simulation finished — tweak parameters and run again!")

else:
    st.info("Set parameters in the left panel and click **Run simulation** to start.")
