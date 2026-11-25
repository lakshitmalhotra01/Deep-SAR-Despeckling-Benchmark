"""
unet_autoencoder_visuals.py
===========================
Generates visual summary (PNG table & bar chart) for the UNet‑Autoencoder model on real SAR data.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Output directory (same as other results)
OUT_DIR = Path('d:/image video project/results/real_sar_evaluation/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per‑channel metrics (from the evaluation)
metrics = {
    'S1_VV': {'ENL': 2.15, 'EPI': 0.9514, 'SSI': 0.8213, 'SMPI': 1.69},
    'S1_VH': {'ENL': 1.62, 'EPI': 0.9518, 'SSI': 0.8074, 'SMPI': 1.66},
    'UAVSAR_HH': {'ENL': 2.38, 'EPI': 0.9400, 'SSI': 0.7532, 'SMPI': 1.99},
    'UAVSAR_HV': {'ENL': 1.98, 'EPI': 0.9296, 'SSI': 0.6844, 'SMPI': 2.38},
    'UAVSAR_VH': {'ENL': 1.89, 'EPI': 0.9286, 'SSI': 0.6955, 'SMPI': 2.55},
    'UAVSAR_VV': {'ENL': 2.70, 'EPI': 0.9382, 'SSI': 0.7470, 'SMPI': 2.23},
}

# ---------- Table PNG ----------
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
cols = ['Channel', 'ENL', 'EPI', 'SSI', 'SMPI']
rows = []
for ch, vals in metrics.items():
    rows.append([ch, f"{vals['ENL']:.2f}", f"{vals['EPI']:.4f}", f"{vals['SSI']:.4f}", f"{vals['SMPI']:.2f}"])

table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
# Header style
for j in range(len(cols)):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', weight='bold')
# Row shading
for i in range(1, len(rows)+1):
    bg = '#f9f9f9' if i%2 else '#ffffff'
    for j in range(len(cols)):
        table[(i, j)].set_facecolor(bg)
plt.title('UNet‑Autoencoder – Real SAR Metrics (per channel)', fontsize=12, weight='bold')
plt.tight_layout()
table_path = OUT_DIR / 'unet_autoencoder_metrics_table.png'
plt.savefig(str(table_path), dpi=150, bbox_inches='tight')
plt.close()

# ---------- Bar chart ----------
metrics_names = ['ENL', 'EPI', 'SSI', 'SMPI']
channels = list(metrics.keys())

x = np.arange(len(channels))
width = 0.18
fig, ax = plt.subplots(figsize=(14,6))
for i, mname in enumerate(metrics_names):
    vals = [metrics[ch][mname] for ch in channels]
    ax.bar(x + i*width - 1.5*width, vals, width, label=mname)
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=30, ha='right')
ax.set_ylabel('Metric value')
ax.set_title('UNet‑Autoencoder – Metric comparison across channels')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
bar_path = OUT_DIR / 'unet_autoencoder_bar_chart.png'
plt.savefig(str(bar_path), dpi=150, bbox_inches='tight')
plt.close()

print(f"Table saved: {table_path}")
print(f"Bar chart saved: {bar_path}")
