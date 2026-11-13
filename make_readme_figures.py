"""Generate README figures: a key-findings infographic and a model ROC-AUC heatmap.

Run from the repository root:  python make_readme_figures.py
Outputs: figures/key_findings.png, figures/model_auc_heatmap.png
"""
import os
import matplotlib.pyplot as plt
import logging
from matplotlib.patches import Rectangle

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

os.makedirs('figures', exist_ok=True)

# Palette (light -> dark)
MIST = '#E6C1C9'
SKY = '#BCA0BB'
DEEP = '#8D80AD'
BLUE = '#5863A2'        # highlight
NAVY = '#1F4C96'        # text and darkest fill
BG = '#FFFFFF'

BLACK = NAVY          # text and dark fills
BLUE_MID = SKY        # secondary bars
BLUE_LIGHT = SKY
GREY = DEEP           # small captions
GREY_LIGHT = MIST     # bar tracks
TILE_EDGE = MIST

plt.rcParams['font.family'] = ['Helvetica Neue', 'Arial', 'Inter', 'DejaVu Sans']


# ---------------------------------------------------------------- helpers
def tile(fig, rect, title, source=None):
    """Draw a tile frame with a title; return an axes covering the tile body."""
    x, y, w, h = rect
    frame = fig.add_axes(rect)
    frame.set_xlim(0, 1)
    frame.set_ylim(0, 1)
    frame.axis('off')
    frame.add_patch(Rectangle((0, 0), 1, 1, facecolor=BG, edgecolor=TILE_EDGE, lw=1.2))
    frame.text(0.06, 0.92, title, fontsize=14.4, fontweight='bold', color=BLACK, va='top')
    if source:
        frame.text(0.94, 0.04, source, fontsize=9.4, color=GREY, ha='right', va='bottom')
    return frame


# ---------------------------------------------------------------- infographic
fig = plt.figure(figsize=(12, 6.6), facecolor=BG)

W, H = 0.315, 0.47
xs = [0.015, 0.3425, 0.67]
ys = [0.515, 0.02]

# 1. Data scale
ax = tile(fig, (xs[0], ys[0], W, H), 'Events analyzed')
ax.text(0.06, 0.50, '42.4M', fontsize=49.7, color=BLUE, va='center')
ax.text(0.06, 0.22, '3.0M users   ·   9.2M sessions\n573K cart sessions modeled',
        fontsize=11.9, color=BLACK, va='center', linespacing=1.6)

# 2. Event mix
ax = tile(fig, (xs[1], ys[0], W, H), 'Event mix')
mix = [('view', 96.1, BLACK), ('cart', 2.2, BLUE), ('purchase', 1.7, BLUE_LIGHT)]
y0 = 0.62
for i, (name, pct, color) in enumerate(mix):
    yy = y0 - i * 0.19
    ax.add_patch(Rectangle((0.06, yy), 0.50, 0.12, facecolor=GREY_LIGHT, edgecolor='none'))
    ax.add_patch(Rectangle((0.06, yy), 0.50 * max(pct, 2.5) / 100, 0.12,
                           facecolor=color, edgecolor='none'))
    ax.text(0.60, yy + 0.06, f'{name}  {pct}%', fontsize=12.5, color=BLACK, va='center')

# 3. Purchases that skip the cart (donut)
ax = tile(fig, (xs[2], ys[0], W, H), 'Purchases that skip the cart')
skip = 53.6
donut = fig.add_axes((xs[2] + 0.02, ys[0] + 0.08, 0.13, 0.29))
donut.set_aspect('equal')
donut.axis('off')
donut.pie([skip, 100 - skip], colors=[BLUE, BLACK], startangle=90, counterclock=False,
          wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=1.5))
ax.text(0.53, 0.52, f'{skip}%', fontsize=34.5, color=BLUE, va='center')
ax.text(0.53, 0.33, 'of purchasing sessions\nhave no add-to-cart', fontsize=10.8,
        color=BLACK, va='center', linespacing=1.4)
ax.text(0.94, 0.04, 'Cart funnels miss most purchases', fontsize=9.4, color=GREY,
        ha='right', va='bottom')

# 4. ROC-AUC heatmap (model x feature set)
ax = tile(fig, (xs[0], ys[1], W, H), 'ROC-AUC by model and features')
auc = [[0.545, 0.696], [0.625, 0.730]]
rows = ['Logistic\nregression', 'LightGBM']
cols = ['Session', '+ History']
vmin, vmax = 0.50, 0.75
cw, ch = 0.30, 0.27
gx, gy = 0.33, 0.12
for i in range(2):
    for j in range(2):
        v = auc[i][j]
        # Map the four values to palette steps by rank (low -> high)
        rank = sorted(sum(auc, [])).index(v)
        color = [SKY, DEEP, BLUE, NAVY][rank]
        t = rank / 3
        x0 = gx + j * (cw + 0.02)
        y0c = gy + (1 - i) * (ch + 0.02)
        ax.add_patch(Rectangle((x0, y0c), cw, ch, facecolor=color, edgecolor='none'))
        ax.text(x0 + cw / 2, y0c + ch / 2, f'{v:.3f}', fontsize=16.2, fontweight='bold',
                color='white' if rank >= 1 else BLACK, ha='center', va='center')
    ax.text(gx - 0.03, gy + (1 - i) * (ch + 0.02) + ch / 2, rows[i], fontsize=10.6,
            color=BLACK, ha='right', va='center', linespacing=1.2)
for j in range(2):
    ax.text(gx + j * (cw + 0.02) + cw / 2, gy + 2 * (ch + 0.02) + 0.01, cols[j],
            fontsize=10.6, color=BLACK, ha='center', va='bottom')

# 5. Precision of top-20% targeting
ax = tile(fig, (xs[1], ys[1], W, H), 'Riskiest 20% of carts')
ax.text(0.06, 0.52, '72%', fontsize=49.7, color=BLUE, va='center')
ax.text(0.06, 0.24, 'are actually abandoned,\nvs. 48% across all carts (1.5× lift)',
        fontsize=11.9, color=BLACK, va='center', linespacing=1.5)

# 6. Abandonment rate by targeting depth
ax = tile(fig, (xs[2], ys[1], W, H), 'Abandonment rate by targeting')
bars = [('Top 10', 76.1), ('Top 20', 72.3), ('Top 30', 69.7), ('Top 50', 64.8), ('All', 48.0)]
bx0, bw, gap, base, top = 0.07, 0.14, 0.04, 0.16, 0.58
for k, (lab, v) in enumerate(bars):
    x0 = bx0 + k * (bw + gap)
    hgt = (v / 100) * top
    color = MIST if lab == 'All' else (BLUE if lab == 'Top 20' else SKY)
    ax.add_patch(Rectangle((x0, base), bw, hgt, facecolor=color, edgecolor='none'))
    ax.text(x0 + bw / 2, base + hgt + 0.02, f'{v:.0f}%', fontsize=13.8, color=BLACK,
            ha='center', va='bottom')
    ax.text(x0 + bw / 2, base - 0.03, lab, fontsize=11.5, color=BLACK, ha='center', va='top')

fig.savefig('figures/key_findings.png', dpi=200, facecolor=BG)
plt.close(fig)


# ---------------------------------------------------------------- results heatmap
models = ['Logistic regression', 'LightGBM (no category)', 'LightGBM',
          'Logistic regression + history', 'LightGBM + history']
metrics = ['ROC-AUC', 'PR-AUC', 'Accuracy']
values = [
    [0.545, 0.550, 0.538],
    [0.617, 0.629, 0.580],
    [0.625, 0.634, 0.588],
    [0.696, 0.725, 0.640],
    [0.730, 0.751, 0.667],
]
labels = [
    ['0.545', '0.550', '53.8%'],
    ['0.617', '0.629', '58.0%'],
    ['0.625', '0.634', '58.8%'],
    ['0.696', '0.725', '64.0%'],
    ['0.730', '0.751', '66.7%'],
]


import numpy as np
vals = np.array(values)
# Discrete palette: within each metric (column), rank the 5 models and map
# rank 0..4 to the 5 palette colors exactly (no interpolation).
steps = [MIST, SKY, DEEP, BLUE, NAVY]
ranks = vals.argsort(axis=0).argsort(axis=0)
rgb = np.array([[[int(steps[r][k:k + 2], 16) / 255 for k in (1, 3, 5)]
                 for r in row] for row in ranks])

fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=BG)
ax.imshow(rgb, aspect='auto', interpolation='nearest')
ax.set_xticks(range(len(metrics)), metrics, fontsize=12)
ax.set_yticks(range(len(models)), models, fontsize=12)
ax.xaxis.tick_top()
ax.tick_params(length=0, labelcolor=BLACK)
for s_ in ax.spines.values():
    s_.set_visible(False)
for i in range(len(models)):
    for j in range(len(metrics)):
        ax.text(j, i, labels[i][j], ha='center', va='center', fontsize=13,
                fontweight='bold' if i == len(models) - 1 else 'normal',
                color='white' if ranks[i, j] >= 2 else BLACK)
ax.set_xticks([x - 0.5 for x in range(1, len(metrics))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
ax.grid(which='minor', color='white', linewidth=3)
ax.tick_params(which='minor', length=0)
fig.tight_layout()
fig.savefig('figures/model_auc_heatmap.png', dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)

print('Saved figures/key_findings.png and figures/model_auc_heatmap.png')
