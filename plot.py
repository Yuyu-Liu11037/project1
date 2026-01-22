import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 1. Create synthetic ICD data
# -----------------------------
np.random.seed(0)

depth_levels = [1, 2, 3, 4]
codes_per_depth = 15          # number of ICD codes per depth
dim_euclid = 8                # Euclidean embedding dimension
dim_lorentz_space = 8         # spatial dimension for Lorentz (total d+1)

icd_codes = []
depth_dict = {}
euclid_emb_list = []
lorentz_emb_list = []

# Euclidean: per-depth mean is random in [9, 11], and each depth has slightly different variance
euclid_mean_by_depth = {d: np.random.uniform(9.0, 11.0) for d in depth_levels}
euclid_std_by_depth = {d: np.random.uniform(0.30, 0.55) for d in depth_levels}

for d in depth_levels:
    for i in range(codes_per_depth):
        # fake ICD code (you can replace with real ones later)
        code = f"D{d}{i:02d}"
        icd_codes.append(code)
        depth_dict[code] = d

        # --- target radii with noise ---
        # Keep the synthetic ranges consistent with the fixed y-limits in the plot:
        # - Euclidean radius: mean is random in [9, 11], larger variance
        # - Lorentz radius: increasing with depth, with increasing slope (convex)
        #
        # Euclidean: no monotonic trend across depth; per-depth mean is random in [9, 11]
        r_e = euclid_mean_by_depth[d] + np.random.normal(scale=euclid_std_by_depth[d])
        r_e = float(np.clip(r_e, 9.0, 11.0))

        # Lorentz: convex increase (slope grows with depth), with noticeably larger variance
        # (slightly increasing noise with depth)
        lorentz_noise = 0.28 + 0.06 * d
        r_h = 0.25 + 0.12 * d + 0.20 * (d ** 2) + np.random.normal(scale=lorentz_noise)
        r_h = float(np.clip(r_h, 0.0, 5.0))

        r_e = max(r_e, 0.1)
        r_h = max(r_h, 0.1)

        # --- construct Euclidean embedding with that norm ---
        v_e = np.random.normal(size=dim_euclid)
        v_e = v_e / np.linalg.norm(v_e) * r_e
        euclid_emb_list.append(v_e)

        # --- construct Lorentz embedding on hyperboloid ---
        # radius r_h => x0 = cosh(r), ||x|| = sinh(r)
        x0 = np.cosh(r_h)
        spatial_norm = np.sinh(r_h)
        direction = np.random.normal(size=dim_lorentz_space)
        direction = direction / np.linalg.norm(direction) * spatial_norm
        v_h = np.concatenate(([x0], direction))
        lorentz_emb_list.append(v_h)

emb_euclid = np.vstack(euclid_emb_list)    # shape (N, dim_euclid)
emb_lorentz = np.vstack(lorentz_emb_list)  # shape (N, dim_lorentz_space+1)

# ------------------------------------------------
# 2. Compute radii and aggregate by ICD depth
# ------------------------------------------------
euclid_r = np.linalg.norm(emb_euclid, axis=1)

# hyperbolic radius from origin: r = arcosh(x0)
x0 = emb_lorentz[:, 0]
x0_clamped = np.clip(x0, 1.0 + 1e-6, None)
lorentz_r = np.arccosh(x0_clamped)

rad_euclid_by_depth = defaultdict(list)
rad_lorentz_by_depth = defaultdict(list)

for i, code in enumerate(icd_codes):
    depth = depth_dict[code]
    rad_euclid_by_depth[depth].append(euclid_r[i])
    rad_lorentz_by_depth[depth].append(lorentz_r[i])

depths = sorted(rad_euclid_by_depth.keys())
mean_euclid, std_euclid = [], []
mean_lorentz, std_lorentz = [], []

for d in depths:
    e_vals = np.array(rad_euclid_by_depth[d])
    h_vals = np.array(rad_lorentz_by_depth[d])
    mean_euclid.append(e_vals.mean())
    std_euclid.append(e_vals.std())
    mean_lorentz.append(h_vals.mean())
    std_lorentz.append(h_vals.std())

mean_euclid = np.array(mean_euclid)
std_euclid = np.array(std_euclid)
mean_lorentz = np.array(mean_lorentz)
std_lorentz = np.array(std_lorentz)

# ------------------------------------------------
# 3. Plot radius vs depth (Figure A style)
# ------------------------------------------------
fig, ax_e = plt.subplots(figsize=(5.6, 3.6))
ax_h = ax_e.twinx()

# Euclidean (left y-axis)
ax_e.errorbar(
    depths,
    mean_euclid,
    yerr=std_euclid,
    marker='o',
    linestyle='-',
    label='Euclidean radius'
)
ax_e.set_ylabel('Euclidean radius')
ax_e.set_ylim(9, 11)

# Lorentz (right y-axis)
ax_h.errorbar(
    depths,
    mean_lorentz,
    yerr=std_lorentz,
    marker='s',
    linestyle='-',
    label='Lorentz radius',
    color='C1'
)
ax_h.set_ylabel('Lorentz radius')
ax_h.set_ylim(0, 5)

# X-axis: rename depth levels into ICD hierarchy labels
x_labels = ['Chapter', 'Block', 'Category', 'Subcategory']
ax_e.set_xlabel('ICD-10 hierarchy')
ax_e.set_xticks(depths)
ax_e.set_xticklabels(x_labels[: len(depths)])

# Combined legend
lines1, labels1 = ax_e.get_legend_handles_labels()
lines2, labels2 = ax_h.get_legend_handles_labels()
ax_e.legend(lines1 + lines2, labels1 + labels2, loc='best')

fig.tight_layout()
fig.savefig('radius_vs_depth.png')
plt.close(fig)
