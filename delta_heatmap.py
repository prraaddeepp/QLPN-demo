import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─── grid ─────────────────────────────────────────────
n_vals = np.array([128, 192, 256])
d_vals = np.arange(10, 55, 2)
c      = 6

Z = np.empty((len(n_vals), len(d_vals)))   # log₂ δ(n,d)
for i, n in enumerate(n_vals):
    m      = n**c
    w      = int(n * np.log2(n))
    beta_p = n**-0.5 + 2*w/m
    for j, d in enumerate(d_vals):
        delta   = 0.5 * 2**((n+1)/2) * beta_p**d
        Z[i, j] = np.log2(delta)

# ─── plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.5))          # wider & taller

vmin, vmax = -120, 0                               # log₂δ scale
im = ax.imshow(
    Z.clip(vmin, vmax),
    origin="lower",
    aspect="auto",
    cmap="magma_r",                                # light = small δ
    extent=[d_vals[0], d_vals[-1], n_vals[0], n_vals[-1]],
    vmin=vmin, vmax=vmax,
)

# add dashed contour at δ = 2⁻⁴⁰
CS = ax.contour(
    d_vals, n_vals, Z,
    levels=[-40],
    colors="white",
    linestyles="--",
    linewidths=1.2,
)
ax.clabel(CS, fmt=r"$\log_2\delta=-40$", fontsize=9, inline=True)

# colour-bar every 20 dB
cbar = fig.colorbar(im, ax=ax, pad=0.03, label=r"$\log_2\delta$")
cbar.set_ticks(np.arange(0, -121, -20))
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

# labels & ticks
ax.set_xlabel(r"masking depth $d$", fontsize=11)
ax.set_ylabel(r"qubits $n$",        fontsize=11)
ax.set_yticks(n_vals)

plt.tight_layout()
plt.savefig("delta_heatmap.pdf")   # vector PDF for LaTeX
plt.show()
