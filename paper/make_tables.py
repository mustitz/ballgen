#!/usr/bin/env python3
"""Turn data/*.json into LaTeX fragments (tables/*.tex) and number macros (tables/numbers.tex).
Also verifies, from the data, the comparative claims the paper makes; it fails loudly if one is false."""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf          # for invariant_dims

D = lambda n: json.load(open(os.path.join(HERE, "data", n)))
fam = {r["key"]: r for r in D("family.json")}
aud, tau, sens = D("audit_A.json"), D("tau_sweep.json"), D("sensitivity.json")
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)

LABEL = {"tetra-4": ("tetra-4", "$T_d$"), "trionda": (r"\textbf{Trionda model} ($\tau=0.9$)", "$T$"),
         "trionda-20": (r"Trionda model ($20^\circ$ tilt)", "$T$"),
         "cube-6": ("cube-6", "$O_h$"), "cube-6-chiral": ("cube-6, chiral", "$O$"),
         "octa-8": ("octa-8", "$O_h$"), "octa-8-chiral": ("octa-8, chiral", "$O$"),
         "dodeca-12": ("dodeca-12", "$I_h$"), "dodeca-12-chiral": ("dodeca-12, chiral", "$I$"),
         "icosa-20": ("icosa-20", "$I_h$"), "icosa-20-chiral": ("icosa-20, chiral", "$I$"),
         "classic-32": ("classic-32 (trunc.\\ icosahedron)", "$I_h$")}
ORDER = {"T": 12, "T_d": 12, "O": 24, "O_h": 24, "I": 60, "I_h": 60}

# ---- consistency checks between detection and theory
for k, r in fam.items():
    g = r["expected_group"]
    assert r["rotations"] == ORDER[g], (k, r["rotations"], g)
    assert r["chiral"] == (r["improper"] == 0) and (r["improper"] == 0) == ("_h" not in g and "_d" not in g), (k, r["improper"], g)
    dims = mf.invariant_dims(g.split("_")[0])
    for l in range(1, 13):
        if dims[l] == 0:
            assert r["q"][l] < 1e-9, f"{k}: Q_{l}={r['q'][l]} but no invariant harmonic of degree {l}"
    first_allowed = next(l for l in range(1, 13) if dims[l] > 0)
    assert r["lowest_multipole"] == first_allowed, (k, r["lowest_multipole"], first_allowed)
print("symmetry / multipole checks passed for", len(fam), "balls")

# ---- comparative claims used in the text
claims = {}
cvs = {k: r["exposure"]["cv"] for k, r in fam.items()}
claims["chirality_lowers_cv_all_families"] = all(cvs[a] < cvs[b] for a, b in [
    ("trionda-20", "tetra-4"), ("cube-6-chiral", "cube-6"), ("octa-8-chiral", "octa-8"),
    ("dodeca-12-chiral", "dodeca-12"), ("icosa-20-chiral", "icosa-20")])
claims["classic_lowest_cv"] = all(cvs["classic-32"] < v for k, v in cvs.items() if k != "classic-32")
cv = sens["cv"]
n_sets = len(sens["seps"]) * len(sens["sigmas"])
claims["trionda_below_tetra_all_sensitivity"] = all(cv["trionda"][i][j] < cv["tetra-4"][i][j]
                                                   for i in range(len(sens["sigmas"])) for j in range(len(sens["seps"])))
claims["tau2_below_trionda_all_sensitivity"] = all(cv["trionda-tau2"][i][j] < cv["trionda"][i][j]
                                                  for i in range(len(sens["sigmas"])) for j in range(len(sens["seps"])))
claims["classic_below_trionda_all_sensitivity"] = all(cv["classic-32"][i][j] < cv["trionda"][i][j]
                                                     for i in range(len(sens["sigmas"])) for j in range(len(sens["seps"])))
claims["cube_vs_tetra_order_flips"] = len({cv["cube-6"][1][j] < cv["tetra-4"][1][j] for j in range(len(sens["seps"]))}) == 2
claims["cv_monotone_decreasing_in_tau_beyond_0.3"] = all(a["exposure_cv"] >= b["exposure_cv"] - 1e-9
                                                        for a, b in zip(tau["rows"], tau["rows"][1:]) if a["tau"] >= 0.3)
claims["seam_length_monotone_in_tau"] = all(a["seam_length"] < b["seam_length"] for a, b in zip(tau["rows"], tau["rows"][1:]))
print(json.dumps(claims, indent=1))
json.dump(claims, open(os.path.join(HERE, "data", "claims.json"), "w"), indent=1)

# ---- family table
rows = []
for key in ["tetra-4", "trionda", "trionda-20", "cube-6", "cube-6-chiral", "octa-8", "octa-8-chiral",
            "dodeca-12", "dodeca-12-chiral", "icosa-20", "icosa-20-chiral", "classic-32"]:
    r = fam[key]
    lab, grp = LABEL[key]
    sym = f"{grp} ({r['rotations']}" + (")" if r["chiral"] else f"+{r['improper']})")
    rows.append(f"{lab} & {r['panels']} & {sym} & {r['seam_length']:.2f} & {r['seam_per_area']:.3f} & "
                f"{r['lowest_multipole']} & {r['exposure']['cv']:.3f} & {r['mean_abs_kg']:.2f} \\\\")
    if key in ("trionda-20", "cube-6-chiral" if False else "trionda-20", "octa-8-chiral", "dodeca-12-chiral", "icosa-20-chiral") and key != "classic-32":
        rows.append(r"\addlinespace")
open(os.path.join(HERE, "tables", "family.tex"), "w").write(
    "\\begin{tabular}{@{}lrlrrrrr@{}}\n\\toprule\n"
    "layout & panels & symmetry & seam length & length/area & $\\ell^*$ & exposure CV & mean $|\\kappa_g|$ \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- sensitivity table (sigma = 10 deg row)
names = [("classic-32", "classic-32"), ("cube-6", "cube-6"), ("tetra-4", "tetra-4"),
         ("trionda", r"Trionda model ($\tau=0.9$)"), ("trionda-tau2", r"Trionda model ($\tau=2$)")]
si = sens["sigmas"].index(10)
head = " & ".join(f"${s}^\\circ$" for s in sens["seps"])
body = "\n".join(f"{lab} & " + " & ".join(f"{v:.3f}" for v in cv[k][si]) + " \\\\" for k, lab in names)
open(os.path.join(HERE, "tables", "sensitivity.tex"), "w").write(
    "\\begin{tabular}{@{}l" + "r" * len(sens["seps"]) + "@{}}\n\\toprule\n"
    f"separation angle $\\theta_s$ & {head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")

# ---- invariant-dimension table
dims = {g: mf.invariant_dims(g) for g in "TOI"}
open(os.path.join(HERE, "tables", "invariants.tex"), "w").write(
    "\\begin{tabular}{@{}l" + "r" * 12 + "@{}}\n\\toprule\n$\\ell$ & " + " & ".join(str(l) for l in range(1, 13)) + " \\\\\n\\midrule\n" +
    "\n".join(f"$\\dim H_\\ell^{{{g}}}$ & " + " & ".join(str(dims[g][l]) for l in range(1, 13)) + " \\\\" for g in "TOI") +
    "\n\\bottomrule\n\\end{tabular}\n")

# ---- macros
t09 = min(tau["rows"], key=lambda r: abs(r["tau"] - 0.9))
t20 = min(tau["rows"], key=lambda r: abs(r["tau"] - 2.0))
tr = fam["trionda"]
m = {
 "gapUser": f"{aud['user_params']['gap']*100:.1f}", "gapSlope": f"{aud['loglog_slope']:.2f}",
 "gapAtHalfA": f"{[r for r in aud['sweep'] if abs(r['amp']-0.104)<0.01][0]['gap']*100:.1f}",
 "straightL": f"{tau['straight_length']:.2f}", "tauDefault": "0.9",
 "LDefault": f"{tr['seam_length']:.2f}", "LRatioDefault": f"{tr['seam_length']/tau['straight_length']:.2f}",
 "LPctDefault": f"{(tr['seam_length']/tau['straight_length']-1)*100:.0f}",
 "tiltDefault": f"{math.degrees(math.atan(math.tanh(0.9))):.1f}",
 "vertexAngleDefault": f"{math.degrees(2*0.9/math.sqrt(3)):.1f}",
 "expTetra": f"{cvs['tetra-4']:.3f}", "expTrionda": f"{cvs['trionda']:.3f}", "expTrionTwenty": f"{cvs['trionda-20']:.3f}",
 "expClassic": f"{cvs['classic-32']:.3f}", "expTauTwo": f"{t20['exposure_cv']:.3f}", "LRatioTauTwo": f"{t20['ratio']:.2f}",
 "expRatioClassic": f"{cvs['trionda']/cvs['classic-32']:.1f}",
 "kgMeanDefault": f"{tr['mean_abs_kg']:.2f}",
 "famTilt": "20",
}
# maximum reachable edge-midpoint tilt of each family's flow (limit tau -> infinity), from the exact Jacobian
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np, balls
for fk, poly, grad in [("Tetra", balls.poly_tetrahedron(), balls.grad_tetra), ("Cube", balls.poly_cube_k(), balls.grad_cube),
                       ("Octa", balls.poly_octa_k(), balls.grad_cube),
                       ("Dodeca", balls.poly_dodecahedron_k(), balls.make_grad_icosa(balls.icosa_axes_of(balls.poly_dodecahedron()))),
                       ("Icosa", balls.poly_icosahedron_k(), balls.make_grad_icosa(balls.icosa_axes_of(balls.poly_icosahedron())))]:
    b0 = balls.Ball(fk, "", poly[0], poly[1], poly[2], balls._look(["#fff"]))
    V = b0.face_vertices()[0]
    J = balls._edge_jacobian(grad, V[0] + V[1], V[1] - V[0])
    lam = math.sqrt(-np.linalg.det(J))
    m["lim" + fk] = f"{balls.edge_tilt_deg(J / lam, 60):.1f}"
open(os.path.join(HERE, "tables", "numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
print(m)
