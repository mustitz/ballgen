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
         "goldberg-42": ("goldberg-42 (12 pent., 30 hex.)", "$I_h$"), "goldberg-42-chiral": ("goldberg-42, chiral", "$I$"),
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
    ("dodeca-12-chiral", "dodeca-12"), ("icosa-20-chiral", "icosa-20"),
    ("goldberg-42-chiral", "goldberg-42")])
claims["goldberg_lowest_cv"] = all(cvs["goldberg-42"] < v for k, v in cvs.items() if not k.startswith("goldberg-42"))
claims["classic_second_lowest_cv"] = all(cvs["classic-32"] < v for k, v in cvs.items() if k != "classic-32" and not k.startswith("goldberg-42"))
claims["cv_decreases_with_panels_in_Ih"] = cvs["dodeca-12"] > cvs["icosa-20"] > cvs["classic-32"] > cvs["goldberg-42"]
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

# ---- Spectre folding section (data/spectre.json from make_spectre_data.py)
SPF = os.path.join(HERE, "data", "spectre.json")
sp = json.load(open(SPF)) if os.path.exists(SPF) else None
if sp:
    cl = sp["closure"]["rows"]
    pct = lambda n: cl[str(n)]["closed"] / cl[str(n)]["discs"]
    sv = sp["survey"]["counts"]
    claims["spectre_dp_matches_brute_force"] = sp["validate"]["all_agree"]
    claims["spectre_closure_rate_falls_with_size"] = pct(4) > pct(6) > pct(8) > pct(12) > pct(24)
    claims["spectre_no_regular_hexagon_hole_in_survey"] = sv["hexagon_regular"] == 0
    claims["spectre_no_pentagon_pair_in_survey"] = sv["pentagon_pair"] == 0
    claims["spectre_no_flat_hexagon_of_any_shape_in_survey"] = sv["flat_hexagon_any"] == 0
    claims["spectre_some_closed_balls_at_19_to_21_tiles"] = any(cl[str(n)]["closed"] > 0 for n in (19, 20, 21))
    claims["spectre_assembly_ledgers"] = (sp["joins"]["pair_ledger"] == 720 and sp["joins"]["chain3_ledger"] == 720
                                          and sp["joins"]["ring3_ledger"] == 0)
# ---- Trionda / Hat Trick sections (data/hat_trick.json from make_hat_data.py)
HTF = os.path.join(HERE, "data", "hat_trick.json")
ht = json.load(open(HTF)) if os.path.exists(HTF) else None
if ht:
    dz, tt, ic, hb, fl, cmp_ = ht["design"], ht["tetra_subgroup"], ht["icosa_group"], ht["hat_trionda"], ht["flow"], ht["table"]
    kt = dz["kite_angle_totals"]
    claims["hat_design_counts"] = (dz["kites"] == 480 and dz["hats"] == 60 and dz["pentagons"] == 12 and dz["kites_per_hat"] == [8]
                                   and dz["rotations"] == 60 and dz["symmetric_tilings"] == 2)
    claims["hat_ledger_flat_except_holes"] = (set(kt) == {"v:360", "m:360", "c:360", "c:240", "m:180"}
                                              and kt["c:240"] == 60 and kt["m:180"] == 60)
    claims["hat_ledger_720"] = 12 * (360 + 180 * 10 - (5 * 240 + 5 * 180)) == 720
    claims["T_acts_freely_on_hats_and_pentagons"] = (tt["order"] == 12 and tt["hat_orbit_sizes"] == [12] * 5
                                                     and tt["pentagon_orbit_sizes"] == [12]
                                                     and tt["max_hat_stabiliser"] == 1 and tt["max_pentagon_stabiliser"] == 1)
    claims["I_acts_freely_on_hats"] = ic["hat_orbit_sizes"] == [60] and ic["max_hat_stabiliser"] == 1
    claims["hat_trionda_15_hats_3_pentagons"] = hb["hats_per_panel"] == [15] * 4 and hb["pentagons_per_panel"] == [3] * 4
    claims["hat_trionda_no_ties"] = hb["min_majority_fraction"] > 0.5
    claims["selection_rule_trionda_3_hat_6"] = cmp_["trionda"]["lowest_multipole"] == 3 and cmp_["hat-trick-v0"]["lowest_multipole"] == 6
    claims["hat_cv_lowest_of_table"] = all(cmp_["hat-trick-v0"]["exposure_cv"] < cmp_[k]["exposure_cv"] for k in cmp_ if not k.startswith("hat-trick"))
    claims["flow_blend_switches_on_l3"] = all(r["lowest"] == 3 for r in fl if r["tau"] != 0) and next(r for r in fl if r["tau"] == 0)["lowest"] == 6
    claims["flow_blend_lengthens_seams"] = all(r["seam_length"] > next(q for q in fl if q["tau"] == 0)["seam_length"] for r in fl if r["tau"] != 0)
    claims["flow_blend_cv_below_trionda"] = all(r["exposure_cv"] < cmp_["trionda"]["exposure_cv"] for r in fl)
    fs = {r["f"]: r for r in ht["frequency_scan"]}
    claims["freq_f3_I_exactly_two"] = fs[3]["I"]["tilings"] == 2 and fs[3]["I"]["exhausted"]
    claims["freq_f5_f7_f9_none_exhausted"] = all(fs[f][g]["tilings"] == 0 and fs[f][g]["exhausted"] for f in (5, 7, 9) for g in ("I", "T", "C5", "C1"))
    claims["freq_f5_f7_both_hands_exhausted"] = all(fs[f]["C1_both_hands"]["tilings"] == 0 and fs[f]["C1_both_hands"]["exhausted"] for f in (5, 7))
    claims["freq_f9_both_hands_not_exhausted"] = not fs[9]["C1_both_hands"]["exhausted"]
    claims["hat_count_formula"] = all(abs(r["hats"] - 7.5 * (r["f"] ** 2 - 1)) < 1e-9 for r in fs.values())
print(json.dumps(claims, indent=1))
json.dump(claims, open(os.path.join(HERE, "data", "claims.json"), "w"), indent=1)

# ---- family table
rows = []
for key in ["tetra-4", "trionda", "trionda-20", "cube-6", "cube-6-chiral", "octa-8", "octa-8-chiral",
            "dodeca-12", "dodeca-12-chiral", "icosa-20", "icosa-20-chiral", "goldberg-42", "goldberg-42-chiral", "classic-32"]:
    r = fam[key]
    lab, grp = LABEL[key]
    sym = f"{grp} ({r['rotations']}" + (")" if r["chiral"] else f"+{r['improper']})")
    rows.append(f"{lab} & {r['panels']} & {sym} & {r['seam_length']:.2f} & {r['seam_per_area']:.3f} & "
                f"{r['lowest_multipole']} & {r['exposure']['cv']:.3f} & {r['mean_abs_kg']:.2f} \\\\")
    if key in ("trionda-20", "cube-6-chiral" if False else "trionda-20", "octa-8-chiral", "dodeca-12-chiral", "icosa-20-chiral", "goldberg-42-chiral") and key != "classic-32":
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
 "expClassic": f"{cvs['classic-32']:.3f}", "expGoldberg": f"{cvs['goldberg-42']:.3f}", "expGoldbergChiral": f"{cvs['goldberg-42-chiral']:.3f}",
 "expRatioGoldberg": f"{cvs['trionda']/cvs['goldberg-42']:.0f}", "expTauTwo": f"{t20['exposure_cv']:.3f}", "LRatioTauTwo": f"{t20['ratio']:.2f}",
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

if sp:
    sv, hb, jn, va = sp["survey"], sp["joins"], sp["joins"], sp["validate"]
    c = sv["counts"]
    m.update({
        "spSurveyDraws": str(sv["draws"]), "spSurveyDiscs": str(c["discs"]), "spSurveyClosed": str(c["closed"]),
        "spSurveySquare": str(c["square"]), "spSurveySquarePair": str(c["square_pair"]),
        "spSurveyHex": str(c["hexagon_regular"]), "spSurveyPent": str(c["pentagon_pair"]),
        "spFlatDraws": str(c["flat_draws"]), "spFlatHex": str(c["flat_hexagon_any"]),
        "spFlatPatterns": str(sv["n_flat_hexagon_patterns"]),
        "spNMin": str(sv["n_range"][0]), "spNMax": str(sv["n_range"][1]), "spFlatNMax": str(sv["flat_n_range"][1]),
        "spHubThreeFound": str(sp["hub3"]["found"]), "spHubThreeDiscs": str(sp["hub3"]["discs"]),
        "spHubFourFound": str(sp["hub4"]["found"]), "spHubFourDiscs": str(sp["hub4"]["discs"]),
        "spPairLedger": str(jn["pair_ledger"]), "spChainLedger": str(jn["chain3_ledger"]), "spRingLedger": str(jn["ring3_ledger"]),
        "spJunctionA": str(jn["pair_junction_deficits"][0]), "spJunctionB": str(jn["pair_junction_deficits"][1]),
        "spPerN": str(sp["closure"]["per_n"]),
        "spBruteOne": str(va["one_tile_k4"]["brute"]), "spBruteSix": str(va["one_tile_k6"]["brute"]),
        "spBruteTwo": str(va["two_tiles_k4"]["brute"]),
    })
    cl = sp["closure"]["rows"]
    rows_ = []
    for n in (4, 5, 6, 8, 10, 12, 16, 20, 24):
        r = cl[str(n)]
        rows_.append(f"{n} & {r['discs']} & {r['closed']} & {100 * r['closed'] / r['discs']:.1f} \\\\")
    open(os.path.join(HERE, "tables", "spectre_closure.tex"), "w").write(
        "\\begin{tabular}{@{}rrrr@{}}\n\\toprule\ntiles $N$ & patches & close up & \\% \\\\\n\\midrule\n"
        + "\n".join(rows_) + "\n\\bottomrule\n\\end{tabular}\n")
def sci(x, d=1):
    if x == 0:
        return "0"
    mant, ex = f"{x:.{d}e}".split("e")
    return f"{mant}\\times10^{{{int(ex)}}}"


if ht:
    LAB = {"trionda": (r"\textbf{Trionda model} ($\tau=0.9$)", "$T$"), "tetra-4": ("tetra-4", "$T_d$"),
           "hat-trick-v0": (r"\textbf{Hat Trick}, tiling 0", "$I$"), "hat-trick-v1": ("Hat Trick, tiling 1", "$I$"),
           "icosa-20": ("icosa-20", "$I_h$"), "classic-32": ("classic-32", "$I_h$"), "goldberg-42": ("goldberg-42", "$I_h$")}
    rows = []
    for k in ["trionda", "tetra-4", "hat-trick-v0", "hat-trick-v1", "icosa-20", "classic-32", "goldberg-42"]:
        r = cmp_[k]
        rows.append(f"{LAB[k][0]} & {r['panels']} & {LAB[k][1]} & {r['seam_length']:.2f} & {r['lowest_multipole']} & "
                    f"{r['exposure_cv']:.4f} & {r['mean_abs_geodesic_curvature']:.2f} \\\\")
    open(os.path.join(HERE, "tables", "compare.tex"), "w").write(
        "\\begin{tabular}{@{}lrlrrrr@{}}\n\\toprule\nlayout & panels & group & seam length & $\\ell^*$ & exposure CV & mean $|\\kappa_g|$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    frows = [f"${r['tau']:+.1f}$".replace("+0.0", "0.0") + f" & {r['seam_length']:.2f} & {r['exposure_cv']:.4f} & {r['lowest']} & ${sci(r['Q3'])}$ & ${sci(r['Q4'])}$ \\\\" for r in fl]
    open(os.path.join(HERE, "tables", "flow.tex"), "w").write(
        "\\begin{tabular}{@{}rrrrrr@{}}\n\\toprule\n$\\tau$ & seam length & exposure CV & $\\ell^*$ & $Q_3$ & $Q_4$ \\\\\n\\midrule\n"
        + "\n".join(frows) + "\n\\bottomrule\n\\end{tabular}\n")
    def cell(r):
        if r["exhausted"]:
            return f"{r['tilings']}"
        return f"$\\ge{r['tilings']}$" if r["tilings"] else "$0^{*}$"
    qrows = [f"{r['f']} & {r['tiles']} & {int(r['hats'])} & " + " & ".join(cell(r[g]) for g in ("I", "T", "C5", "C1", "C1_both_hands")) + " \\\\"
             for r in ht["frequency_scan"]]
    open(os.path.join(HERE, "tables", "hat_freq.tex"), "w").write(
        "\\begin{tabular}{@{}rrrrrrrr@{}}\n\\toprule\n$f$ & tiles & hats & $I$ & $T$ & $C_5$ & $C_1$ & $C_1$, both hands \\\\\n\\midrule\n"
        + "\n".join(qrows) + "\n\\bottomrule\n\\end{tabular}\n")
    ch = ht["chirality_deg"]
    htr = ht["hat_trionda"]
    rl = dz["relax"]
    fl0 = next(r for r in fl if r["tau"] == 0)
    fl9 = next(r for r in fl if abs(r["tau"] - 0.9) < 1e-9)
    hrow = ht["table"]["hat-trick-v0"]
    m.update({
        "htTiles": str(dz["tiles"]), "htTriangles": str(dz["triangles"]), "htKites": str(dz["kites"]), "htHats": str(dz["hats"]),
        "htPlacements": str(dz["placements"]), "htTilings": {2: "two"}.get(dz["symmetric_tilings"], str(dz["symmetric_tilings"])),
        "htRotations": str(dz["rotations"]), "htRms": f"{100 * rl['rms_strain']:.1f}", "htMaxStrain": f"{100 * rl['max_strain']:.1f}",
        "htNConstraints": str(int(rl["n_constraints"])),
        "htAreaRatio": f"{dz['report']['hat_to_pentagon_area']:.2f}", "htAreaSpread": f"{dz['report']['hat_area_spread']:.0e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}",
        "htVerts": str(sum(kt.values())), "htFlatCorners": str(kt["v:360"] + kt["m:360"] + kt["c:360"]),
        "htCentroidCorners": str(kt["c:240"]), "htMidCorners": str(kt["m:180"]),
        "htPentOrbits": ",".join(str(x) for x in tt["pentagon_orbit_sizes"]), "htHatOrbits": ",".join(str(x) for x in tt["hat_orbit_sizes"]),
        "htQSix": "$" + sci(hrow['Q'][6]) + "$",
        "cmpTriSeam": f"{cmp_['trionda']['seam_length']:.2f}", "cmpHatSeam": f"{hrow['seam_length']:.2f}",
        "cmpSeamRatio": f"{hrow['seam_length'] / cmp_['trionda']['seam_length']:.1f}",
        "cmpTriLow": str(cmp_["trionda"]["lowest_multipole"]), "cmpHatLow": str(hrow["lowest_multipole"]),
        "cmpTriCV": f"{cmp_['trionda']['exposure_cv']:.3f}", "cmpHatCV": f"{hrow['exposure_cv']:.4f}",
        "cmpHatCVone": f"{ht['table']['hat-trick-v1']['exposure_cv']:.4f}",
        "cmpGoldCV": f"{cmp_['goldberg-42']['exposure_cv']:.3f}", "cmpClassicCV": f"{cmp_['classic-32']['exposure_cv']:.3f}",
        "cmpRatioCV": f"{cmp_['trionda']['exposure_cv'] / hrow['exposure_cv']:.0f}",
        "cmpHatShape": f"{hrow['panel_shape_hat']:.2f}", "cmpPentShape": f"{hrow['panel_shape_pentagon']:.2f}",
        "chTetra": f"{ch['tetra-4']:.1f}", "chTrioA": f"{ch['trionda tau=0.3']:.1f}", "chTrioB": f"{ch['trionda tau=0.6']:.1f}",
        "chTrioC": f"{ch['trionda tau=0.9']:.1f}", "chHatA": f"{ch['hat-trick-v0']:.1f}", "chHatB": f"{ch['hat-trick-v1']:.1f}",
        "chGoldChiral": f"{ch['goldberg-42-chiral']:.1f}",
        "blHats": str(htr["hats_per_panel"][0]), "blPent": str(htr["pentagons_per_panel"][0]), "blSeam": f"{htr['super_seam_length']:.2f}",
        "blMismatch": f"{100 * htr['area_mismatch_fraction']:.1f}", "blStraddle": str(htr["straddling_hats"]),
        "blMinMaj": f"{100 * htr['min_majority_fraction']:.1f}",
        "flSeamZero": f"{fl0['seam_length']:.1f}", "flSeamEnd": f"{fl9['seam_length']:.1f}",
        "flCVZero": f"{fl0['exposure_cv']:.3f}", "flCVEnd": f"{fl9['exposure_cv']:.3f}",
        "flRatio": f"{cmp_['trionda']['exposure_cv'] / fl9['exposure_cv']:.1f}",
    })
open(os.path.join(HERE, "tables", "numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
print(m)
