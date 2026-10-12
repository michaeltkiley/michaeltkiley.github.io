"""
Embodied progress only: dynamics of r, output, consumption, investment, wages and the labor share
under four household blocks (all log utility, theta = 1):

  Log                       representative Ricardian household
  Log + habits              internal additive habits, h = 0.7
  + hand-to-mouth (50%)     half of households consume their labor income
  + constant savers (50%)   half of households save a constant share of income (net capital
                            income plus wages; revaluation of existing capital excluded);
                            the share is calibrated to the aggregate net saving rate on the
                            initial BGP (3.8%), so they hold their proportional share of wealth

Production, accumulation, adjustment costs and shocks as in ramsey_variants.py: equal depreciation,
c_K = 1/x, c_A calibrated for equal convergence speed, same 10% long-run output gain, regular
capital (29% fall in P_K) versus AI (3.4-fold fall in P_A).
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ramsey_variants as rv

S_YEARS = 5                                   # the S-curve is complete after this many years
rv.cc.configure(dA=rv.cc.BASE["d"]["K"], adj=True, scurve="ramp", T_end=S_YEARS)
HERE, FIG, r0, G = rv.HERE, rv.FIG, rv.r0, rv.G
CASES = [("Log", rv.HH(1, 0, 0, "Log")),
         ("Log + habits", rv.HH(1, 0.7, 0, "Log + habits")),
         ("+ hand-to-mouth (50%)", rv.HH(1, 0.7, 0.5, "H2M", kind="tank")),
         ("+ constant savers (50%)", rv.HH(1, 0.7, 0.5, "Savers", kind="saver"))]
TIMINGS = [("scurve", f"S-curve, complete by year {S_YEARS}"),
           ("mixed", f"Half jump, half S-curve (complete by year {S_YEARS})"), ("jump", "Surprise jump")]
CAPS = [("K", "Regular capital", "#1f5f9e"), ("A", "AI capital", "#c0512a")]


def tex(s):
    return s.replace("%", r"\%")


def main():
    cA, kK, kA = rv.calibrate_cA()
    out, checks = {}, {}
    for case, _ in TIMINGS:
        for cap, _, _ in CAPS:
            for name, hh in CASES:
                res = rv.run(case, cap, "emb", hh, cA)
                out[(case, cap, name)] = res
                key = f"{name} {case} {cap}"
                checks[f"{key}: residual"] = res["err"]
                checks[f"{key}: long-run ln Y = 10%"] = abs(res["lnY"][-30] - 0.10)
                if res["phi"] is not None:
                    checks[f"{key}: savers' wealth share within [0, 1]"] = float(
                        max(0.0, np.max(res["phi"][:-30]) - 1, -np.min(res["phi"][:-30])))
                d = rv.decade(res, hh)
                print(f"  {case:6s} {cap} {name:26s} avg={d['avg']:+.2f} r1={d['r1']:.2f} "
                      f"peak={d['peak']:.2f}@{d['tpeak']} trough={d['trough']:.2f} "
                      f"dC10={100*res['lnC'][10]:.1f} Y10={d['Y10']:.1f} W10={d['W10']:.1f} "
                      f"err={res['err']:.0e}", flush=True)
    for name, hh in CASES:
        T = 60
        path = {"GK": np.ones(T + 1), "PK": np.ones(T + 1), "GA": np.full(T + 1, rv.BASE["GammaA0"]),
                "PA": np.ones(T + 1)}
        x = {j: np.full(T + 1, rv.cc.bgp(rv.E)["x"][j]) for j in rv.CAPS}
        m = rv.Model2(path, x, {"K": rv.C_K, "A": cA}, hh)
        checks[f"{name}: BGP satisfies the Euler equations"] = float(
            np.max(np.abs(m.resid(np.concatenate([np.full(T, m.term[k]) for k in range(2)])))))
        if m.s_rot is not None:
            print("  saver rate s =", round(m.s_rot, 4))

    # ---------------- figures: one per timing
    H = 21
    t = np.arange(H)
    rows = [("r", "real rate (pct.)", 100, None), ("lnY", "output (log dev., pct.)", 100, 0.0),
            ("lnC", "consumption (log dev., pct.)", 100, 0.0),
            ("invsh", "investment / output (pct.)", 100, None),
            ("lnW", "real wage (log dev., pct.)", 100, 0.0), ("labsh", "labor share (pct.)", 100, None)]
    for case, ttl in TIMINGS:
        fig, axes = plt.subplots(len(rows), len(CASES), figsize=(13, 14), sharey="row")
        for col, (name, hh) in enumerate(CASES):
            for cap, lab, color in CAPS:
                res = out[(case, cap, name)]
                for row, (var, yl, mult, ref) in enumerate(rows):
                    y = mult * res[var][:H] if var != "r" else 100 * res["r"][:H]
                    axes[row, col].plot(t, y, color=color, lw=2, label=lab)
            for row, (var, yl, mult, ref) in enumerate(rows):
                ax = axes[row, col]
                ax.grid(alpha=0.25)
                if var == "r":
                    ax.axhline(100 * r0, color="0.6", lw=1, ls=":")
                elif ref is not None:
                    ax.axhline(0, color="0.6", lw=1, ls=":")
                if col == 0:
                    ax.set_ylabel(yl, fontsize=9)
            axes[0, col].set_title(name, fontsize=10)
            axes[-1, col].set_xlabel("years")
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Embodied progress, {ttl}: same 10% long-run output gain", fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"fig_emb_{case}.{ext}"), dpi=140)
        plt.close(fig)

    # ---------------- consumption by household type (the two rule-of-thumb cases)
    for case, ttl in TIMINGS:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row")
        for row, (cap, lab, color) in enumerate(CAPS):
            for col, name in enumerate(("Log + habits", "+ hand-to-mouth (50%)", "+ constant savers (50%)")):
                ax = axes[row, col]
                res = out[(case, cap, name)]
                c0 = res["ricC"][0] + res["rotC"][0]
                ax.plot(t, 100 * np.log(res["ricC"][:H] / res["ricC"][0] * 1.0) if name == "Log + habits"
                        else 100 * np.log(res["ricC"][:H] / res["ricC"][0]), color="#2a8a4a", lw=2,
                        label="Ricardian households")
                if name != "Log + habits":
                    ax.plot(t, 100 * np.log(res["rotC"][:H] / res["rotC"][0]), color="#7a4fb0", lw=2, ls="--",
                            label="rule-of-thumb households")
                ax.plot(t, 100 * res["lnC"][:H], color="0.3", lw=1.2, ls=":", label="aggregate (vs. pre-shock path)")
                ax.grid(alpha=0.25); ax.axhline(0, color="0.7", lw=0.8)
                if row == 0:
                    ax.set_title(name, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{lab}\nconsumption (log, pct.)", fontsize=9)
                if row == 1:
                    ax.set_xlabel("years")
        axes[0, 1].legend(fontsize=7.5, frameon=False)
        fig.suptitle(f"Consumption by household type, embodied progress, {ttl} (detrended; "
                     "types relative to their own year-0 level)", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"fig_emb_cons_{case}.{ext}"), dpi=140)
        plt.close(fig)

    # ---------------- table
    L = [r"\begin{tabular}{llcccccccc}\toprule",
         r" & & decade avg. $r$ & $r_1$ & trough & peak (yr) & $\ln C$, yr 5 & $\ln C$, yr 10 & $\ln W$, yr 10 & $\ln Y$, yr 10\\",
         r" & & (pp over 7.5\%) & (pct.) & (pct.) & (pct.) & (pct.) & (pct.) & (pct.) & (pct.)\\\midrule"]
    for case, ttl in TIMINGS:
        for cap, lab, _ in CAPS:
            for i, (name, hh) in enumerate(CASES):
                res = out[(case, cap, name)]
                d = rv.decade(res, hh)
                L.append(f"{ {'scurve': 'S-curve', 'mixed': 'Half/half', 'jump': 'Jump'}[case] if (cap == 'K' and i == 0) else ''} & "
                         f"{lab.split()[0] + ': ' if i == 0 else ''}{tex(name) if False else ''}{tex(name)} & "
                         f"{d['avg']:+.2f} & {d['r1']:.2f} & {d['trough']:.2f} & {d['peak']:.2f} ({d['tpeak']}) & "
                         f"{100*res['lnC'][5]:.1f} & {100*res['lnC'][10]:.1f} & {d['W10']:.1f} & {d['Y10']:.1f}\\\\")
            L.append(r"\midrule" if not (cap == "A" and case == TIMINGS[-1][0]) else "")
    L.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_emb.tex"), "w").write("\n".join(l for l in L if l) + "\n")
    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    worst = {}
    for k, v in checks.items():
        key = tex(k.split(": ", 1)[1] if ": " in k else k)
        worst[key] = max(worst.get(key, 0.0), v)
    for k, v in worst.items():
        C.append(f"{k} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_emb_checks.tex"), "w").write("\n".join(C) + "\n")
    json.dump(dict(cA=cA, checks=checks,
                   summary={" | ".join(k): dict(rv.decade(v, dict(CASES)[k[2]]),
                                                lnC=[float(100 * v["lnC"][i]) for i in (0, 5, 10, 20)])
                            for k, v in out.items()}),
              open(os.path.join(HERE, "results_emb.json"), "w"), indent=1, default=float)
    print("max check:", max(checks.values()))
    for k, v in checks.items():
        if v > 1e-6:
            print("  LARGE", k, v)


if __name__ == "__main__":
    main()
