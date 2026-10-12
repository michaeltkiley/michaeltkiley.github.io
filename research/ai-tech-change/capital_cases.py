"""
AI capital versus regular capital: surprise jumps and an anticipated S-curve.

Baseline:
    X = (Gamma_K K)^a (Z N)^(1-a)                                   Cobb-Douglas bundle
    Y = [w^(1/e) X^rho + (1-w)^(1/e) (Gamma_A A)^rho]^(1/rho),  e = 1.5  (AI substitutes)
    K' = (1-d_K) K + I_K/P_K,     A' = (1-d_A) A + I_A/P_A
    adjustment costs  P_j 0.5 c_j (I_j/(P_j K_j) - x_j)^2 K_j, x_j = BGP rate,
    c_j = 1/x_j (elasticity of investment to Q of one for both capital stocks)
    log utility, fixed hours.
Disembodied progress: Gamma_j (production).  Embodied progress: P_j falls (accumulation).

Experiments, each for AI and regular capital, embodied and disembodied, every shock
sized for the same 10% long-run output gain:
  jump     surprise permanent level jump at t = 0
  S-curve  smooth level shift 0.5(1 - cos(pi t/10)), announced at t = 0, complete by year 10
  mixed    half of the shift as a surprise jump at t = 0, half as the S-curve

(A permanent increase in the growth rate of AI technology is excluded: with e > 1 it
has no balanced growth path.)  The balanced growth path is in closed form in bgp();
transitions are perfect-foresight Newton solves of the nonlinear equilibrium.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BASE = dict(a=0.282, g_Z=0.015, r0=0.075, d=dict(K=0.066, A=0.20), sA0=0.03,
            target_level=0.10, mid=10.0, width=2.5)
BASE["beta"] = (1 + BASE["g_Z"]) / (1 + BASE["r0"])          # log utility
BASE["omega"] = 1 - BASE["sA0"]
BASE["GammaA0"] = BASE["r0"] + BASE["d"]["A"]                # lambda_A = 1 initially
CAPS = ("K", "A")
# adjustment costs: elasticity of I/K with respect to Q equal to one, 1/(c x) = 1
# (Kiley 2000's "moderate" case)
ADJ = {j: 1.0 / (BASE["g_Z"] + BASE["d"][j]) for j in CAPS}
# variant settings (set by configure()): adjustment costs on/off and output-file suffix
OPTS = dict(adj=True, suffix="", label="adjustment costs", scurve="ramp", T_end=10)


def configure(dA=0.20, adj=True, suffix="", label=None, scurve="ramp", T_end=10):
    """Set AI depreciation, adjustment costs and the S-curve; recompute dependent calibration.

    scurve = "logistic": centered on year `mid` (baseline note).
    scurve = "ramp":     0.5(1 - cos(pi t/T_end)) for t <= T_end and 1 after: smooth, flat at
                         both ends, fully converged at year T_end.
    """
    BASE["d"]["A"] = dA
    BASE["GammaA0"] = BASE["r0"] + dA                         # keep lambda_A = 1 initially
    for j in CAPS:
        ADJ[j] = 1.0 / (BASE["g_Z"] + BASE["d"][j])
    OPTS.update(adj=adj, suffix=suffix, scurve=scurve, T_end=T_end,
                label=label or ("adjustment costs" if adj else "no adjustment costs"))


# ----------------------------------------------------------------------------
# Production
# ----------------------------------------------------------------------------
def prod(X, M, e, p=BASE):
    om = p["omega"]
    if e == 1.0:
        kap = om ** om * (1 - om) ** (1 - om)
        F = X ** om * M ** (1 - om) / kap
        return F, om * F / X, (1 - om) * F / M
    r = (e - 1) / e
    F = (om ** (1 / e) * X ** r + (1 - om) ** (1 / e) * M ** r) ** (1 / r)
    return F, (om * F / X) ** (1 / e), ((1 - om) * F / M) ** (1 / e)


# ----------------------------------------------------------------------------
# Balanced growth path: closed forms
# ----------------------------------------------------------------------------
def bgp(e, lev=None, gr=None, p=BASE):
    """lev: levels {GK, PK, GA, PA}; gr: growth rates {gGK, gqK, gGA, gqA}.

    Returns growth, r, user costs, shares, W/Z and ratios, all in closed form:
      e = 1:  ln G = ln G_Z + [a w ln(G_GK G_qK) + (1-w) ln(G_GA G_qA)] / ((1-a) w)
      e != 1: AI progress must be zero; ln G = ln G_Z + a/(1-a) ln(G_GK G_qK)
      1 + r = G / beta,  u_j = (1+r)(1+gamma_qj) - (1-d_j),  lambda_j = P_j u_j / Gamma_j
      s_A = (1-w) lambda_A^(1-e),  labor share = (1-a)(1-s_A)
      P_X^(1-e) = (1-s_A)/w   (P_X = lambda_A^(-(1-w)/w) if e = 1)
      W/Z = (1-a) P_X^(1/(1-a)) (a/lambda_K)^(a/(1-a)),  Y/Z = P_X (X/Z)/(1-s_A)
    """
    lev = dict(dict(GK=1.0, PK=1.0, GA=p["GammaA0"], PA=1.0), **(lev or {}))
    gr = dict(dict(gGK=0.0, gqK=0.0, gGA=0.0, gqA=0.0), **(gr or {}))
    a, om = p["a"], p["omega"]
    GZ = 1 + p["g_Z"]
    gK = (1 + gr["gGK"]) * (1 + gr["gqK"])
    gA = (1 + gr["gGA"]) * (1 + gr["gqA"])
    if e == 1.0:
        G = GZ * np.exp((a * om * np.log(gK) + (1 - om) * np.log(gA)) / ((1 - a) * om))
    else:
        if gA != 1.0:
            return None                          # no BGP: AI share -> 1
        G = GZ * gK ** (a / (1 - a))
    r = G / p["beta"] - 1
    u = {"K": (1 + r) * (1 + gr["gqK"]) - (1 - p["d"]["K"]),
         "A": (1 + r) * (1 + gr["gqA"]) - (1 - p["d"]["A"])}
    lamK = lev["PK"] * u["K"] / lev["GK"]
    lamA = lev["PA"] * u["A"] / lev["GA"]
    if e == 1.0:
        sA = 1 - om
        PX = lamA ** (-(1 - om) / om)
    else:
        sA = (1 - om) * lamA ** (1 - e)
        PX = ((1 - sA) / om) ** (1 / (1 - e))
    XZ = (a * PX / lamK) ** (a / (1 - a))
    WZ = (1 - a) * PX * XZ
    YZ = PX * XZ / (1 - sA)
    KY = a * (1 - sA) / (lev["PK"] * u["K"])          # efficiency units per unit of output
    AY = sA / (lev["PA"] * u["A"])
    GKe = G * (1 + gr["gqK"])                         # growth of K in efficiency units
    GAe = G * (1 + gr["gqA"])
    x = {"K": GKe - 1 + p["d"]["K"], "A": GAe - 1 + p["d"]["A"]}
    IY = lev["PK"] * x["K"] * KY + lev["PA"] * x["A"] * AY
    return dict(G=G, g=G - 1, r=r, u=u, lamK=lamK, lamA=lamA, sA=sA, PX=PX, WZ=WZ, YZ=YZ,
                KY=KY, AY=AY, x=x, IY=IY, labsh=(1 - a) * (1 - sA), Gq={"K": 1 + gr["gqK"],
                                                                         "A": 1 + gr["gqA"]})


# ----------------------------------------------------------------------------
# Shock design
# ----------------------------------------------------------------------------
def scurve(T, p=BASE):
    t = np.arange(T + 1)
    if OPTS["scurve"] == "ramp":
        Te = OPTS["T_end"]
        return np.where(t >= Te, 1.0, 0.5 * (1 - np.cos(np.pi * np.minimum(t, Te) / Te)))
    S = 1 / (1 + np.exp(-(t - p["mid"]) / p["width"]))
    return (S - S[0]) / (1 - S[0])


def level_size(e, cap, p=BASE):
    """Log size of Gamma_j/P_j that raises long-run Y/Z by target_level (same for both types)."""
    Y0 = bgp(e)["YZ"]
    key = "GK" if cap == "K" else "GA"
    base = 1.0 if cap == "K" else p["GammaA0"]
    lo, hi = 0.0, 0.5
    f = lambda x: np.log(bgp(e, lev={key: base * np.exp(x)})["YZ"] / Y0) - p["target_level"]
    while f(hi) < 0:
        hi *= 2
    for _ in range(100):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) < 0 else (lo, m)
    return 0.5 * (lo + hi)


def tech(e, case, cap, typ, T, p=BASE):
    """Paths of Gamma_K, P_K, Gamma_A, P_A, and the adjustment-cost targets x_j(t)."""
    path = {"GK": np.ones(T + 1), "PK": np.ones(T + 1),
            "GA": np.full(T + 1, p["GammaA0"]), "PA": np.ones(T + 1)}
    size = level_size(e, cap)
    S = {"jump": np.ones(T + 1), "scurve": scurve(T),
         "mixed": 0.5 + 0.5 * scurve(T)}[case]          # mixed: half at t = 0, half S-curve
    if typ == "dis":
        path["G" + cap] = path["G" + cap] * np.exp(size * S)
    else:
        path["P" + cap] = path["P" + cap] * np.exp(-size * S)
    x = {j: np.full(T + 1, bgp(e)["x"][j]) for j in CAPS}
    return path, x, size, {}


# ----------------------------------------------------------------------------
# Perfect-foresight solver (levels)
# ----------------------------------------------------------------------------
class Model:
    def __init__(self, e, path, x, adj, gr, p=BASE):
        self.e, self.path, self.x, self.p = e, path, x, p
        self.c = {j: (ADJ[j] if adj else 0.0) for j in CAPS}
        self.T = len(path["GK"]) - 1
        self.Z = (1 + p["g_Z"]) ** np.arange(self.T + 1)
        ss0 = bgp(e)
        # initial states (Z_0 = 1): Y_0 = YZ, K = KY*Y, A = AY*Y
        self.K0 = {"K": ss0["KY"] * ss0["YZ"], "A": ss0["AY"] * ss0["YZ"]}
        self.ss0 = ss0
        self.new = bgp(e, gr=gr)
        self.term = np.array([self.new["x"][j] for j in CAPS])

    def sim(self, ii):
        """ii: array (2, T+1) of efficiency investment rates i_j = I_j/(P_j K_j)."""
        p, T, e = self.p, self.T, self.e
        a = p["a"]
        K = {j: np.empty(T + 2) for j in CAPS}
        for j in CAPS:
            K[j][0] = self.K0[j]
            for t in range(T + 1):
                K[j][t + 1] = (1 - p["d"][j] + ii[CAPS.index(j), t]) * K[j][t]
        Kk, Aa = K["K"][:-1], K["A"][:-1]
        X = (self.path["GK"] * Kk) ** a * self.Z ** (1 - a)
        F, FX, FM = prod(X, self.path["GA"] * Aa, e)
        MP = {"K": FX * a * X / Kk, "A": FM * self.path["GA"]}
        W = FX * (1 - a) * X
        P = {"K": self.path["PK"], "A": self.path["PA"]}
        AC, Q, extra, spend = 0.0, {}, {}, 0.0
        for k, j in enumerate(CAPS):
            dev = ii[k] - self.x[j]
            AC = AC + P[j] * 0.5 * self.c[j] * dev ** 2 * K[j][:-1]
            Q[j] = P[j] * (1 + self.c[j] * dev)
            extra[j] = P[j] * (self.c[j] * dev * ii[k] - 0.5 * self.c[j] * dev ** 2)
            spend = spend + P[j] * ii[k] * K[j][:-1]
        Ynet = F - AC
        C = Ynet - spend
        return dict(K=K, F=F, Ynet=Ynet, C=C, W=W, MP=MP, Q=Q, extra=extra, P=P)

    def resid(self, z):
        T = self.T
        ii = np.hstack([z.reshape(2, T), self.term[:, None]])
        with np.errstate(all="ignore"):
            S = self.sim(ii)
            C = S["C"]
            if np.any(~np.isfinite(C)) or np.any(C <= 0) or np.any(ii <= -0.9):
                return np.full(2 * T, 1e3)
            m = self.p["beta"] * C[:-1] / C[1:]
            out = []
            for j in CAPS:
                ret = S["MP"][j][1:] + S["extra"][j][1:] + (1 - self.p["d"][j]) * S["Q"][j][1:]
                out.append(m * ret / S["Q"][j][:-1] - 1)
        res = np.concatenate(out)
        return np.where(np.isfinite(res), res, 1e3)

    def solve(self, tol=1e-10, maxit=60):
        T = self.T
        z = np.concatenate([np.full(T, self.term[k]) for k in range(2)])
        R = self.resid(z)
        n = z.size
        for it in range(maxit):
            nr = np.max(np.abs(R))
            if nr < tol:
                break
            J = np.empty((n, n))
            for i in range(n):
                dz = np.zeros(n); dz[i] = 1e-7
                J[:, i] = (self.resid(z + dz) - R) / 1e-7
            step = np.linalg.solve(J, -R)
            lam = 1.0
            while lam > 1e-6:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, R = zn, Rn
        self.iters, self.err = it, float(np.max(np.abs(R)))
        ii = np.hstack([z.reshape(2, T), self.term[:, None]])
        S = self.sim(ii)
        C = S["C"]
        r = C[1:] / (self.p["beta"] * C[:-1]) - 1
        trend = self.Z
        # deviations from the pre-shock balanced growth path (so impact jumps show up)
        return dict(r=r, lnW=np.log(S["W"] / trend / self.ss0["WZ"]),
                    lnY=np.log(S["Ynet"] / trend / self.ss0["YZ"]),
                    labsh=S["W"] / S["F"], qA=S["Q"]["A"] / S["P"]["A"],
                    qK=S["Q"]["K"] / S["P"]["K"], err=self.err, iters=self.iters,
                    gY=np.log(S["Ynet"][-1] / S["Ynet"][-2]))


# ----------------------------------------------------------------------------
def run(T=250):
    out, checks = {}, {}
    e = 1.5
    # closed-form BGP satisfies the equilibrium conditions
    path, x, _, _ = tech(e, "jump", "K", "dis", 60)
    path = {k: (np.full(61, BASE["GammaA0"]) if k == "GA" else np.ones(61)) for k in path}
    M = Model(e, path, x, OPTS["adj"], {})
    checks["closed-form BGP satisfies equilibrium"] = float(
        np.max(np.abs(M.resid(np.concatenate([np.full(60, M.term[k]) for k in range(2)])))))
    for case in CASES:
        for cap in CAPS:
            for typ in ("dis", "emb"):
                path, x, size, gr = tech(e, case, cap, typ, T)
                res = Model(e, path, x, OPTS["adj"], gr).solve()
                res["size"] = size
                out[(case, cap, typ)] = res
                key = f"{case} {cap} {typ}"
                checks[f"{key}: Newton residual"] = res["err"]
                checks[f"{key}: long-run ln Y gain vs 10%"] = abs(res["lnY"][-1] - BASE["target_level"])
                print(f"  {key:16s} size={size:.4f} err={res['err']:.1e} r1={100*res['r'][0]:.2f} "
                      f"peak={100*res['r'].max():.2f}@{int(np.argmax(res['r']))} "
                      f"lnW0={100*res['lnW'][0]:.2f} lnW10={100*res['lnW'][10]:.2f} "
                      f"lnW_T={100*res['lnW'][-1]:.2f} ls10={100*res['labsh'][10]:.2f}")
    # closed-form long-run wage gains (Proposition 2)
    a = BASE["a"]
    want = {"K": BASE["target_level"], "A": BASE["target_level"] / (1 + (1 - a) * (e - 1))}
    for cap in CAPS:
        for case in CASES:
            for typ in ("dis", "emb"):
                checks[f"{case} {cap} {typ}: long-run ln W vs closed form"] = abs(
                    out[(case, cap, typ)]["lnW"][-1] - want[cap])
    return out, checks


CASES = ("jump", "mixed", "scurve")
TITLES = {"jump": "Surprise permanent jump at t = 0",
          "mixed": "Half jump at t = 0, half S-curve (complete by year 10)",
          "scurve": "S-curve from t = 0, complete by year 10"}
LINES = [(("K", "dis"), "#1f5f9e", "-", "regular capital, disembodied"),
         (("K", "emb"), "#1f5f9e", "--", "regular capital, embodied"),
         (("A", "dis"), "#c0512a", "-", "AI, disembodied"),
         (("A", "emb"), "#c0512a", "--", "AI, embodied")]


def figure(out, H=40):
    t = np.arange(H)
    rows = [("r", "real interest rate (pct.)"), ("lnY", "output, log dev. from\npre-shock path (pct.)"),
            ("lnW", "real wage, log dev. from\npre-shock path (pct.)"), ("labsh", "labor share (pct.)")]
    fig, axes = plt.subplots(4, 3, figsize=(13, 10.5), sharey="row")
    for col, case in enumerate(CASES):
        for (cap, typ), c, ls, lab in LINES:
            res = out[(case, cap, typ)]
            for row, (var, _) in enumerate(rows):
                axes[row, col].plot(t, 100 * res[var][:H], color=c, ls=ls, lw=2, label=lab)
        axes[0, col].axhline(100 * BASE["r0"], color="0.6", lw=1, ls=":")
        axes[1, col].axhline(100 * BASE["target_level"], color="0.6", lw=1, ls=":")
        axes[2, col].axhline(100 * BASE["target_level"], color="0.6", lw=1, ls=":")
        axes[0, col].set_title(TITLES[case], fontsize=10)
        axes[-1, col].set_xlabel("years")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
    for row, (_, lab) in enumerate(rows):
        axes[row, 0].set_ylabel(lab)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Same 10% long-run output gain; Cobb-Douglas K-L bundle, AI substitute (e = 1.5); "
                 + OPTS["label"], fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_capital_cases{OPTS['suffix']}.{ext}"), dpi=150)
    plt.close(fig)


def _f(x):
    return f"{round(x, 1) + 0.0:.1f}"


def summary_table(out):
    L = [r"\begin{tabular}{llcccccccccc}\toprule",
         r" & & \multicolumn{3}{c}{real rate (pct.)} & \multicolumn{3}{c}{output, $\ln Y$ (pct.)} & \multicolumn{3}{c}{real wage, $\ln W$ (pct.)} & labor share\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-12}",
         r" & & $r_1$ & peak & year & $t{=}0$ & $t{=}10$ & $t{=}20$ & $t{=}0$ & $t{=}10$ & $t{=}20$ & $t{=}10$ (pct.)\\\midrule"]
    names = {("K", "dis"): "Regular, disemb.", ("K", "emb"): "Regular, emb.",
             ("A", "dis"): "AI, disemb.", ("A", "emb"): "AI, emb."}
    cnames = {"jump": "Jump", "mixed": "Half/half", "scurve": "S-curve"}
    for case in CASES:
        for i, ((cap, typ), _, _, _) in enumerate(LINES):
            res = out[(case, cap, typ)]
            L.append(f"{cnames[case] if i == 0 else ''} & {names[(cap, typ)]} & {100*res['r'][0]:.2f} & "
                     f"{100*res['r'].max():.2f} & {int(np.argmax(res['r']))} & "
                     f"{_f(100*res['lnY'][0])} & {_f(100*res['lnY'][10])} & {_f(100*res['lnY'][20])} & "
                     f"{_f(100*res['lnW'][0])} & {_f(100*res['lnW'][10])} & {_f(100*res['lnW'][20])} & "
                     f"{100*res['labsh'][10]:.1f}\\\\")
        if case != CASES[-1]:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, f"table_capital_cases{OPTS['suffix']}.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    out, checks = run()
    figure(out)
    summary_table(out)
    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k.replace('%', chr(92) + '%')} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, f"table_capital_checks{OPTS['suffix']}.tex"), "w") as f:
        f.write("\n".join(C) + "\n")
    with open(os.path.join(HERE, f"results_capital_cases{OPTS['suffix']}.json"), "w") as f:
        json.dump(dict(checks=checks, adj=ADJ,
                       sizes={cap: level_size(1.5, cap) for cap in CAPS},
                       summary={" ".join(k): dict(r1=float(v["r"][0]), peak=float(v["r"].max()),
                                                  tpeak=int(np.argmax(v["r"])),
                                                  lnW=[float(v["lnW"][i]) for i in (0, 5, 10, 20, 30)],
                                                  lnY=[float(v["lnY"][i]) for i in (0, 5, 10, 20, 30)],
                                                  lnWT=float(v["lnW"][-1]),
                                                  labsh=[float(v["labsh"][i]) for i in (0, 5, 10, 20)],
                                                  labshT=float(v["labsh"][-1]),
                                                  cum_r40=float(np.sum(np.log1p(v["r"][:40]) - np.log1p(BASE["r0"]))))
                                for k, v in out.items()}), f, indent=1)
    print("max check:", max(checks.values()))


if __name__ == "__main__":
    import sys
    if "--equal-d-noadj" in sys.argv:
        configure(dA=BASE["d"]["K"], adj=False, suffix="_eqd_noadj", scurve="ramp", T_end=10,
                  label="d_A = d_K = 0.066, no adjustment costs")
    main()
