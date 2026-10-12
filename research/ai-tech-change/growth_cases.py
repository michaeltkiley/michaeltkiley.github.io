"""
AI capital versus regular capital: level jumps, faster growth, and an anticipated S-curve.

Baseline (case 3 of ai_cases.tex):
    X = (Gamma_K K)^a (Z N)^(1-a)                                   Cobb-Douglas bundle
    Y = [w^(1/e) X^rho + (1-w)^(1/e) (Gamma_A A)^rho]^(1/rho),  e = 1.5  (AI substitutes)
    K' = (1-d_K) K + I_K/P_K,     A' = (1-d_A) A + I_A/P_A            (linear accumulation)
    optional adjustment costs  P_j 0.5 c_j (I_j/(P_j K_j) - x_j)^2 K_j, x_j = BGP rate
    log utility, fixed hours.
Disembodied progress: Gamma_j (production).  Embodied progress: P_j falls (accumulation).

Experiments (each for AI and regular capital, embodied and disembodied, linear and
adjustment-cost accumulation):
  Case 1  surprise permanent level jump, sized for the same 10% long-run output gain
  Case 2  surprise permanent increase in the growth rate of technology, sized for the
          same +0.5 pp long-run output growth.  With e > 1 there is no BGP under ongoing
          AI progress (AI share -> 1), so AI growth is studied in the e = 1 variant;
          regular-capital growth is shown in both.
  Case 3  S-curve level shift centered on year 10, announced at t = 0, same 10% gain.

Closed forms (balanced growth path) are in bgp(); the transitions are perfect-foresight
Newton solves of the nonlinear equilibrium.
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
            target_level=0.10, target_growth=0.005, mid=10.0, width=2.5)
BASE["beta"] = (1 + BASE["g_Z"]) / (1 + BASE["r0"])          # log utility
BASE["omega"] = 1 - BASE["sA0"]
BASE["GammaA0"] = BASE["r0"] + BASE["d"]["A"]                # lambda_A = 1 initially
CAPS = ("K", "A")
# adjustment costs: elasticity of I/K with respect to Q equal to one, 1/(c x) = 1
# (Kiley 2000's "moderate" case)
ADJ = {j: 1.0 / (BASE["g_Z"] + BASE["d"][j]) for j in CAPS}


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


def growth_size(e, cap, p=BASE):
    """Annual rate gamma of Gamma_j growth (or P_j decline) raising BGP growth by target_growth."""
    a, om = p["a"], p["omega"]
    GZ, Gn = 1 + p["g_Z"], 1 + p["g_Z"] + p["target_growth"]
    if cap == "K":
        expo = a / (1 - a)                                  # same for e = 1 and e != 1
    else:
        if e != 1.0:
            return None
        expo = (1 - om) / ((1 - a) * om)
    return (Gn / GZ) ** (1 / expo) - 1


def tech(e, case, cap, typ, T, p=BASE):
    """Paths of Gamma_K, P_K, Gamma_A, P_A, and the adjustment-cost targets x_j(t)."""
    t = np.arange(T + 1)
    path = {"GK": np.ones(T + 1), "PK": np.ones(T + 1),
            "GA": np.full(T + 1, p["GammaA0"]), "PA": np.ones(T + 1)}
    gr = {}
    if case in ("level", "scurve"):
        size = level_size(e, cap)
        S = np.ones(T + 1) if case == "level" else scurve(T)
        if typ == "dis":
            path["G" + cap] = path["G" + cap] * np.exp(size * S)
        else:
            path["P" + cap] = path["P" + cap] * np.exp(-size * S)
    else:                                                    # permanent growth from t = 0
        size = growth_size(e, cap)
        if typ == "dis":
            path["G" + cap] = path["G" + cap] * (1 + size) ** t
            gr["gG" + cap] = size
        else:
            path["P" + cap] = path["P" + cap] * (1 + size) ** (-t)
            gr["gq" + cap] = size
    new = bgp(e, gr=gr)
    x = {j: np.full(T + 1, new["x"][j]) for j in CAPS}
    return path, x, size, gr


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
def run():
    T = {"level": 250, "scurve": 250, "growth": 250}
    out, checks = {}, {}
    # BGP closed form reproduces itself in the solver (no shock)
    for e in (1.5, 1.0):
        for adj in (False, True):
            path, x, _, _ = tech(e, "level", "K", "dis", 60)
            path = {k: (np.ones(61) if k != "GA" else np.full(61, BASE["GammaA0"])) for k in path}
            M = Model(e, path, x, adj, {})
            z = np.concatenate([np.full(60, M.term[k]) for k in range(2)])
            checks[f"e={e} adj={adj}: closed-form BGP satisfies equilibrium"] = float(
                np.max(np.abs(M.resid(z))))
    specs = []
    for case in ("level", "growth", "scurve"):
        for e in ((1.5, 1.0) if case == "growth" else (1.5,)):
            for cap in CAPS:
                if case == "growth" and e != 1.0 and cap == "A":
                    continue
                for typ in ("dis", "emb"):
                    for adj in (False, True):
                        specs.append((case, e, cap, typ, adj))
    for spec in specs:
        case, e, cap, typ, adj = spec
        path, x, size, gr = tech(e, case, cap, typ, T[case])
        M = Model(e, path, x, adj, gr)
        res = M.solve()
        res.update(size=size, new=M.new, ss0=M.ss0)
        out[spec] = res
        key = f"{case} e={e} {cap} {typ} {'AC' if adj else 'lin'}"
        checks[f"{key}: Newton residual"] = res["err"]
        if case == "growth":
            checks[f"{key}: growth at T vs closed form"] = abs(np.exp(res["gY"]) - M.new["G"])
        else:
            want = BASE["target_level"]
            checks[f"{key}: long-run ln Y gain vs target"] = abs(res["lnY"][-1] - want)
        print(f"  {key:32s} size={size:.4f} iters={res['iters']:2d} err={res['err']:.1e} "
              f"r0={100*res['r'][0]:.2f} peak={100*res['r'].max():.2f} "
              f"lnW10={100*res['lnW'][10]:.2f} lnW_T={100*res['lnW'][-1]:.2f}")
    return out, checks


# ----------------------------------------------------------------------------
COL = {("dis", False): "#1f5f9e", ("dis", True): "#1f5f9e",
       ("emb", False): "#c0512a", ("emb", True): "#c0512a"}


def figure(out, case, e_by_cap, fname, title, H=40):
    t = np.arange(H)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.4))
    for col, cap in enumerate(CAPS):
        e = e_by_cap[cap]
        for typ in ("dis", "emb"):
            for adj in (False, True):
                res = out[(case, e, cap, typ, adj)]
                ls = "--" if adj else "-"
                lab = f"{'disembodied' if typ == 'dis' else 'embodied'}, {'adj. costs' if adj else 'linear'}"
                axes[0, col].plot(t, 100 * res["r"][:H], color=COL[(typ, adj)], ls=ls, lw=1.8, label=lab)
                axes[1, col].plot(t, 100 * res["lnW"][:H], color=COL[(typ, adj)], ls=ls, lw=1.8)
                axes[2, col].plot(t, 100 * res["labsh"][:H], color=COL[(typ, adj)], ls=ls, lw=1.8)
        axes[0, col].axhline(100 * BASE["r0"], color="0.6", lw=1, ls=":")
        etxt = f"e = {e:g}"
        axes[0, col].set_title(("Regular capital K" if cap == "K" else "AI capital A") + f" ({etxt})",
                               fontsize=10)
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
        lo, hi = axes[2, col].get_ylim()
        if hi - lo < 2:
            c = 0.5 * (lo + hi); axes[2, col].set_ylim(c - 1, c + 1)
        axes[2, col].set_xlabel("years after announcement")
    axes[0, 0].set_ylabel("real interest rate (pct.)")
    axes[1, 0].set_ylabel("real wage, log dev. from\ninitial trend (pct.)")
    axes[2, 0].set_ylabel("labor share (pct.)")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{fname}.{ext}"), dpi=150)
    plt.close(fig)


def table(out, case, e_by_cap, fname):
    L = [r"\begin{tabular}{llccccccc}\toprule",
         r" & & shock & $r_1$ & peak $r$ & year & $\ln W$, $t{=}10$ & $\ln W$, long run & labor share\\",
         r" & & & (pct.) & (pct.) & & (pct.) & (pct.) & long run (pct.)\\\midrule"]
    for cap in CAPS:
        e = e_by_cap[cap]
        for typ in ("dis", "emb"):
            for adj in (False, True):
                res = out[(case, e, cap, typ, adj)]
                name = ("AI" if cap == "A" else "$K$") + (" disemb." if typ == "dis" else " emb.")
                acc = "adj. costs" if adj else "linear"
                if case == "growth":
                    lr = f"+{100*res['new']['g'] - 100*BASE['g_Z']:.2f} pp/yr"
                    lsh = 100 * res["new"]["labsh"]
                else:
                    lr = f"{100*res['lnW'][-1]:.1f}"
                    lsh = 100 * res["labsh"][-1]
                L.append(f"{name} & {acc} & {res['size']:.3f} & {100*res['r'][0]:.2f} & "
                         f"{100*res['r'].max():.2f} & {int(np.argmax(res['r']))} & "
                         f"{100*res['lnW'][10]:.1f} & {lr} & {lsh:.1f}\\\\")
        if cap == "K":
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, f"{fname}.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def table_growth(out, e_by_cap, fname):
    L = [r"\begin{tabular}{llcccccccc}\toprule",
         r" & & $\gamma$ & \multicolumn{4}{c}{real interest rate (pct.)} & \multicolumn{2}{c}{$\ln W$ vs. old trend (pct.)} & $W/Z$ level\\",
         r" & & (pct./yr) & $r_1$ & $t{=}5$ & $t{=}20$ & new BGP & $t{=}10$ & $t{=}30$ & new/old BGP\\\midrule"]
    for cap in CAPS:
        e = e_by_cap[cap]
        for typ in ("dis", "emb"):
            for adj in (False, True):
                res = out[("growth", e, cap, typ, adj)]
                name = ("AI" if cap == "A" else "$K$") + (" disemb." if typ == "dis" else " emb.")
                acc = "adj. costs" if adj else "linear"
                lvl = res["new"]["WZ"] / res["ss0"]["WZ"]
                L.append(f"{name} & {acc} & {100*res['size']:.2f} & {100*res['r'][0]:.2f} & "
                         f"{100*res['r'][4]:.2f} & {100*res['r'][19]:.2f} & {100*res['new']['r']:.2f} & "
                         f"{100*res['lnW'][10]:.1f} & {100*res['lnW'][30]:.1f} & {lvl:.3f}\\\\")
        if cap == "K":
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, f"{fname}.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def bgp_table():
    rows = [("Initial", 1.5, {}), ("$K$ disembodied, $\\gamma$", 1.5, {"gGK": growth_size(1.5, "K")}),
            ("$K$ embodied, $\\gamma$", 1.5, {"gqK": growth_size(1.5, "K")})]
    rows1 = [("Initial", 1.0, {}), ("$K$ disembodied", 1.0, {"gGK": growth_size(1.0, "K")}),
             ("$K$ embodied", 1.0, {"gqK": growth_size(1.0, "K")}),
             ("AI disembodied", 1.0, {"gGA": growth_size(1.0, "A")}),
             ("AI embodied", 1.0, {"gqA": growth_size(1.0, "A")})]
    L = [r"\begin{tabular}{llccccccc}\toprule",
         r" & & $\gamma$ & $g$ & $r$ & $u_K$ & $u_A$ & $W/Z$ & labor share\\\midrule"]
    for blk, (rr, lab) in enumerate(((rows, r"$\e=1.5$"), (rows1, r"$\e=1$"))):
        w0 = bgp(rr[0][1])["WZ"]
        for name, e, gr in rr:
            b = bgp(e, gr=gr)
            gam = max(gr.values()) if gr else 0.0
            L.append(f"{lab if name == 'Initial' else ''} & {name} & {100*gam:.2f} & {100*b['g']:.2f} & "
                     f"{100*b['r']:.2f} & {100*b['u']['K']:.2f} & {100*b['u']['A']:.2f} & "
                     f"{b['WZ']/w0:.3f} & {100*b['labsh']:.1f}\\\\")
        if blk == 0:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_g_bgp.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    out, checks = run()
    E = {"K": 1.5, "A": 1.5}
    figure(out, "level", E, "fig_g_case1",
           "Case 1: surprise permanent jump, same 10% long-run output gain (baseline, e = 1.5)")
    figure(out, "scurve", E, "fig_g_case3",
           "Case 3: S-curve centered on year 10, announced at t = 0, same 10% long-run output gain")
    figure(out, "growth", {"K": 1.0, "A": 1.0}, "fig_g_case2",
           "Case 2: permanent rise in technology growth, +0.5 pp long-run output growth (e = 1 variant)")
    table(out, "level", E, "table_g_case1")
    table(out, "scurve", E, "table_g_case3")
    table_growth(out, {"K": 1.0, "A": 1.0}, "table_g_case2")
    bgp_table()
    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_g_checks.tex"), "w") as f:
        f.write("\n".join(C) + "\n")
    sizes = {f"level {cap}": level_size(1.5, cap) for cap in CAPS}
    sizes.update({f"growth e={e} {cap}": growth_size(e, cap) for e in (1.5, 1.0) for cap in CAPS})
    with open(os.path.join(HERE, "results_growth_cases.json"), "w") as f:
        json.dump(dict(checks=checks, sizes=sizes, adj=ADJ,
                       summary={" ".join(map(str, k)): dict(r1=float(v["r"][0]), peak=float(v["r"].max()),
                                                             tpeak=int(np.argmax(v["r"])),
                                                             lnW10=float(v["lnW"][10]),
                                                             lnWT=float(v["lnW"][-1]),
                                                             labshT=float(v["labsh"][-1]))
                                for k, v in out.items()}), f, indent=1, default=float)
    print("Checks (max):", max(checks.values()))
    for k, v in checks.items():
        if v > 1e-6:
            print("  LARGE:", k, v)


if __name__ == "__main__":
    main()
