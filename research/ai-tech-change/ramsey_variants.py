"""
Real-rate response to AI vs regular-capital progress, within the Ramsey (Euler-equation) model.

Production, accumulation and shocks are those of capital_cases.py with equal depreciation
(d_A = d_K = 0.066), e = 1.5, a cosine S-curve complete by year 10, and a 10% long-run output gain.
New here: the household block has three switches.

  theta   curvature (1/IES):  U = sum B^t X_t^(1-theta)/(1-theta),  X_t = C_t - h C_{t-1}
  h       internal additive habits
  lam     share of hand-to-mouth households who consume their labor income; Ricardian households
          own all capital and satisfy the Euler equation
              lambda_t = X_t^-theta - B h X_{t+1}^-theta,   lambda_t = B R_{t+1} lambda_{t+1}
          with B = (1+g)^theta/(1+r0), so the steady-state r (7.5%) and its invariance to level
          shifts do not depend on theta, h or lam.

Accumulation: adjustment costs P_j (c_j/2)(i_j - x_j)^2 K_j.  c_K = 1/x (unit elasticity of the
investment rate to Q).  c_A is calibrated so that AI capital converges to a small shock as fast as
regular capital does (see calibrate_cA); alternatives are shown as sensitivity.

Identity (h = 0):  ln(1+r_{t+1}) - ln(1+r0) = theta * [dln c_{t+1} - g], with c the Ricardian
consumption, so the first-decade sum of r deviations equals theta * [ln(c_10/Z_10) - ln(c_0/Z_0)].
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import capital_cases as cc

HERE, FIG = cc.HERE, cc.FIG
cc.configure(dA=cc.BASE["d"]["K"], adj=True, scurve="ramp", T_end=10)
BASE = cc.BASE
CAPS = cc.CAPS
g, r0, G = BASE["g_Z"], BASE["r0"], 1 + BASE["g_Z"]
E = 1.5
X_SS = g + BASE["d"]["K"]                 # steady-state investment rate (same for both capitals)
C_K = 1.0 / X_SS                          # unit elasticity of I/K to Q
T_DEF = 250


class HH:
    """kind = "tank": the lam share consumes its labor income (hand-to-mouth).
    kind = "saver": the lam share saves a constant fraction s of gross income (its wage plus its share
    of gross capital income F - W; revaluation of existing capital is not income) and so
    accumulates wealth; s is calibrated so the group's wealth grows at g on the initial BGP
    (s = 17.5%, the aggregate gross investment rate)."""

    def __init__(self, theta=1.0, h=0.0, lam=0.0, name="", kind="tank"):
        self.theta, self.h, self.lam, self.name, self.kind = theta, h, lam, name, kind


class Model2(cc.Model):
    """capital_cases.Model with a general Ricardian household block."""

    def __init__(self, path, x, c, hh):
        super().__init__(E, path, x, True, {})
        self.c = dict(c)
        self.hh = hh
        self.beta_ = G ** hh.theta / (1 + r0)
        ss0 = self.ss0
        spend0 = sum(self.K0[j] * x[j][0] for j in CAPS)       # pre-shock BGP investment (P = 1)
        c_pre = ss0["YZ"] - spend0
        w_pre = ss0["WZ"]
        self.V_pre = self.K0["K"] + self.K0["A"]                  # wealth at end of t = -1 (Q = P = 1)
        self.c_pre = c_pre
        if hh.kind == "saver":
            c_rot_pre = hh.lam * (w_pre + (r0 - g) * self.V_pre)      # group consumption on the initial BGP
            self.s_rot = 1 - c_rot_pre / (hh.lam * ss0["YZ"])
        else:
            self.s_rot = None
            c_rot_pre = hh.lam * w_pre
        self.CRm1 = ((c_pre - c_rot_pre) / (1 - hh.lam)) / G

    def rot_consumption(self, S):
        """Aggregate consumption of the rule-of-thumb group, t = 0..T."""
        hh, T = self.hh, self.T
        if hh.kind == "tank" or hh.lam == 0:
            return hh.lam * S["W"]
        d_ = self.p["d"]
        # Income = wage + share of aggregate net capital income (revaluation of existing capital
        # excluded); cash on hand includes the full payoff, so wealth still absorbs gains and losses.
        Kt = {j: S["K"][j][:T + 1] for j in CAPS}                    # units at start of t
        Kn = {j: S["K"][j][1:T + 2] for j in CAPS}                   # units at end of t
        Vagg = sum(S["Q"][j] * Kn[j] for j in CAPS)                  # end-of-t value, t = 0..T
        Vlag = np.concatenate([[self.V_pre], Vagg[:-1]])
        Gpay = sum(Kt[j] * (S["MP"][j] + S["extra"][j] + (1 - d_[j]) * S["Q"][j]) for j in CAPS)
        Pi = S["F"] - S["W"]                                         # gross capital income
        C = np.empty(T + 1)
        phis = np.empty(T + 1)
        VR = hh.lam * self.V_pre
        for t in range(T + 1):
            phi = VR / Vlag[t]
            phis[t] = phi
            y = hh.lam * S["W"][t] + phi * Pi[t]
            C[t] = (1 - self.s_rot) * y
            VR = phi * Gpay[t] + hh.lam * S["W"][t] - C[t]
        self.last_phi = phis
        return C

    def household(self, S):
        hh, T = self.hh, self.T
        CR = (S["C"] - self.rot_consumption(S)) / (1 - hh.lam)
        CRx = np.concatenate([[self.CRm1], CR, [G * CR[-1]]])          # t = -1 .. T+1
        X = CRx[1:] - hh.h * CRx[:-1]                                   # t = 0 .. T+1
        return CR, X

    def resid(self, z):
        T, hh = self.T, self.hh
        ii = np.hstack([z.reshape(2, T), self.term[:, None]])
        with np.errstate(all="ignore"):
            S = self.sim(ii)
            CR, X = self.household(S)
            if (np.any(~np.isfinite(CR)) or np.any(CR <= 0) or np.any(X <= 0) or np.any(ii <= -0.9)):
                return np.full(2 * T, 1e3)
            lamU = X[:-1] ** (-hh.theta) - self.beta_ * hh.h * X[1:] ** (-hh.theta)   # t = 0 .. T
            if np.any(lamU <= 0):
                return np.full(2 * T, 1e3)
            m = self.beta_ * lamU[1:] / lamU[:-1]                                    # t = 0 .. T-1
            out = []
            for j in CAPS:
                ret = S["MP"][j][1:] + S["extra"][j][1:] + (1 - self.p["d"][j]) * S["Q"][j][1:]
                out.append(m * ret / S["Q"][j][:-1] - 1)
        res = np.concatenate(out)
        return np.where(np.isfinite(res), res, 1e3)

    def solve(self, tol=1e-10, maxit=60, z0=None):
        T = self.T
        z = np.concatenate([np.full(T, self.term[k]) for k in range(2)]) if z0 is None else z0.copy()
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
            try:
                step = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(J, -R, rcond=None)[0]
            lam = 1.0
            while lam > 1e-6:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, R = zn, Rn
        if np.max(np.abs(R)) > tol:                                   # Levenberg-Marquardt polish
            mu = 1e-3
            for it in range(150):
                if np.max(np.abs(R)) < tol:
                    break
                J = np.empty((n, n))
                for i in range(n):
                    dz = np.zeros(n); dz[i] = 1e-7
                    J[:, i] = (self.resid(z + dz) - R) / 1e-7
                JTJ, gr = J.T @ J, J.T @ R
                f0 = R @ R
                while mu < 1e12:
                    step = np.linalg.solve(JTJ + mu * np.diag(np.diag(JTJ) + 1e-12), -gr)
                    Rn = self.resid(z + step)
                    if Rn @ Rn < f0:
                        z, R, mu = z + step, Rn, max(mu / 5, 1e-12)
                        break
                    mu *= 10
        self.err = float(np.max(np.abs(R)))
        self.zsol = z
        ii = np.hstack([z.reshape(2, T), self.term[:, None]])
        S = self.sim(ii)
        CR, X = self.household(S)
        lamU = X[:-1] ** (-self.hh.theta) - self.beta_ * self.hh.h * X[1:] ** (-self.hh.theta)
        m = self.beta_ * lamU[1:] / lamU[:-1]
        Z = self.Z
        rotC = self.rot_consumption(S)
        return dict(r=1 / m - 1, lnY=np.log(S["Ynet"] / Z / self.ss0["YZ"]),
                    lnW=np.log(S["W"] / Z / self.ss0["WZ"]), labsh=S["W"] / S["F"],
                    cR=CR / Z, K=S["K"]["K"], A=S["K"]["A"], err=self.err, z=z,
                    lnC=np.log(S["C"] / Z / self.c_pre), invsh=(S["Ynet"] - S["C"]) / S["Ynet"],
                    lnYg=np.log(S["F"] / Z / self.ss0["YZ"]), invg=(S["F"] - S["C"]) / S["F"],
                    invg0=(self.ss0["YZ"] - self.c_pre) / self.ss0["YZ"],
                    rotC=rotC / Z, ricC=(1 - self.hh.lam) * CR / Z, Ynet=S["Ynet"] / Z,
                    phi=getattr(self, "last_phi", None) if self.hh.kind == "saver" else None)


def run(case, cap, typ, hh, cA, T=T_DEF, cK=C_K, size_scale=1.0, z0=None):
    path, x, size, gr = cc.tech(E, case, cap, typ, T)
    m = Model2(path, x, {"K": cK, "A": cA}, hh)
    res = m.solve(z0=z0)
    res["size"] = size
    return res


# ----------------------------------------------------------------------------
# Adjustment-cost calibration: equal speed of convergence for AI and regular capital
# ----------------------------------------------------------------------------
def decay_rate(cap, cA, cK=C_K, size=0.02, T=120):
    """Annual convergence rate of capital `cap` to a small permanent disembodied shock in Gamma_cap."""
    path = {"GK": np.ones(T + 1), "PK": np.ones(T + 1),
            "GA": np.full(T + 1, BASE["GammaA0"]), "PA": np.ones(T + 1)}
    path["G" + cap] = path["G" + cap] * np.exp(size)
    x = {j: np.full(T + 1, cc.bgp(E)["x"][j]) for j in CAPS}
    m = Model2(path, x, {"K": cK, "A": cA}, HH())
    res = m.solve()
    lev = {"GK": 1.0, "PK": 1.0, "GA": BASE["GammaA0"], "PA": 1.0}
    lev["G" + cap] *= np.exp(size)
    new = cc.bgp(E, lev=lev)
    Kn = {"K": new["KY"] * new["YZ"], "A": new["AY"] * new["YZ"]}[cap] * G ** np.arange(T + 1)
    K = res["K"] if cap == "K" else res["A"]
    dev = np.abs(np.log(K[:T + 1] / Kn))
    t = np.arange(8, 30)
    return -np.polyfit(t, np.log(dev[t]), 1)[0]


def calibrate_cA():
    kK = decay_rate("K", C_K)
    lo, hi = 0.2 * C_K, 6 * C_K
    f = lambda c: decay_rate("A", c) - kK
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None, kK, decay_rate("A", C_K)
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return np.sqrt(lo * hi), kK, decay_rate("A", C_K)


# ----------------------------------------------------------------------------
VARIANTS = [HH(1, 0, 0, "Base"), HH(2, 0, 0, "IES 0.5"), HH(5, 0, 0, "IES 0.2"),
            HH(1, 0.7, 0, "Habits"), HH(1, 0, 0.3, "TANK 30%"), HH(1, 0, 0.5, "TANK 50%"),
            HH(2, 0.7, 0.3, "All three")]
SCEN = [("K", "dis"), ("K", "emb"), ("A", "dis"), ("A", "emb")]
SNAME = {("K", "dis"): "Regular, disembodied", ("K", "emb"): "Regular, embodied",
         ("A", "dis"): "AI, disembodied", ("A", "emb"): "AI, embodied"}
COL = {("K", "dis"): "#1f5f9e", ("K", "emb"): "#1f5f9e", ("A", "dis"): "#c0512a", ("A", "emb"): "#c0512a"}
LSTY = {"dis": "-", "emb": "--"}


def decade(res, hh):
    r = res["r"]
    dev = np.log1p(r[:10]) - np.log1p(r0)
    return dict(avg=100 * np.mean(r[:10] - r0), r1=100 * r[0], peak=100 * np.max(r[:10]),
                tpeak=int(np.argmax(r[:10])), trough=100 * np.min(r[:10]),
                cum=100 * np.sum(dev), ident=100 * hh.theta * (np.log(res["cR"][10]) - np.log(res["cR"][0])),
                dC=100 * (np.log(res["cR"][10]) - np.log(res["cR"][0])),
                W10=100 * res["lnW"][10], Y10=100 * res["lnY"][10], LS10=100 * res["labsh"][10],
                LSLR=100 * res["labsh"][-30])


def main():
    checks, out = {}, {}
    cA_star, kK, kA_same = calibrate_cA()
    print(f"c_K = {C_K:.3f}; convergence rate K = {kK:.4f}; AI at c_A = c_K: {kA_same:.4f}; "
          f"equal-speed c_A = {cA_star}")
    cA_main = cA_star if cA_star is not None else C_K

    # ---- main grid
    for hh in VARIANTS:
        for case in ("scurve", "jump"):
            for cap, typ in SCEN:
                res = run(case, cap, typ, hh, cA_main)
                out[(hh.name, case, cap, typ)] = res
                d_ = decade(res, hh)
                key = f"{hh.name} {case} {cap} {typ}"
                checks[f"{key}: residual"] = res["err"]
                checks[f"{key}: long-run ln Y = 10%"] = abs(res["lnY"][-30] - cc.BASE["target_level"])
                if hh.h == 0:
                    checks[f"{key}: identity sum ln(1+r) = theta dln c"] = abs(d_["cum"] - d_["ident"]) / 100
                print(f"  {hh.name:10s} {case:6s} {cap} {typ}  avg={d_['avg']:+.2f} r1={d_['r1']:.2f} "
                      f"peak={d_['peak']:.2f}@{d_['tpeak']} trough={d_['trough']:.2f} dC={d_['dC']:+.2f} "
                      f"W10={d_['W10']:.1f} LS10={d_['LS10']:.1f} err={res['err']:.0e}", flush=True)
    # ---- BGP consistency for every household variant
    for hh in VARIANTS:
        T = 60
        path = {"GK": np.ones(T + 1), "PK": np.ones(T + 1), "GA": np.full(T + 1, BASE["GammaA0"]),
                "PA": np.ones(T + 1)}
        x = {j: np.full(T + 1, cc.bgp(E)["x"][j]) for j in CAPS}
        m = Model2(path, x, {"K": C_K, "A": cA_main}, hh)
        checks[f"{hh.name}: BGP satisfies the Euler equations"] = float(
            np.max(np.abs(m.resid(np.concatenate([np.full(T, m.term[k]) for k in range(2)])))))

    # ---- adjustment-cost sensitivity (base household)
    CALS = [("equal speed", cA_main, C_K), ("same c", C_K, C_K), ("c x 0.5", 0.5 * C_K, 0.5 * C_K),
            ("c x 2", 2 * C_K, 2 * C_K), ("none", 0.0, 0.0)]
    sens = {}
    for cname, cA, cK in CALS:
        for case in ("scurve", "jump"):
            for cap, typ in SCEN:
                res = run(case, cap, typ, VARIANTS[0], cA, cK=cK)
                sens[(cname, case, cap, typ)] = decade(res, VARIANTS[0])
                checks[f"sens {cname} {case} {cap} {typ}: residual"] = res["err"]
        print("  sensitivity done:", cname, flush=True)

    # ---------------- figures
    t = np.arange(16)
    def paths_fig(variants, fname, title):
        fig, axes = plt.subplots(2, len(variants), figsize=(3.6 * len(variants), 6.4), sharey="row")
        for col, hh in enumerate(variants):
            for row, case in enumerate(("scurve", "jump")):
                ax = axes[row, col]
                for cap, typ in SCEN:
                    res = out[(hh.name, case, cap, typ)]
                    ax.plot(t, 100 * res["r"][:16], color=COL[(cap, typ)], ls=LSTY[typ], lw=1.9,
                            label=SNAME[(cap, typ)])
                ax.axhline(100 * r0, color="0.6", lw=1, ls=":")
                ax.grid(alpha=0.25)
                if row == 0:
                    ax.set_title(hh.name, fontsize=10)
                if row == 1:
                    ax.set_xlabel("years")
        axes[0, 0].set_ylabel("real rate, S-curve (pct.)")
        axes[1, 0].set_ylabel("real rate, jump (pct.)")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"{fname}.{ext}"), dpi=150)
        plt.close(fig)
    paths_fig(VARIANTS[:3], "fig_r_ies", "Real rate: base household and lower intertemporal elasticity (Ramsey)")
    paths_fig([VARIANTS[0]] + VARIANTS[3:], "fig_r_hh", "Real rate: habits, hand-to-mouth households, and all three")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    xs = np.arange(len(VARIANTS))
    w = 0.2
    for ax, case, ttl in ((axes[0], "scurve", "S-curve (complete by year 10)"), (axes[1], "jump", "Surprise jump")):
        for k, (cap, typ) in enumerate(SCEN):
            vals = [decade(out[(hh.name, case, cap, typ)], hh)["avg"] for hh in VARIANTS]
            ax.bar(xs + (k - 1.5) * w, vals, w, color=COL[(cap, typ)], alpha=1.0 if typ == "dis" else 0.55,
                   hatch="" if typ == "dis" else "//", edgecolor="white", label=SNAME[(cap, typ)])
        ax.set_xticks(xs); ax.set_xticklabels([h_.name for h_ in VARIANTS], fontsize=8, rotation=20)
        ax.set_title(ttl, fontsize=10); ax.grid(axis="y", alpha=0.25); ax.axhline(0, color="0.4", lw=0.8)
    axes[0].set_ylabel("average real rate, years 1-10, minus 7.5% (pp)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("First-decade real-rate effect by shock type and household block (same 10% long-run output gain)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_r_decade.{ext}"), dpi=150)
    plt.close(fig)

    # ---------------- tables
    L = [r"\begin{tabular}{llccccccc}\toprule",
         " & & " + " & ".join(h_.name.replace("%", r"\%") for h_ in VARIANTS) + r"\\\midrule"]
    for case, cname in (("scurve", "S-curve"), ("jump", "Jump")):
        for i, (cap, typ) in enumerate(SCEN):
            vals = [decade(out[(hh.name, case, cap, typ)], hh)["avg"] for hh in VARIANTS]
            L.append(f"{cname if i == 0 else ''} & {SNAME[(cap, typ)]} & " +
                     " & ".join(f"{v:+.2f}" for v in vals) + r"\\")
        if case == "scurve":
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_r_decade.tex"), "w").write("\n".join(L) + "\n")

    D = [r"\begin{tabular}{llccccccc}\toprule",
         r" & & $\Delta\ln c$, yr 0$\to$10 & $\theta\Delta\ln c/10$ & $r_1$ & peak $r$ (yr) & $\ln W$, yr 10 & $\ln Y$, yr 10 & labor share, yr 10\\\midrule"]
    for case, cname in (("scurve", "S-curve"), ("jump", "Jump")):
        for i, (cap, typ) in enumerate(SCEN):
            d_ = decade(out[("Base", case, cap, typ)], VARIANTS[0])
            D.append(f"{cname if i == 0 else ''} & {SNAME[(cap, typ)]} & {d_['dC']:.1f} & {d_['ident']/10:+.2f} & "
                     f"{d_['r1']:.2f} & {d_['peak']:.2f} ({d_['tpeak']}) & {d_['W10']:.1f} & {d_['Y10']:.1f} & "
                     f"{d_['LS10']:.1f}\\\\")
        if case == "scurve":
            D.append(r"\midrule")
    D.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_r_drivers.tex"), "w").write("\n".join(D) + "\n")

    S_ = [r"\begin{tabular}{llccccc}\toprule",
          " & & " + " & ".join(c_[0] for c_ in CALS) + r"\\\midrule"]
    for case, cname in (("scurve", "S-curve"), ("jump", "Jump")):
        for i, (cap, typ) in enumerate(SCEN):
            S_.append(f"{cname if i == 0 else ''} & {SNAME[(cap, typ)]} & " +
                      " & ".join(f"{sens[(c_[0], case, cap, typ)]['avg']:+.2f}" for c_ in CALS) + r"\\")
        if case == "scurve":
            S_.append(r"\midrule")
    S_.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_r_adjsens.tex"), "w").write("\n".join(S_) + "\n")

    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    worst = {}
    for k, v in checks.items():
        key = k.split(": ", 1)[1] if ": " in k else k
        key = key.replace("%", r"\%")
        worst[key] = max(worst.get(key, 0.0), v)
    for k, v in worst.items():
        C.append(f"{k} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_r_checks.tex"), "w").write("\n".join(C) + "\n")
    json.dump(dict(cK=C_K, cA_equal_speed=cA_star, conv_K=kK, conv_A_same_c=kA_same, checks=checks,
                   decade={f"{k[0]} | {k[1]} {k[2]} {k[3]}": decade(v, next(h_ for h_ in VARIANTS if h_.name == k[0]))
                           for k, v in out.items()},
                   sens={" | ".join(k): v for k, v in sens.items()}),
              open(os.path.join(HERE, "results_r_variants.json"), "w"), indent=1, default=float)
    print("max check:", max(checks.values()))
    for k, v in checks.items():
        if v > 1e-6:
            print("  LARGE", k, v)


if __name__ == "__main__":
    main()
