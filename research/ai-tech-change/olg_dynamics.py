"""
Annual OLG (60 cohorts, work ages 1-45) versus Ramsey: embodied AI / regular-capital progress with
adjustment costs, d_A = d_K, same calibration as embodied_dynamics.py (ACs halved, 5-year S-curve).

Household blocks: log utility, and log utility with internal habits (h = 0.7) at the cohort level,
    U = sum_k beta^k ln(c_k - h c_{k-1}),   habit stock c_{-1} = 0 at entry (age 20); cohorts alive
    when the shock hits keep their own pre-shock habit stock.
Cohort Euler: lam_k = 1/X_k - beta h/X_{k+1} = beta R lam_{k+1},  X = c - h c_{-1}.
It is solved exactly for any return path: with u = 1/lam_0 and Y_k = sum_m (beta h)^m ell_k+m,
ell_k = Rc_0/(beta^k Rc_k),   X_k = u/Y_k,  c_k = X_k + h c_{k-1},  and u from the lifetime budget.

Equilibrium: (i) the two capital stocks earn the same return, (ii) resource consumption (output net of
adjustment costs minus investment) equals the sum of cohort consumptions.  beta is calibrated so that
the initial balanced growth path has r = 7.5% with the same aggregate consumption share as Ramsey.
Long-run r after the shock comes from the closed-form fixed point in r (as in saving_models.py).
"""

import json
import os

import numpy as np

import saving_models as sm
import ramsey_variants as rv

cc = sm.cc
S_YEARS = 5
cc.configure(dA=cc.BASE["d"]["K"], adj=True, scurve="ramp", T_end=S_YEARS)
J, E_OLG = sm.J_OLG, sm.E_OLG
G, r0, d, g, a_ = sm.G, sm.r0, sm.d, sm.g, sm.a
E = rv.E
HERE, FIG = cc.HERE, cc.FIG


# ----------------------------------------------------------------------------
# Cohort consumption with habits, exact for any return / wage path
# ----------------------------------------------------------------------------
def cohort_c(Rc, W, s, j0, a0, hab0, beta, h):
    """Rc[t] = prod_{u<t} R_u (gross return u -> u+1); W[t] aggregate wage bill.
    Cohorts start at time s (array) at age j0 with assets a0 (start of period, after returns)
    and habit stock hab0.  Returns c (M, J), validity mask and absolute time index."""
    k = np.arange(J)
    age = j0[:, None] + k[None, :]
    valid = age < J
    tt = np.minimum(s[:, None] + k[None, :], len(Rc) - 1)
    Rs = Rc[s][:, None]
    D = np.where(valid, Rs / Rc[tt], 0.0)
    earn = np.where(valid, E_OLG[np.minimum(age, J - 1)] / J * W[tt], 0.0)
    Hh = (earn * D).sum(1)
    ell = np.where(valid, Rs / (beta ** k[None, :] * Rc[tt]), 0.0)
    Y = np.zeros_like(ell)
    nxt = np.zeros(len(s))
    for kk in range(J - 1, -1, -1):
        nxt = ell[:, kk] + beta * h * nxt
        Y[:, kk] = nxt
    inv = np.where(valid, 1.0 / np.where(valid, Y, 1.0), 0.0)
    cu = np.zeros_like(ell)
    cz = np.zeros_like(ell)
    pu, pz = np.zeros(len(s)), hab0.astype(float).copy()
    for kk in range(J):
        pu = (inv[:, kk] + h * pu) * valid[:, kk]
        pz = h * pz * valid[:, kk]
        cu[:, kk], cz[:, kk] = pu, pz
    S1, Sz = (D * cu).sum(1), (D * cz).sum(1)
    u = (a0 + Hh - Sz) / S1
    c = u[:, None] * cu + cz
    return c, valid, tt, dict(D=D, earn=earn, u=u)


def ss_profile(r, beta, h):
    """Consumption of a newborn on a balanced growth path (birth wage 1), and the C / wage-bill ratio."""
    n = 2 * J + 2
    Rc = (1 + r) ** np.arange(n)
    W = G ** np.arange(n, dtype=float)
    c, valid, tt, _ = cohort_c(Rc, W, np.array([0]), np.array([0]), np.array([0.0]), np.array([0.0]), beta, h)
    c = c[0]
    return c, np.sum(G ** (-np.arange(J, dtype=float)) * c)


def cw(r, beta, h):
    return ss_profile(r, beta, h)[1]


def ss_assets(r, beta, h):
    c, _ = ss_profile(r, beta, h)
    k = np.arange(J)
    a = np.zeros(J + 1)
    for j in range(J):
        a[j + 1] = (1 + r) * (a[j] + E_OLG[j] / J * G ** j - c[j])
    return c, a[:J] / G ** k, a[J] / G ** J


class OLGH:
    """Long-run closed form of the annual OLG (with or without habits), as sm.Saving('olg')."""

    model = "olg"

    def __init__(self, e, h):
        self.e, self.h = e, h
        c0 = sm.core(e, 1.0, sm.LEV0)
        V0 = c0["cap"] / sm.U0
        target = 1 - (g + d) * V0
        self.beta = sm.bisect(lambda b: c0["LS"] * cw(r0, b, h) - target, 0.5, 0.9999)

    def rate(self, lev):
        e = self.e

        def f(r):
            c = sm.core(e, (r + d) / sm.U0, lev)
            if c is None:
                return -1.0 if (r + d) > sm.U0 else 1.0
            return (1 - (g + d) * c["cap"] / (r + d)) - c["LS"] * cw(r, self.beta, self.h)
        return sm.bisect(f, 0.035, 0.4)

    def mult(self, r):
        return (r + d) / sm.U0

    def longrun(self, lev):
        r = self.rate(lev)
        return dict(r=r, **sm.core(self.e, self.mult(r), lev))


# ----------------------------------------------------------------------------
class OLGModel(cc.Model):
    def __init__(self, case, cap, size, olg, cK, cA, T):
        S = {"jump": np.ones(T + 1), "scurve": cc.scurve(T), "mixed": 0.5 + 0.5 * cc.scurve(T)}[case]
        path = {"GK": np.ones(T + 1), "PK": np.ones(T + 1), "GA": np.full(T + 1, cc.BASE["GammaA0"]),
                "PA": np.ones(T + 1)}
        path["P" + cap] = np.exp(-size * S)
        x = {j: np.full(T + 1, cc.bgp(E)["x"][j]) for j in cc.CAPS}
        super().__init__(E, path, x, True, {})
        self.c = {"K": cK, "A": cA}
        self.olg, self.size = olg, size
        lev_end = dict(GK=path["GK"][-1], PK=path["PK"][-1], GA=path["GA"][-1], PA=path["PA"][-1])
        self.lrT = olg.longrun(lev_end)
        self.rT = self.lrT["r"]
        self.W0 = self.ss0["WZ"]
        c_ss, alpha, aend = ss_assets(r0, olg.beta, olg.h)
        self.alpha, self.c_ss, self.aend = alpha, c_ss, aend
        self.V_pre = self.K0["K"] + self.K0["A"]
        self.Gpay_ss = (1 + r0) * self.V_pre
        # cohorts: newborns s = 0..T-1, plus cohorts alive at 0 with ages 1..J-1
        j0o = np.arange(1, J)
        self.s = np.concatenate([np.arange(T), np.zeros(J - 1, dtype=int)])
        self.j0 = np.concatenate([np.zeros(T, dtype=int), j0o])
        self.a0_old = np.concatenate([np.zeros(T), alpha[j0o] * self.W0])
        self.hab0 = np.concatenate([np.zeros(T), G ** (-j0o.astype(float)) * c_ss[j0o - 1] * self.W0])
        self.isold = np.concatenate([np.zeros(T, dtype=bool), np.ones(J - 1, dtype=bool)])

    def parts(self, ii):
        S = self.sim(ii)
        T, dd = self.T, self.p["d"]
        ret = {j: self.MPQ(S, j) for j in cc.CAPS}
        R = {j: ret[j][1:] / S["Q"][j][:-1] for j in cc.CAPS}
        return S, R

    def MPQ(self, S, j):
        return S["MP"][j] + S["extra"][j] + (1 - self.p["d"][j]) * S["Q"][j]

    def olg_consumption(self, S, R):
        T = self.T
        Wx = np.concatenate([S["W"], S["W"][-1] * G ** np.arange(1, J + 1)])
        gross = np.concatenate([R["K"], np.full(J + 1, 1 + self.rT)])
        Rc = np.concatenate([[1.0], np.cumprod(gross)])
        Gpay0 = sum(S["K"][j][0] * self.MPQ(S, j)[0] for j in cc.CAPS)
        scale = Gpay0 / self.Gpay_ss
        a0 = np.where(self.isold, self.a0_old * scale, 0.0)
        c, valid, tt, aux = cohort_c(Rc, Wx, self.s, self.j0, a0, self.hab0, self.olg.beta, self.olg.h)
        m = valid & (tt < T)
        C = np.bincount(tt[m], weights=c[m], minlength=T)[:T]
        return C

    def resid(self, z):
        T = self.T
        ii = np.hstack([z.reshape(2, T), self.term[:, None]])
        with np.errstate(all="ignore"):
            S, R = self.parts(ii)
            if np.any(~np.isfinite(S["C"])) or np.any(S["C"] <= 0) or np.any(ii <= -0.9):
                return np.full(2 * T, 1e3)
            e1 = np.log(R["K"] / R["A"])
            Co = self.olg_consumption(S, R)
            e2 = np.log(S["C"][:T] / Co)
        res = np.concatenate([e1, e2])
        return np.where(np.isfinite(res), res, 1e3)

    def solve(self, z0, tol=1e-9, maxit=80):
        z = z0.copy()
        R = self.resid(z)
        n, mu = z.size, 1e-3
        for it in range(maxit):
            if np.max(np.abs(R)) < tol:
                break
            Jm = np.empty((n, n))
            for i in range(n):
                dz = np.zeros(n); dz[i] = 1e-7
                Jm[:, i] = (self.resid(z + dz) - R) / 1e-7
            JTJ, gr = Jm.T @ Jm, Jm.T @ R
            f0 = R @ R
            while mu < 1e12:
                step = np.linalg.solve(JTJ + mu * np.diag(np.diag(JTJ) + 1e-12), -gr)
                Rn = self.resid(z + step)
                if Rn @ Rn < f0:
                    z, R, mu = z + step, Rn, max(mu / 5, 1e-12)
                    break
                mu *= 10
        self.err, self.iters, self.zsol = float(np.max(np.abs(R))), it, z
        T = self.T
        ii = np.hstack([z.reshape(2, T), self.term[:, None]])
        S, R_ = self.parts(ii)
        Zt = self.Z
        spend_gdp = (S["F"] - S["C"]) / S["F"]
        gdp0 = self.ss0["YZ"]
        return dict(r=R_["K"] - 1, lnYg=np.log(S["F"] / Zt / gdp0), lnW=np.log(S["W"] / Zt / self.ss0["WZ"]),
                    labsh=S["W"] / S["F"], invg=spend_gdp,
                    invg0=(gdp0 - (gdp0 - sum(self.K0[j] * self.x[j][0] for j in cc.CAPS))) / gdp0,
                    lnC=np.log(S["C"] / Zt / (gdp0 - sum(self.K0[j] * self.x[j][0] for j in cc.CAPS))),
                    err=self.err, rT=self.rT, size=self.size, z=z, K=S["K"]["K"], A=S["K"]["A"])


def olg_size(olg, cap):
    return sm.size_for_target(olg, cap)


# ----------------------------------------------------------------------------
BLOCKS = [("Log", 0.0), ("Log + habits", 0.7)]
TIMINGS = [("scurve", f"S-curve, complete by year {S_YEARS}"), ("mixed", "Half jump, half S-curve"),
           ("jump", "Surprise jump")]
CAPS = [("K", "Regular capital", "#1f5f9e"), ("A", "AI capital", "#c0512a")]
T_RUN = 250


def stats(res):
    r = 100 * (res["r"] - r0)
    dI = 100 * (res["invg"][:21] - res["invg0"])
    return dict(avg1_10=float(np.mean(r[:10])), avg11_20=float(np.mean(r[10:20])), avg21_50=float(np.mean(r[20:50])),
                r1=float(100 * res["r"][0]), trough=float(100 * res["r"][:10].min()),
                peak=float(100 * res["r"][:10].max()), r20=float(100 * res["r"][20]),
                r50=float(100 * res["r"][50]), rLR=float(100 * res["r"][150]),
                rT=float(100 * res.get("rT", r0)), gdp10=float(100 * res["lnYg"][10]),
                closed10=float(100 * res["lnYg"][10] / (100 * res["lnYg"][150])),
                dI_peak=float(dI.max()), dI_t=int(dI.argmax()), dI10=float(dI[10]), dI20=float(dI[20]),
                W10=float(100 * res["lnW"][10]), LS10=float(100 * res["labsh"][10]),
                LSLR=float(100 * res["labsh"][150]), gdpLR=float(100 * res["lnYg"][150]))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = json.load(open(os.path.join(HERE, "results_emb.json")))
    cK, cA = j["cK"], j["cA"]
    print(f"c_K = {cK:.3f}, c_A = {cA:.3f}")
    ram, olg_res, sz, st, checks = {}, {}, {}, {}, {}
    models = {h: OLGH(E, h) for _, h in BLOCKS}
    for name, h in BLOCKS:
        print(f"{name}: beta = {models[h].beta:.5f}")
    for case, _ in TIMINGS:
        for cap, _, _ in CAPS:
            for name, h in BLOCKS:
                hh = rv.HH(1, h, 0, name)
                rr = rv.run(case, cap, "emb", hh, cA, cK=cK, T=T_RUN)
                ram[(case, cap, name)] = rr
                size = olg_size(models[h], cap)
                m = OLGModel(case, cap, size, models[h], cK, cA, T_RUN)
                out = m.solve(rr["z"])
                olg_res[(case, cap, name)] = out
                sz[(case, cap, name)] = size
                s_o, s_r = stats(out), stats(rr)
                st[("OLG", case, cap, name)], st[("Ramsey", case, cap, name)] = s_o, s_r
                checks[f"{name} {case} {cap}: OLG residual"] = out["err"]
                checks[f"{name} {case} {cap}: Ramsey residual"] = rr["err"]
                checks[f"{name} {case} {cap}: OLG long-run ln Y = 10%"] = abs(out["lnYg"][150] - 0.10)
                print(f"  {case:6s} {cap} {name:13s} OLG avg={s_o['avg1_10']:+.2f} r1={s_o['r1']:.2f} "
                      f"r20={s_o['r20']:.2f} r50={s_o['r50']:.2f} rLR={s_o['rLR']:.2f} | Ramsey avg={s_r['avg1_10']:+.2f} "
                      f"r1={s_r['r1']:.2f} r20={s_r['r20']:.2f} | err={out['err']:.0e}", flush=True)
    # BGP consistency (initial steady state is a steady state of the OLG equilibrium)
    for name, h in BLOCKS:
        m = OLGModel("jump", "K", 0.0, models[h], cK, cA, 80)
        z = np.concatenate([np.full(80, m.term[k]) for k in range(2)])
        checks[f"{name}: initial BGP solves the OLG equilibrium"] = float(np.max(np.abs(m.resid(z))))

    # ---- figure: real rate to year 50
    H = 51
    t = np.arange(H)
    fig, axes = plt.subplots(len(BLOCKS), len(TIMINGS), figsize=(13, 7), sharey="row")
    for row, (name, h) in enumerate(BLOCKS):
        for col, (case, ttl) in enumerate(TIMINGS):
            ax = axes[row, col]
            for cap, lab, color in CAPS:
                ax.plot(t, 100 * olg_res[(case, cap, name)]["r"][:H], color=color, lw=2, label=f"{lab}, OLG")
                ax.plot(t, 100 * ram[(case, cap, name)]["r"][:H], color=color, lw=1.4, ls="--", label=f"{lab}, Ramsey")
            ax.axhline(100 * r0, color="0.6", lw=1, ls=":")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(ttl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{name}\nreal rate (pct.)", fontsize=9)
            if row == len(BLOCKS) - 1:
                ax.set_xlabel("years")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Real rate, annual OLG (solid) vs Ramsey (dashed), embodied progress, same adjustment costs and habits",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_olg_r.{ext}"), dpi=140)
    plt.close(fig)

    # ---- figure: AI minus regular gap in r (OLG vs Ramsey), 60 years
    H2 = 61
    t2 = np.arange(H2)
    fig, axes = plt.subplots(len(BLOCKS), len(TIMINGS), figsize=(13, 6), sharey=True)
    for row, (name, h) in enumerate(BLOCKS):
        for col, (case, ttl) in enumerate(TIMINGS):
            ax = axes[row, col]
            for mdl, res, sty, lw in (("OLG", olg_res, "-", 2), ("Ramsey", ram, "--", 1.4)):
                gap = 100 * (res[(case, "A", name)]["r"][:H2] - res[(case, "K", name)]["r"][:H2])
                ax.plot(t2, gap, color="0.15", ls=sty, lw=lw, label=mdl)
            ax.axhline(0, color="0.6", lw=1, ls=":")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(ttl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{name}\nr(AI) - r(regular), pp", fontsize=9)
            if row == len(BLOCKS) - 1:
                ax.set_xlabel("years")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Real-rate gap between AI and regular-capital progress (same 10% long-run output gain)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_olg_gap.{ext}"), dpi=140)
    plt.close(fig)

    # ---- tables
    L = [r"\begin{tabular}{lllccccccc}\toprule",
         r" & & & avg. $r$ & avg. $r$ & avg. $r$ & $r_1$ & $r$, & $r$, & long-run $r$\\",
         r" & & & yrs 1--10 & yrs 11--20 & yrs 21--50 & & yr 20 & yr 50 & (closed form)\\",
         r" & & & \multicolumn{3}{c}{(pp over 7.5\%)} & \multicolumn{4}{c}{(percent)}\\\midrule"]
    for case, _ in TIMINGS:
        for name, h in BLOCKS:
            for cap, lab, _c in CAPS:
                for mdl in ("Ramsey", "OLG"):
                    s = st[(mdl, case, cap, name)]
                    tl = {"scurve": "S-curve", "mixed": "Half/half", "jump": "Jump"}[case] if (name == "Log" and cap == "K" and mdl == "Ramsey") else ""
                    bl = (name if (cap == "K" and mdl == "Ramsey") else "").replace("+", "+")
                    cp = lab.split()[0] if mdl == "Ramsey" else ""
                    L.append(f"{tl} & {bl} & {cp} {mdl} & {s['avg1_10']:+.2f} & {s['avg11_20']:+.2f} & {s['avg21_50']:+.2f} & "
                             f"{s['r1']:.2f} & {s['r20']:.2f} & {s['r50']:.2f} & {s['rT'] if mdl == 'OLG' else 100*r0:.2f}\\\\")
            L.append(r"\addlinespace" if name == "Log" else r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_olg.tex"), "w").write("\n".join(L) + "\n")

    G_ = [r"\begin{tabular}{llcccccc}\toprule",
          r" & & \multicolumn{3}{c}{AI $-$ regular, OLG (pp)} & \multicolumn{3}{c}{AI $-$ regular, Ramsey (pp)}\\",
          r" & & yrs 1--10 & yrs 11--20 & yrs 21--50 & yrs 1--10 & yrs 11--20 & yrs 21--50\\\midrule"]
    for case, _ in TIMINGS:
        for i, (name, h) in enumerate(BLOCKS):
            def gp(mdl, k):
                return st[(mdl, case, "A", name)][k] - st[(mdl, case, "K", name)][k]
            tl = {"scurve": "S-curve", "mixed": "Half/half", "jump": "Jump"}[case] if i == 0 else ""
            G_.append(f"{tl} & {name} & " + " & ".join(f"{gp('OLG', k):+.2f}" for k in ("avg1_10", "avg11_20", "avg21_50")) +
                      " & " + " & ".join(f"{gp('Ramsey', k):+.2f}" for k in ("avg1_10", "avg11_20", "avg21_50")) + r"\\")
        G_.append(r"\midrule")
    G_[-1] = r"\bottomrule\end{tabular}"
    open(os.path.join(HERE, "table_olg_gap.tex"), "w").write("\n".join(G_) + "\n")

    O = [r"\begin{tabular}{llcccccc}\toprule",
         r" & & GDP gap closed, yr 10 (\%) & $\Delta$ inv./GDP yr 10 (pp) & $\Delta$ inv./GDP yr 20 (pp) & $\ln W$, yr 10 (\%) & labor share yr 10 (\%) & labor share, long run (\%)\\\midrule"]
    for case, _ in TIMINGS:
        for name, h in BLOCKS:
            for cap, lab, _c in CAPS:
                for mdl in ("Ramsey", "OLG"):
                    s = st[(mdl, case, cap, name)]
                    tl = {"scurve": "S-curve", "mixed": "Half/half", "jump": "Jump"}[case] if (name == "Log" and cap == "K" and mdl == "Ramsey") else ""
                    O.append(f"{tl} & {name if (cap == 'K' and mdl == 'Ramsey') else ''} {lab.split()[0]} {mdl} & {s['closed10']:.0f} & "
                             f"{s['dI10']:+.2f} & {s['dI20']:+.2f} & {s['W10']:.1f} & {s['LS10']:.1f} & {s['LSLR']:.1f}\\\\")
            O.append(r"\midrule")
    O[-1] = r"\bottomrule\end{tabular}"
    open(os.path.join(HERE, "table_olg_real.tex"), "w").write("\n".join(O) + "\n")

    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    worst = {}
    for k, v in checks.items():
        key = (k.split(": ", 1)[1] if ": " in k else k).replace("%", r"\%")
        worst[key] = max(worst.get(key, 0.0), v)
    for k, v in worst.items():
        C.append(f"{k} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    open(os.path.join(HERE, "table_olg_checks.tex"), "w").write("\n".join(C) + "\n")
    json.dump(dict(beta={n: models[h].beta for n, h in BLOCKS}, sizes={" | ".join(k): v for k, v in sz.items()},
                   checks=checks, stats={" | ".join(k): v for k, v in st.items()},
                   r_paths={" | ".join(("OLG",) + k): [float(x) for x in 100 * v["r"][:101]] for k, v in olg_res.items()}),
              open(os.path.join(HERE, "results_olg.json"), "w"), indent=1, default=float)
    print("max check:", max(checks.values()))
    for k, v in checks.items():
        if v > 1e-6:
            print("  LARGE", k, v)


if __name__ == "__main__":
    main()
