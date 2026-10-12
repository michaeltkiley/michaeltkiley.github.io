"""
How the saving model shapes the response of r, W and the labor share to AI progress.

Production (as in capital_cases.py):
    X = (Gamma_K K)^a (Z N)^(1-a),  Y = CES(X, Gamma_A A; e),  e = 1.5 baseline (AI substitutes)
    K_j' = (1-d) K_j + I_j/P_j,  d_A = d_K = d = 0.066,  no adjustment costs, fixed hours.
With no adjustment costs the two capital stocks earn equal returns, so investors care
only about total wealth.

Saving models (annual):
  Ramsey          log utility, 1 + r = G/beta in the long run (perfectly elastic capital supply)
  Solow           fixed gross saving rate s:  C = (1-s) Y
  Perpetual youth Blanchard-Yaari, death probability p, annuities, log utility:
                  C = (1 - beta(1-p)) (A + H),  H_t = W_t + (1-p)/(1+r_{t+1}) H_{t+1}
                  (p = 0 is Ramsey)
  Diamond OLG     two periods of L = 30 years, log utility: young save beta_L/(1+beta_L) of
                  wages; long run only (closed-form fixed point)

Long-run results come from closed forms (one-dimensional fixed points in r):
  every model:  capital share = 1 - (1-a)(1-s_A),  s_A = (1-w) lambda_A^(1-e),
                lambda_j = P_j (r+d)/Gamma_j,  V/Y = capital share/(r+d)
  Ramsey:  r = r0 (unchanged by level shifts)
  Solow:   (g+d) * capital share/(r+d) = s
  Diamond: r_L + d_L = G^L capital share / (s_y * labor share)
  PY:      C/Y = (1-beta(1-p)) [ (1+r) V/Y + labor share/(1 - G(1-p)/(1+r)) ]
All shocks are scaled within each model to raise long-run output by 10%.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import capital_cases as cc

HERE = cc.HERE
FIG = cc.FIG
cc.configure(dA=cc.BASE["d"]["K"], adj=False, scurve="ramp", T_end=10)
P0 = cc.BASE
a, om, d, g, r0 = P0["a"], P0["omega"], P0["d"]["K"], P0["g_Z"], P0["r0"]
G = 1 + g
U0 = r0 + d                                 # initial user cost (both capitals)
TARGET = 0.10
L = 30                                      # Diamond period length (years)
P_PY = 0.025                                # perpetual-youth death probability (40-yr horizon)
J_OLG, JW_OLG = 60, 45                      # annual OLG: ages 20-79, work 20-64
E_OLG = np.where(np.arange(J_OLG) < JW_OLG, J_OLG / JW_OLG, 0.0)   # flat earnings, mean one


def olg_kappa(beta, n):
    """Marginal propensity to consume out of wealth with n periods left (log utility)."""
    return (1 - beta) / (1 - beta ** n)


def olg_cw(r, beta):
    """Balanced-growth aggregate consumption / aggregate wage bill in the annual OLG."""
    k = np.arange(J_OLG)
    Ht = np.sum(E_OLG / J_OLG * (G / (1 + r)) ** k)
    return olg_kappa(beta, J_OLG) * Ht * np.sum((beta * (1 + r) / G) ** k)


def olg_profile(r, beta):
    """Steady-state assets by age (start of period, after returns), relative to the wage bill."""
    k = np.arange(J_OLG)
    Wk = G ** k
    h = np.sum(E_OLG / J_OLG * Wk / (1 + r) ** k)
    c = olg_kappa(beta, J_OLG) * h * (beta * (1 + r)) ** k
    a = np.zeros(J_OLG + 1)
    for j in range(J_OLG):
        a[j + 1] = (1 + r) * (a[j] + E_OLG[j] * Wk[j] / J_OLG - c[j])
    return a[:J_OLG] / Wk, a[J_OLG] / G ** J_OLG


# ----------------------------------------------------------------------------
# Closed-form building blocks
# ----------------------------------------------------------------------------
def core(e, mult, lev):
    """Shares and levels given a user-cost multiplier mult = u/u0 and technology levels.

    lambda_j = (P_j/Gamma_j) * u0 * mult, with Gamma_A normalized so lambda_A0 = 1.
    """
    lamK = lev["PK"] / lev["GK"] * U0 * mult
    lamA = lev["PA"] / (lev["GA"] / P0["GammaA0"]) * mult       # = 1 initially
    if e == 1.0:
        sA = 1 - om
        PX = lamA ** (-(1 - om) / om)
    else:
        sA = (1 - om) * lamA ** (1 - e)
        if sA >= 1:
            return None
        PX = ((1 - sA) / om) ** (1 / (1 - e))
    XZ = (a * PX / lamK) ** (a / (1 - a))
    LS = (1 - a) * (1 - sA)
    return dict(sA=sA, LS=LS, cap=1 - LS, WZ=(1 - a) * PX * XZ, YZ=PX * XZ / (1 - sA))


def bisect(f, lo, hi, n=200):
    flo = f(lo)
    if (flo > 0) == (f(hi) > 0):
        raise ValueError("bracket does not contain a root")
    for _ in range(n):
        m = 0.5 * (lo + hi)
        fm = f(m)
        if (fm > 0) == (flo > 0):
            lo, flo = m, fm
        else:
            hi = m
    return 0.5 * (lo + hi)


class Saving:
    """Long-run (balanced growth) closed forms for each saving model."""

    def __init__(self, model, e, p=P_PY):
        self.model, self.e, self.p = model, e, p
        c0 = core(e, 1.0, LEV0)
        V0 = c0["cap"] / U0
        if model == "solow":
            self.s = (g + d) * V0
        elif model == "py":
            CY = 1 - (g + d) * V0
            AY = (1 + r0) * V0
            HY = c0["LS"] / (1 - G * (1 - p) / (1 + r0))
            self.mu = CY / (AY + HY)
            self.beta = (1 - self.mu) / (1 - p)
        elif model == "ramsey":
            self.beta = G / (1 + r0)
            self.mu = 1 - self.beta
        elif model == "olg":
            target = 1 - (g + d) * V0                     # C/Y on the initial BGP
            self.beta = bisect(lambda b: c0["LS"] * olg_cw(r0, b) - target, 0.5, 0.999)
        elif model == "diamond":
            self.GL = G ** L
            self.dL = 1 - (1 - d) ** L
            self.rL0 = (1 + r0) ** L - 1
            self.sy = self.GL * c0["cap"] / (c0["LS"] * (self.rL0 + self.dL))

    def rate(self, lev):
        """Long-run annual r for technology levels lev."""
        e = self.e
        if self.model == "ramsey":
            return r0
        if self.model == "solow":
            def f(r):                       # decreasing in r
                c = core(e, (r + d) / U0, lev)
                if c is None:                   # AI share >= 1: outside the valid region
                    return -1.0 if (r + d) > U0 else 1.0
                return (g + d) * c["cap"] / (r + d) - self.s
            return bisect(f, 0.0, 2.0)
        if self.model == "py":
            def f(r):                       # increasing in r
                c = core(e, (r + d) / U0, lev)
                if c is None:
                    return 1.0 if (r + d) > U0 else -1.0
                V = c["cap"] / (r + d)
                CY = 1 - (g + d) * V
                return CY - self.mu * ((1 + r) * V + c["LS"] / (1 - G * (1 - self.p) / (1 + r)))
            return bisect(f, max(0.0, G * (1 - self.p) - 1 + 1e-6), 2.0)
        if self.model == "olg":
            # Two steady states exist (one near r = 1.5%); the calibrated one is at r0, where the
            # excess of resource C/Y over household C/Y is decreasing in r. Bracket around it.
            def f(r):
                c = core(e, (r + d) / U0, lev)
                if c is None:
                    return -1.0 if (r + d) > U0 else 1.0
                return (1 - (g + d) * c["cap"] / (r + d)) - c["LS"] * olg_cw(r, self.beta)
            return bisect(f, 0.035, 0.4)
        # diamond: solve for the period rate, report the annualized rate
        uL0 = self.rL0 + self.dL

        def f(rL):                          # increasing in r
            c = core(e, (rL + self.dL) / uL0, lev)
            if c is None:
                return 1.0 if (rL + self.dL) > uL0 else -1.0
            return rL + self.dL - self.GL * c["cap"] / (self.sy * c["LS"])
        rL = bisect(f, 0.0, 1e4)
        return (1 + rL) ** (1 / L) - 1

    def mult(self, r):
        if self.model == "diamond":
            rL = (1 + r) ** L - 1
            return (rL + self.dL) / (self.rL0 + self.dL)
        return (r + d) / U0

    def longrun(self, lev):
        r = self.rate(lev)
        c = core(self.e, self.mult(r), lev)
        return dict(r=r, **c)


LEV0 = dict(GK=1.0, PK=1.0, GA=P0["GammaA0"], PA=1.0)


def size_for_target(sv, cap):
    """Log improvement in Gamma_j/P_j giving a 10% long-run output gain in this model."""
    Y0 = sv.longrun(LEV0)["YZ"]
    key = "GK" if cap == "K" else "GA"

    def f(x):
        lev = dict(LEV0); lev[key] = LEV0[key] * np.exp(x)
        out = sv.longrun(lev)
        return np.log(out["YZ"] / Y0) - TARGET

    hi = 0.5
    while f(hi) < 0:
        hi *= 2
        if hi > 60:
            return None
    return bisect(f, 0.0, hi, 100)


# ----------------------------------------------------------------------------
# Annual transitions: Ramsey, perpetual youth, Solow
# ----------------------------------------------------------------------------
class Transition:
    def __init__(self, sv, case, cap, typ, T=250):
        self.sv, self.e, self.T = sv, sv.e, T
        self.size = size_for_target(sv, cap)
        t = np.arange(T + 1)
        self.Z = G ** t
        S = {"jump": np.ones(T + 1), "scurve": cc.scurve(T), "mixed": 0.5 + 0.5 * cc.scurve(T)}[case]
        self.Gam = {"K": np.ones(T + 1), "A": np.full(T + 1, P0["GammaA0"])}
        self.P = {"K": np.ones(T + 1), "A": np.ones(T + 1)}
        if typ == "dis":
            self.Gam[cap] = self.Gam[cap] * np.exp(self.size * S)
        else:
            self.P[cap] = self.P[cap] * np.exp(-self.size * S)
        lev_end = dict(GK=self.Gam["K"][-1], PK=self.P["K"][-1], GA=self.Gam["A"][-1], PA=self.P["A"][-1])
        self.lr0, self.lrT = sv.longrun(LEV0), sv.longrun(lev_end)
        # initial states (Z_0 = 1)
        c0 = self.lr0
        self.K0 = {"K": a * (1 - c0["sA"]) / U0 * c0["YZ"], "A": c0["sA"] / U0 * c0["YZ"]}
        rT = self.lrT["r"]
        uT = rT + d
        self.KT_ratio = {"K": a * (1 - self.lrT["sA"]) / (lev_end["PK"] * uT),
                         "A": self.lrT["sA"] / (lev_end["PA"] * uT)}

    def guess(self):
        T = self.T
        t = np.arange(1, T + 1)
        w = np.clip(t / 20.0, 0, 1)
        out = []
        for j in ("K", "A"):
            start = np.log(self.K0[j]) + t * np.log(G)
            end = np.log(self.KT_ratio[j] * self.lrT["YZ"]) + t * np.log(G)
            out.append((1 - w) * start + w * end)
        return np.concatenate(out)

    def paths(self, z):
        T = self.T
        K = {j: np.empty(T + 2) for j in ("K", "A")}
        for k, j in enumerate(("K", "A")):
            K[j][0] = self.K0[j]
            K[j][1:T + 1] = np.exp(z[k * T:(k + 1) * T])
            if self.sv.model == "olg":                     # pin to the new BGP beyond T
                K[j][T + 1] = self.KT_ratio[j] * self.lrT["YZ"] * G ** (T + 1)
            else:
                K[j][T + 1] = G * K[j][T]
        a_ = a
        X = (self.Gam["K"] * K["K"][:-1]) ** a_ * self.Z ** (1 - a_)
        F, FX, FM = cc.prod(X, self.Gam["A"] * K["A"][:-1], self.e)
        MP = {"K": FX * a_ * X / K["K"][:-1], "A": FM * self.Gam["A"]}
        W = FX * (1 - a_) * X
        inv = sum(self.P[j] * (K[j][1:] - (1 - d) * K[j][:-1]) for j in ("K", "A"))
        C = F - inv
        A = sum((MP[j] + (1 - d) * self.P[j]) * K[j][:-1] for j in ("K", "A"))
        R = {j: (MP[j][1:] + (1 - d) * self.P[j][1:]) / self.P[j][:-1] for j in ("K", "A")}
        return dict(K=K, Y=F, W=W, C=C, A=A, R=R)

    def resid(self, z):
        T, sv = self.T, self.sv
        with np.errstate(all="ignore"):
            S = self.paths(z)
            if np.any(S["C"] <= 0) or np.any(~np.isfinite(S["C"])):
                return np.full(2 * T, 1e3)
            e1 = np.log(S["R"]["K"][:T] / S["R"]["A"][:T])
            if sv.model == "solow":
                e2 = np.log(S["C"][:T] / ((1 - sv.s) * S["Y"][:T]))
            elif sv.model == "olg":
                e2 = np.log(S["C"][:T] / self.olg_consumption(S))
            else:
                p = sv.p if sv.model == "py" else 0.0
                rT = self.lrT["r"]
                H = np.empty(T + 1)
                H[T] = S["W"][T] / (1 - G * (1 - p) / (1 + rT))
                for t in range(T - 1, -1, -1):
                    H[t] = S["W"][t] + (1 - p) / S["R"]["K"][t] * H[t + 1]
                e2 = np.log(S["C"][:T] / (sv.mu * (S["A"][:T] + H[:T])))
        res = np.concatenate([e1, e2])
        return np.where(np.isfinite(res), res, 1e3)

    def olg_consumption(self, S):
        """Aggregate household consumption C_t, t < T, given wage and return paths."""
        T, J, b = self.T, J_OLG, self.sv.beta
        if not hasattr(self, "alpha"):
            self.alpha, _ = olg_profile(r0, b)
            c0 = self.lr0
            self.W0ss = c0["LS"] * c0["YZ"]                 # pre-shock wage bill at t = 0
            self.A0ss = np.sum(self.alpha) * self.W0ss
        rT = self.lrT["r"]
        Wx = np.concatenate([S["W"], S["W"][-1] * G ** np.arange(1, J + 1)])
        gross = np.concatenate([S["R"]["K"], np.full(J + 1, 1 + rT)])
        Rc = np.concatenate([[1.0], np.cumprod(gross)])[:T + J + 1]
        k = np.arange(J)
        # newborn cohorts tau = 0..T-1
        h = np.zeros(T)
        for kk in range(J):
            h += E_OLG[kk] / J * Wx[kk:kk + T] * Rc[:T] / Rc[kk:kk + T]
        cb = olg_kappa(b, J) * h
        # cohorts alive at t = 0 with ages j0 = 1..J-1, assets revalued by the shock
        scale = S["A"][0] / self.A0ss
        c0 = np.zeros(J)
        for j0 in range(1, J):
            n = J - j0
            h0 = np.sum(E_OLG[j0:j0 + n] / J * Wx[:n] / Rc[:n])
            c0[j0] = olg_kappa(b, n) * (self.alpha[j0] * self.W0ss * scale + h0)
        C = np.zeros(T)
        t = np.arange(T)
        for kk in range(J):
            ok = t - kk >= 0
            tt = t[ok]
            C[ok] += cb[tt - kk] * b ** kk * Rc[tt] / Rc[tt - kk]
        for j0 in range(1, J):
            tt = t[t <= J - 1 - j0]
            C[tt] += c0[j0] * b ** tt * Rc[tt]
        return C

    def solve(self, tol=1e-10, maxit=60):
        if self.sv.model == "olg":
            # Newton started from the Ramsey solution for the same technology path
            # (OLG capital supply is nearly as elastic, so this lies in Newton's basin).
            helper = Transition(Saving("ramsey", self.e), "jump", "K", "dis", T=self.T)
            helper.Gam, helper.P = self.Gam, self.P
            lev_end = dict(GK=self.Gam["K"][-1], PK=self.P["K"][-1], GA=self.Gam["A"][-1],
                           PA=self.P["A"][-1])
            helper.lrT = helper.sv.longrun(lev_end)
            helper.KT_ratio = self.KT_ratio
            helper.solve(tol=1e-10)
            z = helper.zsol.copy()
            # Levenberg-Marquardt: plain Newton overshoots because life-cycle saving is very
            # sensitive to returns (beta(1+r)/G is close to one).
            Rv = self.resid(z)
            n, mu = z.size, 1e-3
            for it in range(maxit):
                if np.max(np.abs(Rv)) < tol:
                    break
                J = np.empty((n, n))
                for i_ in range(n):
                    dz = np.zeros(n); dz[i_] = 1e-7
                    J[:, i_] = (self.resid(z + dz) - Rv) / 1e-7
                JTJ, gr = J.T @ J, J.T @ Rv
                f0 = Rv @ Rv
                while mu < 1e12:
                    step = np.linalg.solve(JTJ + mu * np.diag(np.diag(JTJ) + 1e-12), -gr)
                    Rn = self.resid(z + step)
                    if Rn @ Rn < f0:
                        z, Rv, mu = z + step, Rn, max(mu / 5, 1e-12)
                        break
                    mu *= 10
            self.err, self.iters = float(np.max(np.abs(Rv))), it
            self.zsol = z
            S = self.paths(z)
            c0 = self.lr0
            # the last ~40 years carry a horizon-truncation artifact (cohorts alive at T);
            # long-run values are read at T - 100
            return dict(r=S["R"]["K"] - 1, lnY=np.log(S["Y"] / self.Z / c0["YZ"]),
                        lnW=np.log(S["W"] / self.Z / c0["WZ"]), labsh=S["W"] / S["Y"],
                        err=self.err, size=self.size, rT=self.lrT["r"], lr=self.T - 100)
        z = self.guess()
        Rv = self.resid(z)
        n = z.size
        for it in range(maxit):
            nr = np.max(np.abs(Rv))
            if nr < tol:
                break
            J = np.empty((n, n))
            for i in range(n):
                dz = np.zeros(n); dz[i] = 1e-7
                J[:, i] = (self.resid(z + dz) - Rv) / 1e-7
            step = np.linalg.solve(J, -Rv)
            lam = 1.0
            while lam > 1e-6:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, Rv = zn, Rn
        self.err, self.iters = float(np.max(np.abs(Rv))), it
        self.zsol = z
        S = self.paths(z)
        c0 = self.lr0
        return dict(r=S["R"]["K"] - 1, lnY=np.log(S["Y"] / self.Z / c0["YZ"]),
                    lnW=np.log(S["W"] / self.Z / c0["WZ"]), labsh=S["W"] / S["Y"],
                    err=self.err, size=self.size, rT=self.lrT["r"], lr=-1)


# ----------------------------------------------------------------------------
def main():
    checks, out = {}, {}
    MODELS = [("ramsey", "Ramsey"), ("py", f"Perpetual youth (p={P_PY})"),
              ("olg", "Annual OLG (60 cohorts)"), ("diamond", "Diamond OLG (30-yr periods)"),
              ("solow", "Solow")]
    TMODELS = ("ramsey", "py", "olg", "solow")

    # OLG consistency: steady-state household assets equal the value of capital
    for e_ in (1.5,):
        svo = Saving("olg", e_)
        alpha, aend = olg_profile(r0, svo.beta)
        c0 = core(e_, 1.0, LEV0)
        checks["OLG: household assets = (1+r)V on initial BGP"] = abs(
            alpha.sum() - (1 + r0) * c0["cap"] / U0 / c0["LS"])
        checks["OLG: lifetime budget (terminal assets)"] = abs(aend)

    # ---------- long run: AI vs regular capital, and the AI elasticity
    lr_rows = []
    for e in (0.75, 1.0, 1.5, 2.0):
        for m, mname in MODELS:
            sv = Saving(m, e)
            checks[f"{m} e={e}: initial r reproduces 7.5%"] = abs(sv.rate(LEV0) - r0)
            base = sv.longrun(LEV0)
            for cap in ("K", "A"):
                if cap == "K" and e != 1.5:
                    continue
                x = size_for_target(sv, cap)
                if x is None:
                    lr_rows.append((e, m, cap, None))
                    continue
                key = "GK" if cap == "K" else "GA"
                lev = dict(LEV0); lev[key] = LEV0[key] * np.exp(x)
                new = sv.longrun(lev)
                lr_rows.append((e, m, cap, dict(size=x, dr=100 * (new["r"] - base["r"]),
                                                r=100 * new["r"], dW=100 * np.log(new["WZ"] / base["WZ"]),
                                                LS=100 * new["LS"], sA=100 * new["sA"],
                                                dY=100 * np.log(new["YZ"] / base["YZ"]))))
    for row in lr_rows:
        print(" LR", row[0], row[1], row[2], None if row[3] is None else
              {k: round(v, 3) for k, v in row[3].items()})

    # ---------- transitions (e = 1.5): Ramsey, PY, Solow
    e = 1.5
    for m in TMODELS:
        sv = Saving(m, e)
        for case in ("jump", "mixed", "scurve"):
            for cap in ("K", "A"):
                for typ in ("dis", "emb"):
                    tr = Transition(sv, case, cap, typ, T=(300 if m == "olg" else 250))
                    res = tr.solve()
                    out[(m, case, cap, typ)] = res
                    checks[f"{m} {case} {cap} {typ}: residual"] = res["err"]
                    checks[f"{m} {case} {cap} {typ}: long-run ln Y = 10%"] = abs(res["lnY"][res["lr"]] - TARGET)
                    checks[f"{m} {case} {cap} {typ}: long-run r = closed form"] = abs(
                        res["r"][res["lr"]] - res["rT"])
                    print(f"  {m:6s} {case:6s} {cap} {typ}  size={res['size']:.3f} err={res['err']:.0e} "
                          f"r1={100*res['r'][0]:.2f} max={100*res['r'].max():.2f} "
                          f"min={100*res['r'].min():.2f} rLR={100*res['rT']:.2f} "
                          f"W0={100*res['lnW'][0]:.1f} W10={100*res['lnW'][10]:.1f} "
                          f"WLR={100*res['lnW'][res['lr']]:.1f} LS10={100*res['labsh'][10]:.1f}")
    # Ramsey here must reproduce capital_cases --equal-d-noadj
    ref = json.load(open(os.path.join(HERE, "results_capital_cases_eqd_noadj.json")))["summary"]
    for case in ("jump", "mixed", "scurve"):
        for cap in ("K", "A"):
            for typ in ("dis", "emb"):
                checks[f"Ramsey {case} {cap} {typ}: r1 = Euler-equation solver"] = abs(
                    out[("ramsey", case, cap, typ)]["r"][0] - ref[f"{case} {cap} {typ}"]["r1"])

    # ---------- figure 1: long-run change in r vs elasticity
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    eps = (0.75, 1.0, 1.5, 2.0)
    sty = {"ramsey": ("0.3", "-"), "py": ("#1f5f9e", "-"), "olg": ("#7a4fb0", "-"),
           "solow": ("#c0512a", "--"), "diamond": ("#2a8a4a", "-.")}
    for m, mname in MODELS:
        get = lambda e_, k: next(((r[3] or {}).get(k, np.nan) for r in lr_rows
                                  if r[0] == e_ and r[1] == m and r[2] == "A"))
        dr = [get(e_, "dr") for e_ in eps]
        dw = [get(e_, "dW") for e_ in eps]
        ax[0].plot(eps, dr, marker="o", color=sty[m][0], ls=sty[m][1], label=mname)
        ax[1].plot(eps, dw, marker="o", color=sty[m][0], ls=sty[m][1], label=mname)
    ax[0].set_title("Long-run change in r (pp), AI progress", fontsize=10)
    ax[1].set_title("Long-run change in ln W (pct.), AI progress", fontsize=10)
    for x_ in ax:
        x_.set_xlabel("elasticity of substitution, AI vs. K-L bundle")
        x_.grid(alpha=0.25)
        x_.axvline(1.0, color="0.7", lw=1)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Same 10% long-run output gain; regular-capital progress leaves r unchanged in every model",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_saving_longrun.{ext}"), dpi=150)
    plt.close(fig)

    # ---------- figure 2: transitions, AI and K progress, three saving models
    H = 40
    t = np.arange(H)
    for cap in ("A", "K"):
        fig, axes = plt.subplots(3, 3, figsize=(13, 8.4), sharey="row")
        for col, case in enumerate(("jump", "mixed", "scurve")):
            for m, (c_, ls) in (("ramsey", ("0.3", "-")), ("py", ("#1f5f9e", "-")),
                                ("olg", ("#7a4fb0", "-")), ("solow", ("#c0512a", "-"))):
                for typ, lw, ls2 in (("dis", 2.0, "-"), ("emb", 1.6, "--")):
                    res = out[(m, case, cap, typ)]
                    lab = f"{dict(MODELS)[m]}, {'disembodied' if typ == 'dis' else 'embodied'}"
                    axes[0, col].plot(t, 100 * res["r"][:H], color=c_, ls=ls2, lw=lw, label=lab)
                    axes[1, col].plot(t, 100 * res["lnW"][:H], color=c_, ls=ls2, lw=lw)
                    axes[2, col].plot(t, 100 * res["labsh"][:H], color=c_, ls=ls2, lw=lw)
            axes[0, col].set_title({"jump": "Surprise jump at t = 0", "mixed": "Half jump, half S-curve",
                                    "scurve": "S-curve, complete by year 10"}[case], fontsize=10)
            axes[2, col].set_xlabel("years")
            for x_ in axes[:, col]:
                x_.grid(alpha=0.25)
        axes[0, 0].set_ylabel("real interest rate (pct.)")
        axes[1, 0].set_ylabel("real wage, log dev. (pct.)")
        axes[2, 0].set_ylabel("labor share (pct.)")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(("AI" if cap == "A" else "Regular-capital") + " progress, e = 1.5, d_A = d_K, "
                     "no adjustment costs, same 10% long-run output gain", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"fig_saving_trans_{cap}.{ext}"), dpi=150)
        plt.close(fig)

    # ---------- substitutability and r under Ramsey: transitions for several elasticities
    grid = {}
    for e_ in (0.75, 1.0, 1.5, 2.0):
        sv = Saving("ramsey", e_)
        for case in ("jump", "scurve"):
            for typ in ("dis", "emb"):
                res = Transition(sv, case, "A", typ).solve()
                checks[f"Ramsey grid e={e_} {case} {typ}: residual"] = res["err"]
                cum = np.sum(np.log1p(res["r"][:40]) - np.log1p(r0))
                grid[(e_, case, typ)] = dict(r1=100 * res["r"][0], mx=100 * res["r"].max(),
                                             mn=100 * res["r"].min(), cum=100 * cum,
                                             W0=100 * res["lnW"][0], WLR=100 * res["lnW"][-1],
                                             LS=100 * res["labsh"][-1], size=res["size"])
    G_ = [r"\begin{tabular}{lcccccccc}\toprule",
          r" & \multicolumn{2}{c}{jump, disemb.} & \multicolumn{2}{c}{jump, emb.} & S-curve, disemb. & S-curve, emb. & \multicolumn{2}{c}{long run}\\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}",
          r"$\e$ (AI) & $r_1$ & 40-yr excess & $r_1$ & 40-yr excess & peak & trough/peak & $\Delta\ln W$ & labor share\\\midrule"]
    for e_ in (0.75, 1.0, 1.5, 2.0):
        jd, je = grid[(e_, "jump", "dis")], grid[(e_, "jump", "emb")]
        sd, se = grid[(e_, "scurve", "dis")], grid[(e_, "scurve", "emb")]
        G_.append(f"{e_:g} & {jd['r1']:.2f} & {jd['cum']:.1f} & {je['r1']:.2f} & {je['cum']:.1f} & "
                  f"{sd['mx']:.2f} & {se['mn']:.2f}/{se['mx']:.2f} & {jd['WLR']:.1f} & {jd['LS']:.1f}\\\\")
    G_.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_saving_epsgrid.tex"), "w") as f:
        f.write("\n".join(G_) + "\n")

    # ---------- tables
    L_ = [r"\begin{tabular}{llcccccc}\toprule",
          r"$\e$ & saving model & shock (log) & $\Delta r$ (pp) & long-run $r$ & $\Delta\ln W$ & labor share & AI share\\\midrule"]
    for e_ in eps:
        for i, (m, mname) in enumerate(MODELS):
            row = next(r[3] for r in lr_rows if r[0] == e_ and r[1] == m and r[2] == "A")
            if row is None:
                L_.append(f"{e_:g} & {mname} & \\multicolumn{{6}}{{c}}{{10\\% output gain not attainable}}\\\\")
                continue
            L_.append(f"{e_:g} & {mname} & {row['size']:.2f} & {row['dr']:+.2f} & {row['r']:.2f} & "
                      f"{row['dW']:.1f} & {row['LS']:.1f} & {row['sA']:.1f}\\\\")
        L_.append(r"\midrule")
    for i, (m, mname) in enumerate(MODELS):
        row = next(r[3] for r in lr_rows if r[0] == 1.5 and r[1] == m and r[2] == "K")
        L_.append(f"{'reg. capital' if i == 0 else ''} & {mname} & {row['size']:.2f} & {row['dr']:+.2f} & "
                  f"{row['r']:.2f} & {row['dW']:.1f} & {row['LS']:.1f} & {row['sA']:.1f}\\\\")
    L_.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_saving_longrun.tex"), "w") as f:
        f.write("\n".join(L_) + "\n")

    def fm(x):
        return f"{round(x, 2) + 0.0:.2f}"
    T_ = [r"\begin{tabular}{lllcccccc}\toprule",
          r" & & & \multicolumn{3}{c}{real rate (pct.)} & \multicolumn{2}{c}{$\ln W$ (pct.)} & labor share\\",
          r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
          r"timing & progress & saving model & $r_1$ & max & long run & $t{=}10$ & long run & $t{=}10$\\\midrule"]
    for case in ("jump", "mixed", "scurve"):
        for cap in ("A", "K"):
            for typ in ("dis", "emb"):
                for i, m in enumerate(TMODELS):
                    res = out[(m, case, cap, typ)]
                    lab = ("AI" if cap == "A" else "Reg.") + (" disemb." if typ == "dis" else " emb.")
                    tname = {"jump": "Jump", "mixed": "Half/half", "scurve": "S-curve"}[case]
                    T_.append(f"{tname if (cap == 'A' and typ == 'dis' and i == 0) else ''} & "
                              f"{lab if i == 0 else ''} & {dict(MODELS)[m].split(' (')[0]} & "
                              f"{fm(100*res['r'][0])} & {fm(100*res['r'].max())} & {fm(100*res['rT'])} & "
                              f"{100*res['lnW'][10]:.1f} & {100*res['lnW'][res['lr']]:.1f} & {100*res['labsh'][10]:.1f}\\\\")
        if case != "scurve":
            T_.append(r"\midrule")
    T_.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_saving_trans.tex"), "w") as f:
        f.write("\n".join(T_) + "\n")
    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k.replace('%', chr(92) + '%')} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_saving_checks.tex"), "w") as f:
        f.write("\n".join(C) + "\n")
    with open(os.path.join(HERE, "results_saving_models.json"), "w") as f:
        json.dump(dict(longrun=[dict(e=r[0], model=r[1], cap=r[2], **(r[3] or {})) for r in lr_rows],
                       eps_grid={f"{k[0]} {k[1]} {k[2]}": v for k, v in grid.items()},
                       transitions={f"{k[0]} {k[1]} {k[2]} {k[3]}": dict(
                           r1=float(v["r"][0]), mx=float(v["r"].max()), mn=float(v["r"].min()),
                           tmx=int(np.argmax(v["r"])), tmn=int(np.argmin(v["r"])), rLR=float(v["rT"]),
                           W0=float(v["lnW"][0]), W10=float(v["lnW"][10]), WLR=float(v["lnW"][v["lr"]]),
                           Y10=float(v["lnY"][10]), LS10=float(v["labsh"][10]), LSLR=float(v["labsh"][v["lr"]]),
                           size=float(v["size"])) for k, v in out.items()},
                       checks=checks, p=P_PY, L=L), f, indent=1, default=float)
    print("max check:", max(checks.values()))
    for k, v in checks.items():
        if v > 1e-6:
            print("  LARGE", k, v)


if __name__ == "__main__":
    main()
