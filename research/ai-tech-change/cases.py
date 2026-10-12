"""
How do real interest rates and real wages respond to AI versus other technical progress?

Three cases, each with disembodied progress (Gamma in production) and embodied progress
(P, the price of new capital in efficiency units, in the accumulation equation), each
shock scaled to raise long-run output by the same 10 percent:

  Case 1  one capital stock, CES with labor, complements (e = 0.5)
  Case 2  one capital stock, CES with labor, substitutes (e = 1.5)
  Case 3  regular capital K in a bundle X with labor (Cobb-Douglas, or CES complement),
          AI capital A a substitute for the bundle (e_2 = 1.5), small initially;
          progress in K versus progress in A.

Benchmark: labor-augmenting progress Z with the same long-run output gain.

Accumulation follows the Kiley (2004) pairing: each capital stock accumulates through a
CES function with the elasticity of the production nest it enters (Cobb-Douglas when that
elasticity is one), normalized to agree to first order with K' = (1-d)K + I on the
initial balanced growth path (so Q = P there).

Cases 1-2 with IES = e are solved exactly (state-independent saving rates from a backward
recursion) and checked against a generic Newton solve.  All cases are also solved
numerically with a common IES = 1 (log utility, as in the paper).  Hours are fixed.
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

COMMON = dict(g=0.015, r0=0.075, d=dict(K=0.066, A=0.20), sK=0.282, sA=0.03,
              target=0.10, T=150, mid=10.0, width=2.5)


def scurve(T, mid=COMMON["mid"], width=COMMON["width"]):
    t = np.arange(T + 1)
    S = 1 / (1 + np.exp(-(t - mid) / width))
    return (S - S[0]) / (1 - S[0])


def rho_of(e):
    return 0.0 if e == 1.0 else (e - 1) / e


# ----------------------------------------------------------------------------
# Economy: production, accumulation, normalization
# ----------------------------------------------------------------------------
class Economy:
    """kind = 'one' (K and L) or 'two' (X(K, L; e1) and A; e2)."""

    def __init__(self, kind, e, e2=None, ies=1.0, c=COMMON):
        self.kind, self.e, self.e2, self.ies, self.c = kind, e, e2, ies, c
        self.G = 1 + c["g"]
        self.beta = self.G ** (1 / ies) / (1 + c["r0"])
        self.caps = ["K"] if kind == "one" else ["K", "A"]
        R = {j: c["r0"] + c["d"][j] for j in self.caps}
        self.R = R
        if kind == "one":
            s = {"K": c["sK"]}
            self.K0 = {"K": s["K"] / R["K"]}
            r = rho_of(e)
            self.rho = {"K": r}
            if r != 0:
                self.aK = (s["K"] * R["K"] ** (e - 1)) ** (1 / e)
                self.aL = 1 - s["K"]
            else:
                self.alpha = s["K"]
        else:
            sA, sK = c["sA"], c["sK"] * (1 - c["sA"])
            self.shares = dict(K=sK, A=sA, L=1 - sK - sA)
            self.K0 = {"K": sK / R["K"], "A": sA / R["A"]}
            r1, r2 = rho_of(e), rho_of(e2)
            self.rho = {"K": r1, "A": r2}
            self.cA = (sA * R["A"] ** (e2 - 1)) ** (1 / e2)
            self.cX = 1 - sA
            self.sig = sK / (1 - sA)                    # K's share of bundle income
            if r1 != 0:
                self.bK = self.sig * self.K0["K"] ** (-r1)
                self.bL = 1 - self.sig
        # accumulation normalization (Q = P = 1 on the initial BGP)
        self.delta, self.Abar = {}, {}
        for j in self.caps:
            x, d, r = c["g"] + c["d"][j], c["d"][j], self.rho[j]
            self.delta[j] = x ** (1 - r) / (x ** (1 - r) + 1 - d)
            self.Abar[j] = (self.G ** (r - 1) * (x ** (1 - r) + 1 - d)) ** (1 / r) if r != 0 else \
                self.G ** (1 - x / self.G) / (x / self.G) ** (x / self.G)
            if r == 0:
                self.delta[j] = x / self.G

    # -- production: returns Y, dict of MP_j (per unit of K_j), MP_L
    def output(self, K, L, Gam):
        if self.kind == "one":
            k, gk, r = K["K"], Gam["K"], self.rho["K"]
            if r == 0:
                Y = (gk * k / self.K0["K"]) ** self.alpha * L ** (1 - self.alpha)
                return Y, {"K": self.alpha * Y / k}, (1 - self.alpha) * Y / L
            Y = (self.aK * (gk * k) ** r + self.aL * L ** r) ** (1 / r)
            return Y, {"K": self.aK * gk ** r * k ** (r - 1) * Y ** (1 - r)}, \
                self.aL * L ** (r - 1) * Y ** (1 - r)
        k, a = K["K"], K["A"]
        r1, r2 = self.rho["K"], self.rho["A"]
        if r1 == 0:
            X = (Gam["K"] * k / self.K0["K"]) ** self.sig * L ** (1 - self.sig)
            XK, XL = self.sig * X / k, (1 - self.sig) * X / L
        else:
            X = (self.bK * (Gam["K"] * k) ** r1 + self.bL * L ** r1) ** (1 / r1)
            XK = self.bK * Gam["K"] ** r1 * k ** (r1 - 1) * X ** (1 - r1)
            XL = self.bL * L ** (r1 - 1) * X ** (1 - r1)
        Y = (self.cX * X ** r2 + self.cA * (Gam["A"] * a) ** r2) ** (1 / r2)
        YX = self.cX * X ** (r2 - 1) * Y ** (1 - r2)
        YA = self.cA * Gam["A"] ** r2 * a ** (r2 - 1) * Y ** (1 - r2)
        return Y, {"K": YX * XK, "A": YA}, YX * XL

    # -- accumulation: K' and partial derivatives
    def accum(self, j, K, I, P):
        r, dl, Ab = self.rho[j], self.delta[j], self.Abar[j]
        if r == 0:
            Kn = Ab * K ** (1 - dl) * (I / P) ** dl
            return Kn, dl * Kn / I, (1 - dl) * Kn / K
        Kn = Ab * ((1 - dl) * K ** r + dl * (I / P) ** r) ** (1 / r)
        dI = Ab ** r * dl * P ** (-r) * I ** (r - 1) * Kn ** (1 - r)
        dK = Ab ** r * (1 - dl) * K ** (r - 1) * Kn ** (1 - r)
        return Kn, dI, dK

    # -- steady state (detrended, L = 1) for constant technology
    def steady(self, Gam, P):
        r = self.G ** (1 / self.ies) / self.beta - 1
        lk = {j: np.log(self.K0[j]) for j in self.caps}
        for _ in range(200):
            K = {j: np.exp(lk[j]) for j in self.caps}
            f = np.array([np.log(self.output(K, 1.0, Gam)[1][j] / (P[j] * (r + self.c["d"][j])))
                          for j in self.caps])
            if np.max(np.abs(f)) < 1e-13:
                break
            J = np.empty((len(self.caps),) * 2)
            for i, jj in enumerate(self.caps):
                K2 = dict(K); K2[jj] = K[jj] * np.exp(1e-7)
                f2 = np.array([np.log(self.output(K2, 1.0, Gam)[1][j] / (P[j] * (r + self.c["d"][j])))
                               for j in self.caps])
                J[:, i] = (f2 - f) / 1e-7
            step = np.linalg.solve(J, -f)
            for i, j in enumerate(self.caps):
                lk[j] += step[i]
        K = {j: np.exp(lk[j]) for j in self.caps}
        Y, MP, W = self.output(K, 1.0, Gam)
        s = {j: P[j] * (self.G - 1 + self.c["d"][j]) * K[j] / Y for j in self.caps}
        return dict(K=K, Y=Y, W=W, s=s, r=r, labsh=W / Y)


# ----------------------------------------------------------------------------
# Paths: technology and the generic perfect-foresight solver
# ----------------------------------------------------------------------------
def tech_paths(eco, source, size, T, step=False):
    """source: ('dis'|'emb', capital) or ('Z', None)."""
    S = np.ones(T + 1) if step else scurve(T)
    t = np.arange(T + 1)
    Z = eco.G ** t
    Gam = {j: np.ones(T + 1) for j in eco.caps}
    P = {j: np.ones(T + 1) for j in eco.caps}
    kind, j = source
    if kind == "dis":
        Gam[j] = np.exp(size * S)
    elif kind == "emb":
        P[j] = np.exp(-size * S)
    else:
        Z = Z * np.exp(size * S)
    return Z, Gam, P


class Solver:
    def __init__(self, eco, Z, Gam, P):
        self.eco, self.Z, self.Gam, self.P = eco, Z, Gam, P
        self.T = len(Z) - 1
        ssT = eco.steady({j: Gam[j][-1] for j in eco.caps}, {j: P[j][-1] for j in eco.caps})
        self.term = [ssT["s"][j] for j in eco.caps]
        self.ssT = ssT
        self.n = len(eco.caps)

    def simulate(self, s):
        """s: array (n, T+1) of saving rates (I_j / Y)."""
        eco, T = self.eco, self.T
        K = {j: np.empty(T + 2) for j in eco.caps}
        for j in eco.caps:
            K[j][0] = eco.K0[j]
        Y = np.empty(T + 1); W = np.empty(T + 1)
        MP = {j: np.empty(T + 1) for j in eco.caps}
        dI = {j: np.empty(T + 1) for j in eco.caps}
        dK = {j: np.empty(T + 1) for j in eco.caps}
        for t in range(T + 1):
            Kt = {j: K[j][t] for j in eco.caps}
            Y[t], mp, W[t] = eco.output(Kt, self.Z[t], {j: self.Gam[j][t] for j in eco.caps})
            for i, j in enumerate(eco.caps):
                MP[j][t] = mp[j]
                K[j][t + 1], dI[j][t], dK[j][t] = eco.accum(j, K[j][t], s[i, t] * Y[t], self.P[j][t])
        C = (1 - s.sum(axis=0)) * Y
        return dict(K=K, Y=Y, W=W, MP=MP, dI=dI, dK=dK, C=C, s=s)

    def resid(self, z):
        T, n, eco = self.T, self.n, self.eco
        s = np.hstack([z.reshape(n, T), np.array(self.term)[:, None]])
        if np.any(s <= 0) or np.any(s.sum(axis=0) >= 1):
            return np.full(n * T, 1e3)
        with np.errstate(all="ignore"):
            R = self.simulate(s)
        C = R["C"]
        m = eco.beta * (C[1:] / C[:-1]) ** (-1 / eco.ies)
        out = []
        for j in eco.caps:
            Q = 1 / R["dI"][j]
            dKn = R["dK"][j]
            out.append(m[:T] * (R["MP"][j][1:] + Q[1:] * dKn[1:]) / Q[:T] - 1)
        res = np.concatenate(out)
        return np.where(np.isfinite(res), res, 1e3)

    def solve(self, z0=None, tol=1e-11, maxit=80):
        T, n = self.T, self.n
        if z0 is None:
            z0 = np.concatenate([np.full(T, self.term[i]) for i in range(n)])
        z, Rv = z0.copy(), self.resid(z0)
        N = z.size
        for it in range(maxit):
            nr = np.max(np.abs(Rv))
            if nr < tol:
                break
            J = np.empty((N, N))
            for i in range(N):
                h = 1e-7 * max(1.0, abs(z[i]))
                dz = np.zeros(N); dz[i] = h
                J[:, i] = (self.resid(z + dz) - Rv) / h
            step = np.linalg.solve(J, -Rv)
            lam = 1.0
            while lam > 1e-6:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, Rv = zn, Rn
        self.iters, self.err = it, float(np.max(np.abs(Rv)))
        s = np.hstack([z.reshape(n, T), np.array(self.term)[:, None]])
        return self.report(s)

    def report(self, s):
        eco = self.eco
        R = self.simulate(s)
        C, Y = R["C"], R["Y"]
        r = (C[1:] / C[:-1]) ** (1 / eco.ies) / eco.beta - 1
        base_t = eco.G ** np.arange(self.T + 1)
        # R["W"] is dY/d(ZN); the wage per hour is Z times that, reported relative to trend Z
        return dict(r=r, W=R["W"] * self.Z / base_t, Y=Y / base_t, labsh=R["W"] * self.Z / Y,
                    s=s, C=C / base_t, err=getattr(self, "err", 0.0), zT=self.Z[-1] / base_t[-1])


# ----------------------------------------------------------------------------
# Exact solution for the one-capital CES pair with IES = e (cases 1-2)
# ----------------------------------------------------------------------------
def exact_one(eco, Z, Gam, P):
    """Backward recursion: phi_t = beta kappa_{t+1} Abar^rho delta P_t^-rho,
    s_t = phi^e/(1+phi^e), Omega_t = (1+phi^e)^(1/e),
    kappa_t = Omega_t a_K Gamma_t^rho + beta Abar^rho (1-delta) kappa_{t+1}."""
    assert eco.kind == "one" and eco.ies == eco.e
    e, r = eco.e, eco.rho["K"]
    b, Ab, dl, aK = eco.beta, eco.Abar["K"] ** r, eco.delta["K"], eco.aK
    T = len(Z) - 1
    g, p = Gam["K"], P["K"]

    def step(kap, t):
        phi = b * kap * Ab * dl * p[t] ** (-r)
        Om = (1 + phi ** e) ** (1 / e)
        return Om * aK * g[t] ** r + b * Ab * (1 - dl) * kap, phi ** e / (1 + phi ** e)

    kap = 1.0
    for _ in range(50000):                     # terminal fixed point
        new, sT = step(kap, T)
        if abs(new - kap) < 1e-15:
            break
        kap = new
    s = np.empty(T + 1)
    for t in range(T, -1, -1):
        kap, s[t] = step(kap, t)
    return s


def calibrate_size(eco, source, target=COMMON["target"]):
    """Shock size giving ln Y_ss(new) - ln Y_ss(old) = target."""
    kind, j = source
    if kind == "Z":
        return target                           # Y/Z unchanged at constant r
    ones = {k: 1.0 for k in eco.caps}
    Y0 = eco.steady(ones, ones)["Y"]

    def gain(x):
        Gam, P = dict(ones), dict(ones)
        if kind == "dis":
            Gam[j] = np.exp(x)
        else:
            P[j] = np.exp(-x)
        return np.log(eco.steady(Gam, P)["Y"] / Y0) - target

    lo, hi = 0.0, 0.5
    while gain(hi) < 0:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if gain(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def closed_form_size_one(eco, target=COMMON["target"]):
    """Case 1-2 closed form: (Gamma/P)^(e-1) = [1 - (1-s_K)exp(-rho*target)]/s_K."""
    e, r, sK = eco.e, eco.rho["K"], COMMON["sK"]
    if r == 0:
        return target / sK * (1 - sK)
    return np.log(((1 - (1 - sK) * np.exp(-r * target)) / sK) ** (1 / (e - 1)))


# ----------------------------------------------------------------------------
def run():
    T = COMMON["T"]
    checks, out = {}, {}
    SOURCES1 = [("dis", "K"), ("emb", "K"), ("Z", None)]

    # ---------------- Cases 1-2: exact (IES = e) and numerical (IES = 1)
    for label, e in (("case1", 0.5), ("case2", 1.5)):
        for ies_mode in ("exact", "log"):
            eco = Economy("one", e, ies=(e if ies_mode == "exact" else 1.0))
            out[(label, ies_mode)] = {}
            for src in SOURCES1:
                size = calibrate_size(eco, src)
                if src[0] != "Z":
                    checks[f"{label}: shock size = closed form ({src[0]})"] = abs(
                        size - closed_form_size_one(eco))
                for step in (False, True):
                    Z, Gam, P = tech_paths(eco, src, size, T, step)
                    sol = Solver(eco, Z, Gam, P)
                    if ies_mode == "exact":
                        s_cf = exact_one(eco, Z, Gam, P)
                        rep_cf = sol.report(s_cf[None, :])
                        num = sol.solve(z0=0.95 * s_cf[:T])
                        checks[f"{label} {src[0]} {'step' if step else 'S'}: generic = exact s"] = float(
                            np.max(np.abs(num["s"][0] - s_cf)))
                        res = rep_cf
                    else:
                        res = sol.solve()
                    res["size"] = size
                    res["ssT"] = sol.ssT
                    out[(label, ies_mode)][(src[0], step)] = res
                    print(f"  {label} IES={'e' if ies_mode=='exact' else 1} {src[0]:3s} "
                          f"{'step' if step else 'S  '} size={size:.4f} err={getattr(sol,'err',0):.1e} "
                          f"peak r={100*res['r'].max():.2f} W_T={res['W'][-1]/res['W'][0]:.4f}")
            # long-run closed forms: ln W gain = target/e (capital progress), = target (Z)
            for src in SOURCES1:
                res = out[(label, ies_mode)][(src[0], False)]
                ss0 = eco.steady({"K": 1.0}, {"K": 1.0})
                lw = np.log(res["ssT"]["W"] * res["zT"] / ss0["W"])
                want = COMMON["target"] / e if src[0] != "Z" else COMMON["target"]
                checks[f"{label} IES={ies_mode}: long-run ln W = {'target/e' if src[0]!='Z' else 'target'} ({src[0]})"] = abs(lw - want)

    # ---------------- Case 3: numerical, IES = 1, inner CD (baseline) and CES 0.5
    SOURCES3 = [("dis", "K"), ("emb", "K"), ("dis", "A"), ("emb", "A"), ("Z", None)]
    for label, e1 in (("case3", 1.0), ("case3_ces", 0.5)):
        eco = Economy("two", e1, e2=1.5, ies=1.0)
        out[(label, "log")] = {}
        ss0 = eco.steady({"K": 1.0, "A": 1.0}, {"K": 1.0, "A": 1.0})
        checks[f"{label}: initial steady state reproduces calibration"] = max(
            abs(ss0["Y"] - 1), abs(ss0["K"]["K"] - eco.K0["K"]), abs(ss0["K"]["A"] - eco.K0["A"]))
        for src in SOURCES3:
            size = calibrate_size(eco, src)
            for step in (False, True):
                Z, Gam, P = tech_paths(eco, src, size, T, step)
                sol = Solver(eco, Z, Gam, P)
                res = sol.solve()
                res["size"], res["ssT"] = size, sol.ssT
                key = (src[0] + (src[1] or ""), step)
                out[(label, "log")][key] = res
                print(f"  {label} {key[0]:5s} {'step' if step else 'S  '} size={size:.4f} "
                      f"iters={sol.iters} err={sol.err:.1e} peak r={100*res['r'].max():.2f} "
                      f"W_T={res['W'][-1]/res['W'][0]:.4f} labsh_T={res['labsh'][-1]:.3f}")
                checks[f"{label} {key[0]} {'step' if step else 'S'}: Newton residual"] = res["err"]
    return out, checks


# ----------------------------------------------------------------------------
COLS = {"dis": "#1f5f9e", "emb": "#c0512a", "Z": "0.45"}
NAMES = {"dis": "disembodied", "emb": "embodied", "Z": "labor-augmenting (benchmark)"}


def _min_span(ax, span=2.0):
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        c = 0.5 * (lo + hi)
        ax.set_ylim(c - span / 2, c + span / 2)


def fig_cases12(out, mode, fname, H=40):
    t = np.arange(H)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.4))
    for col, (label, e) in enumerate((("case1", 0.5), ("case2", 1.5))):
        R = out[(label, mode)]
        for src in ("dis", "emb", "Z"):
            res = R[(src, False)]
            ls = ":" if src == "Z" else "-"
            axes[0, col].plot(t, 100 * res["r"][:H], color=COLS[src], ls=ls, lw=2, label=NAMES[src])
            axes[1, col].plot(t, 100 * (res["W"][:H] / res["W"][0] - 1), color=COLS[src], ls=ls, lw=2)
            axes[2, col].plot(t, 100 * res["labsh"][:H], color=COLS[src], ls=ls, lw=2)
        ies = f"IES = {e:g}" if mode == "exact" else "IES = 1"
        axes[0, col].set_title(f"Case {col+1}: e = {e:g} ({'complements' if e < 1 else 'substitutes'}), {ies}",
                               fontsize=10)
        axes[1, col].axhline(100 * np.expm1(COMMON["target"] / e), color="0.6", lw=1, ls="--")
        axes[1, col].axhline(100 * np.expm1(COMMON["target"]), color="0.6", lw=1, ls=":")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
        axes[2, col].set_xlabel("years after announcement")
        _min_span(axes[2, col])
    axes[0, 0].set_ylabel("real interest rate (pct.)")
    axes[1, 0].set_ylabel("real wage W/Z (pct. dev.)")
    axes[2, 0].set_ylabel("labor share (pct.)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("One capital stock: shocks with the same 10% long-run output gain "
                 f"({'exact solution' if mode == 'exact' else 'numerical, log utility'})", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{fname}.{ext}"), dpi=150)
    plt.close(fig)


def fig_case3(out, label, fname, H=40):
    t = np.arange(H)
    R = out[(label, "log")]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.4))
    for col, cap in enumerate(("K", "A")):
        for src in ("dis", "emb"):
            res = R[(src + cap, False)]
            axes[0, col].plot(t, 100 * res["r"][:H], color=COLS[src], lw=2, label=NAMES[src])
            axes[1, col].plot(t, 100 * (res["W"][:H] / res["W"][0] - 1), color=COLS[src], lw=2)
            axes[2, col].plot(t, 100 * res["labsh"][:H], color=COLS[src], lw=2)
        z = R[("Z", False)]
        axes[0, col].plot(t, 100 * z["r"][:H], color=COLS["Z"], ls=":", lw=2, label=NAMES["Z"])
        axes[1, col].plot(t, 100 * (z["W"][:H] / z["W"][0] - 1), color=COLS["Z"], ls=":", lw=2)
        axes[2, col].plot(t, 100 * z["labsh"][:H], color=COLS["Z"], ls=":", lw=2)
        axes[0, col].set_title("progress in regular capital K" if cap == "K" else "progress in AI capital A",
                               fontsize=10)
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
        axes[2, col].set_xlabel("years after announcement")
        _min_span(axes[2, col])
    axes[0, 0].set_ylabel("real interest rate (pct.)")
    axes[1, 0].set_ylabel("real wage W/Z (pct. dev.)")
    axes[2, 0].set_ylabel("labor share (pct.)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    inner = "Cobb-Douglas" if label == "case3" else "CES, e1 = 0.5"
    fig.suptitle(f"Case 3: K-L bundle {inner}; AI substitutes for the bundle (e2 = 1.5); "
                 "same 10% long-run output gain, log utility", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{fname}.{ext}"), dpi=150)
    plt.close(fig)


def summary_rows(R, keys):
    rows = []
    for name, key in keys:
        S, St = R[(key, False)], R[(key, True)]
        cum = np.sum(np.log1p(S["r"]) - np.log1p(COMMON["r0"]))
        rows.append(dict(name=name, size=S["size"], peak_r=100 * S["r"].max(),
                         t_peak=int(np.argmax(S["r"])), r1_step=100 * St["r"][0],
                         w10=100 * (S["W"][10] / S["W"][0] - 1),
                         wT=100 * (S["ssT"]["W"] * S["zT"] / S["W"][0] - 1),
                         lsT=100 * S["ssT"]["labsh"], sT=sum(S["ssT"]["s"].values()),
                         cum_r=100 * cum))
    return rows


def tex_table(rows, path, caption_cols=True):
    L = [r"\begin{tabular}{lcccccccc}\toprule",
         r" & shock & peak $r$ & year & $r_1$, step & $W$, $t{=}10$ & $W$, long run & labor share & $\sum\Delta\ln(1+r)$\\",
         r" & (log) & (pct.) & & (pct.) & (pct.) & (pct.) & long run (pct.) & (pct.)\\\midrule"]
    for r in rows:
        L.append(f"{r['name']} & {r['size']:.3f} & {r['peak_r']:.2f} & {r['t_peak']} & {r['r1_step']:.2f} & "
                 f"{r['w10']:.1f} & {r['wT']:.1f} & {r['lsT']:.1f} & {r['cum_r']:.1f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    out, checks = run()
    fig_cases12(out, "exact", "fig_case12_exact")
    fig_cases12(out, "log", "fig_case12_log")
    fig_case3(out, "case3", "fig_case3")
    fig_case3(out, "case3_ces", "fig_case3_ces")

    all_rows = {}
    for label, mode, keys in (
            ("case1", "exact", [("Disembodied", "dis"), ("Embodied", "emb"), ("Labor-augmenting", "Z")]),
            ("case2", "exact", [("Disembodied", "dis"), ("Embodied", "emb"), ("Labor-augmenting", "Z")]),
            ("case1", "log", [("Disembodied", "dis"), ("Embodied", "emb"), ("Labor-augmenting", "Z")]),
            ("case2", "log", [("Disembodied", "dis"), ("Embodied", "emb"), ("Labor-augmenting", "Z")]),
            ("case3", "log", [("$K$ disembodied", "disK"), ("$K$ embodied", "embK"),
                              ("AI disembodied", "disA"), ("AI embodied", "embA"),
                              ("Labor-augmenting", "Z")]),
            ("case3_ces", "log", [("$K$ disembodied", "disK"), ("$K$ embodied", "embK"),
                                  ("AI disembodied", "disA"), ("AI embodied", "embA"),
                                  ("Labor-augmenting", "Z")])):
        rows = summary_rows(out[(label, mode)], keys)
        all_rows[f"{label}_{mode}"] = rows
        tex_table(rows, os.path.join(HERE, f"table_{label}_{mode}.tex"))

    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k.replace('_', ' ')} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_checks_cases.tex"), "w") as f:
        f.write("\n".join(C) + "\n")
    with open(os.path.join(HERE, "results_cases.json"), "w") as f:
        json.dump(dict(rows=all_rows, checks=checks, common=COMMON), f, indent=1, default=float)
    print("Checks:")
    for k, v in checks.items():
        print(f"  {k:70s} {v:.2e}")
    for k, rows in all_rows.items():
        print(k)
        for r in rows:
            print("   ", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})


if __name__ == "__main__":
    main()
