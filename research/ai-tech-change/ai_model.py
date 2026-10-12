"""
AI-driven technical change in the Kiley (2000/2004) production economy.

Blueprint: Kiley, "Stock Prices and Fundamentals in a Production Economy,"
FEDS 2000-05 (published as "Stock Prices and Fundamentals: A Macroeconomic
Perspective," Journal of Business 77(4), 2004).  Notation follows that paper:

  a  capital share in the traditional bundle      d_K  depreciation of K
  b  consumption weight in utility                s    curvature, U = [C^b L^(1-b)]^(1-s)/(1-s)
  B  discount factor                              c_K  adjustment-cost parameter
  Z  labor-augmenting technology (grows at g_Z)   N    hours, L = 1 - N leisure
  Q  price (marginal value) of capital            V    market value of firms
  Adjustment cost: 0.5 c (I/K - x)^2 K with x = g + d  (paper eq. 2.7 and p. 11)

Extension (this note):
  X = K^a (Z N)^(1-a)                                    traditional bundle
  F = [w^(1/e) X^rho + (1-w)^(1/e) M^rho]^(1/rho)        rho = (e-1)/e
  M = Gamma * A                                          effective AI
  Gamma : disembodied AI progress (grows at gamma_G)
  P_A   : price of AI investment in efficiency units, falls at gamma_q
          (embodied, Greenwood-Hercowitz-Krusell)
  A(t+1) = (1-d_A) A(t) + I_A(t);  AI spending in output units = P_A I_A
  AI adjustment cost: P_A * 0.5 c_A (I_A/A - x_A)^2 A

The script
  1. evaluates the closed-form balanced-growth expressions,
  2. verifies them against the equilibrium conditions (levels, no detrending),
  3. solves perfect-foresight transitions for an S-curve AI shock that is
     either disembodied (Gamma) or embodied (P_A), for e < 1, = 1, > 1,
  4. writes figures and LaTeX tables used in ai_tech_change.tex.

Run:  python3 ai_model.py      (needs numpy and matplotlib)
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

# ----------------------------------------------------------------------------
# Calibration: Kiley (2000), Table 1, moderate adjustment costs, plus AI block
# ----------------------------------------------------------------------------
BASE = dict(
    a=0.282,      # Table 1
    b=0.25,       # Table 1
    s=1.0,        # Table 1 (log utility)
    d_K=0.066,    # Table 1
    c_K=15.0,     # "moderate adjustment costs"
    r0=0.075,     # Table 1 required return in initial steady state
    g_Z=0.015,    # paper sets g = 0 "for convenience"; we allow trend growth
    d_A=0.20,     # AI capital (compute, software, models) depreciates fast
    c_A=15.0,
    sM0=0.03,     # initial AI share of gross income
)
BASE["theta"] = 1.0 - BASE["b"] * (1.0 - BASE["s"])           # 1/IES (paper fn. 11)
BASE["B"] = (1.0 + BASE["g_Z"]) ** BASE["theta"] / (1.0 + BASE["r0"])
BASE["omega"] = 1.0 - BASE["sM0"]                              # weight on X
# normalize Gamma_0 so that lambda_0 = P_A u_A / Gamma = 1 (common initial BGP for all e)
BASE["Gamma0"] = BASE["r0"] + BASE["d_A"]
BASE["PA0"] = 1.0

DELTA = np.log(20.0)   # total log fall in the effective cost of AI (Gamma/P_A rises 20x)


# ----------------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------------
def ai_share(lam, e, p):
    """s_M = (1-omega) lambda^(1-e): AI share of gross income."""
    return (1.0 - p["omega"]) * lam ** (1.0 - e)


def bgp(p, e, gam_G=0.0, gam_q=0.0, Gamma=None, PA=None):
    """Closed-form balanced growth path.

    For e = 1 this is the exact BGP with permanent progress (gam_G, gam_q).
    For e != 1 a BGP requires gam_G = gam_q = 0 (constant Gamma/P_A).
    Returns ratios and the detrended wage W/Z at the date where Gamma, P_A apply.
    """
    if e != 1.0 and (gam_G or gam_q):
        raise ValueError("BGP with ongoing AI progress exists only for e = 1")
    Gamma = p["Gamma0"] if Gamma is None else Gamma
    PA = p["PA0"] if PA is None else PA
    a, om = p["a"], p["omega"]
    GZ = 1.0 + p["g_Z"]
    Gq, GG = 1.0 + gam_q, 1.0 + gam_G
    # growth: exact (1+g) = (1+g_Z) [(1+gam_G)(1+gam_q)]^((1-w)/((1-a)w))
    G = GZ * (GG * Gq) ** ((1.0 - om) / ((1.0 - a) * om)) if e == 1.0 else GZ
    r = G ** p["theta"] / p["B"] - 1.0                      # 1+r = (1+g)^theta / B
    R_K = r + p["d_K"]                                      # user cost of K (Q = 1)
    u_A = (1.0 + r) * Gq - (1.0 - p["d_A"])                 # user cost of AI, incl. obsolescence
    lam = PA * u_A / Gamma                                  # cost of an effective AI unit
    sM = ai_share(lam, e, p) if e != 1.0 else 1.0 - om
    if sM >= 1.0:
        return None                                         # AK region, no interior BGP
    if e == 1.0:
        PX = lam ** (-(1.0 - om) / om)
    else:
        PX = ((1.0 - sM) / om) ** (1.0 / (1.0 - e))
    wZ = (1.0 - a) * (PX * (a / R_K) ** a) ** (1.0 / (1.0 - a))   # W/Z
    KY = a * (1.0 - sM) / R_K
    AY = sM / u_A                                           # P_A A / Y
    GA = G * Gq                                             # growth of A (efficiency units)
    IKY = (G - 1.0 + p["d_K"]) * KY
    IAY = (GA - 1.0 + p["d_A"]) * AY                        # nominal AI investment / Y
    CY = 1.0 - IKY - IAY
    labsh = (1.0 - a) * (1.0 - sM)
    N = p["b"] * labsh / (p["b"] * labsh + (1.0 - p["b"]) * CY)
    VY = G * (KY + Gq * AY)                                 # V_t = Q_K K_{t+1} + Q_A A_{t+1}
    DY = 1.0 - labsh - IKY - IAY
    # paper-style (continuous-time) approximation of eq. (3.2)
    VY_approx = KY + sM / (r + p["d_A"] + gam_q)
    return dict(g=G - 1.0, r=r, R_K=R_K, u_A=u_A, lam=lam, sM=sM, PX=PX, wZ=wZ,
                KY=KY, AY=AY, IKY=IKY, IAY=IAY, CY=CY, labsh=labsh, N=N, VY=VY,
                DY=DY, VY_approx=VY_approx, G=G, GA=GA)


def ak_limit(p, e, Gamma_over_PA):
    """e > 1 and Gamma/P_A constant: AK balanced growth if it beats g_Z.

    Asymptotically F = Abar * M with Abar = (1-w)^(1/(e-1)); the AI Euler
    equation gives r = Abar*Gamma/P_A - d_A, and 1+g = [B(1+r)]^(1/theta).
    """
    Abar = (1.0 - p["omega"]) ** (1.0 / (e - 1.0))
    r = Abar * Gamma_over_PA - p["d_A"]
    G = (p["B"] * (1.0 + r)) ** (1.0 / p["theta"])
    a = p["a"]
    lnGZ, lnG = np.log(1.0 + p["g_Z"]), np.log(G)
    wage_growth = lnGZ + (lnG - lnGZ) / (e * (1.0 - a) + a)          # log growth of W
    lab_share_decline = (e - 1.0) * (1.0 - a) * (lnG - lnGZ) / (e * (1.0 - a) + a)
    return dict(Abar=Abar, r=r, g=G - 1.0, wage_growth=wage_growth,
                lab_share_decline=lab_share_decline)


def ak_threshold(p, e):
    """Gamma/P_A above which an interior BGP fails (s_M >= 1 at r = r_Z)."""
    Abar = (1.0 - p["omega"]) ** (1.0 / (e - 1.0))
    rZ = (1.0 + p["g_Z"]) ** p["theta"] / p["B"] - 1.0
    return (rZ + p["d_A"]) / Abar


def baumol_limit_wZ(p, e):
    """e < 1, Gamma/P_A -> infinity: P_X -> w^(-1/(1-e)), g -> g_Z."""
    a = p["a"]
    r = (1.0 + p["g_Z"]) ** p["theta"] / p["B"] - 1.0
    PX = p["omega"] ** (-1.0 / (1.0 - e))
    return (1.0 - a) * (PX * (a / (r + p["d_K"])) ** a) ** (1.0 / (1.0 - a))


# ----------------------------------------------------------------------------
# Production block
# ----------------------------------------------------------------------------
def prod(X, M, e, p):
    """Gross output and marginal products dF/dX, dF/dM."""
    om = p["omega"]
    if e == 1.0:
        kappa = om ** om * (1.0 - om) ** (1.0 - om)
        F = X ** om * M ** (1.0 - om) / kappa
        return F, om * F / X, (1.0 - om) * F / M
    rho = (e - 1.0) / e
    F = (om ** (1.0 / e) * X ** rho + (1.0 - om) ** (1.0 / e) * M ** rho) ** (1.0 / rho)
    FX = (om * F / X) ** (1.0 / e)
    FM = ((1.0 - om) * F / M) ** (1.0 / e)
    return F, FX, FM


# ----------------------------------------------------------------------------
# Verification 1: exact BGP in levels (no detrending) satisfies every FOC
# ----------------------------------------------------------------------------
def levels_check(p, e, gam_G, gam_q, T=60):
    ss = bgp(p, e, gam_G, gam_q)
    a, b, s = p["a"], p["b"], p["s"]
    t = np.arange(T + 1)
    Z = (1.0 + p["g_Z"]) ** t
    Gam = p["Gamma0"] * (1.0 + gam_G) ** t
    PA = p["PA0"] * (1.0 + gam_q) ** (-t)
    N = ss["N"]
    # build levels from the closed form: pin K_0 from K/Y and the production function
    # Y = F(K, Z N, Gamma A) with K = KY*Y, P_A A = AY*Y. Solve Y_0 by fixed point.
    Y0 = 1.0
    for _ in range(200):
        K0, A0 = ss["KY"] * Y0, ss["AY"] * Y0 / PA[0]
        X0 = K0 ** a * (Z[0] * N) ** (1 - a)
        Y0 = prod(X0, Gam[0] * A0, e, p)[0]
    Y = Y0 * ss["G"] ** t
    K = ss["KY"] * Y
    A = ss["AY"] * Y / PA
    X = K ** a * (Z * N) ** (1 - a)
    F, FX, FM = prod(X, Gam * A, e, p)
    MPK, MPN, MPA = FX * a * X / K, FX * (1 - a) * X / N, FM * Gam
    IK = K[1:] - (1 - p["d_K"]) * K[:-1]
    IA = A[1:] - (1 - p["d_A"]) * A[:-1]
    jK, jA = IK / K[:-1], IA / A[:-1]
    xK, xA = ss["G"] - 1 + p["d_K"], ss["GA"] - 1 + p["d_A"]
    CK = 0.5 * p["c_K"] * (jK - xK) ** 2 * K[:-1]
    CA = PA[:-1] * 0.5 * p["c_A"] * (jA - xA) ** 2 * A[:-1]
    Ynet = F[:-1] - CK - CA
    C = Ynet - IK - PA[:-1] * IA
    W = MPN[:-1]
    QK = 1 + p["c_K"] * (jK - xK)
    QA = PA[:-1] * (1 + p["c_A"] * (jA - xA))
    L = 1 - N
    mu = b * C ** (b * (1 - s) - 1) * L ** ((1 - b) * (1 - s))     # U_C (scale-free form)
    m = p["B"] * mu[1:] / mu[:-1]                                  # 1/(1+r_{t+1})
    res = {}
    res["F = Y (prod. fn. at closed form)"] = np.max(np.abs(F / Y - 1))
    res["labor FOC (1-b)C/(bL) = W"] = np.max(np.abs((1 - b) * C / (b * L) / W - 1))
    res["Euler K (2.12)"] = np.max(np.abs(m * (MPK[1:-1] + (1 - p["d_K"]) * QK[1:]) / QK[:-1] - 1))
    res["Euler A"] = np.max(np.abs(m * (MPA[1:-1] + (1 - p["d_A"]) * QA[1:]) / QA[:-1] - 1))
    res["r = (1+g)^theta/B - 1"] = np.max(np.abs(1 / m - 1 - ss["r"]))
    # W/Z is constant only if Gamma/P_A is; evaluate the closed form date by date
    wZ = np.array([bgp(p, e, gam_G, gam_q, Gamma=Gam[i], PA=PA[i])["wZ"] for i in range(T)])
    res["wage W/Z closed form"] = np.max(np.abs(W / Z[:-1] / wZ - 1))
    res["labor share"] = np.max(np.abs(W * N / Ynet - ss["labsh"]))
    # market value: ex-dividend V_t = Q_K K_{t+1} + Q_A A_{t+1}; check vs PV of dividends
    D = Ynet - W * N - IK - PA[:-1] * IA
    V = QK * K[1:] + QA * A[1:]
    res["V/Y closed form"] = np.max(np.abs(V / Ynet - ss["VY"]))
    G, r = ss["G"], ss["r"]
    # Gordon on the BGP: V_t = D_{t+1}/(r-g) = D_t (1+g)/(r-g)
    res["V = PV(dividends) (Hayashi)"] = np.max(np.abs(V / (D * G / (r - (G - 1))) - 1))
    return {k: float(v) for k, v in res.items()}


# ----------------------------------------------------------------------------
# Perfect-foresight transitions (variables detrended by Z)
# ----------------------------------------------------------------------------
def scurve(T, mid=10.0, width=2.5):
    t = np.arange(T + 1)
    S = 1.0 / (1.0 + np.exp(-(t - mid) / width))
    S0 = S[0]
    return (S - S0) / (1.0 - S0)


class Transition:
    def __init__(self, p, e, Gam, PA, T):
        self.p, self.e, self.T = p, e, T
        self.Gam, self.PA = Gam, PA                    # length T+1
        self.ss0 = bgp(p, e, Gamma=Gam[0], PA=PA[0])
        self.ssT = bgp(p, e, Gamma=Gam[-1], PA=PA[-1])
        GZ = 1 + p["g_Z"]
        self.xK, self.xA = GZ - 1 + p["d_K"], GZ - 1 + p["d_A"]
        # initial detrended states from the initial BGP
        self.k0, self.a0 = self.states(self.ss0, Gam[0], PA[0])

    def states(self, ss, Gam, PA):
        p, e = self.p, self.e
        y = 1.0
        for _ in range(500):
            k, A = ss["KY"] * y, ss["AY"] * y / PA
            X = k ** p["a"] * ss["N"] ** (1 - p["a"])
            y = prod(X, Gam * A, e, p)[0]
        return ss["KY"] * y, ss["AY"] * y / PA

    def unpack(self, z):
        T = self.T
        return z[:T], z[T:2 * T], z[2 * T:]

    def paths(self, z):
        p, e, T = self.p, self.e, self.T
        jK, jA, N = self.unpack(z)
        ssT = self.ssT
        jK = np.append(jK, self.xK)
        jA = np.append(jA, self.xA)
        N = np.append(N, ssT["N"])
        GZ = 1 + p["g_Z"]
        k = np.empty(T + 1); A = np.empty(T + 1)
        k[0], A[0] = self.k0, self.a0
        for t in range(T):
            k[t + 1] = (1 - p["d_K"] + jK[t]) * k[t] / GZ
            A[t + 1] = (1 - p["d_A"] + jA[t]) * A[t] / GZ
        a = p["a"]
        X = k ** a * N ** (1 - a)
        F, FX, FM = prod(X, self.Gam * A, e, p)
        MPK, MPN, MPA = FX * a * X / k, FX * (1 - a) * X / N, FM * self.Gam
        CK = 0.5 * p["c_K"] * (jK - self.xK) ** 2
        CA = 0.5 * p["c_A"] * (jA - self.xA) ** 2
        y = F - CK * k - self.PA * CA * A
        c = y - jK * k - self.PA * jA * A
        QK = 1 + p["c_K"] * (jK - self.xK)
        QA = self.PA * (1 + p["c_A"] * (jA - self.xA))
        b, s = p["b"], p["s"]
        mu = c ** (b * (1 - s) - 1) * (1 - N) ** ((1 - b) * (1 - s))
        m = p["B"] * GZ ** (-p["theta"]) * mu[1:] / mu[:-1]
        return dict(jK=jK, jA=jA, N=N, k=k, A=A, F=F, y=y, c=c, QK=QK, QA=QA, m=m,
                    MPK=MPK, MPN=MPN, MPA=MPA, CKt=CK, CAt=CA)

    def resid(self, z):
        p = self.p
        P = self.paths(z)
        if np.any(P["c"] <= 0) or np.any(P["N"] <= 0) or np.any(P["N"] >= 1):
            return np.full(3 * self.T, 1e3)
        jK, jA, QK, QA, m = P["jK"], P["jA"], P["QK"], P["QA"], P["m"]
        cK, cA = p["c_K"], p["c_A"]
        retK = P["MPK"][1:] + cK * (jK[1:] - self.xK) * jK[1:] - P["CKt"][1:] * 1.0 \
            + (1 - p["d_K"]) * QK[1:]
        retA = P["MPA"][1:] + self.PA[1:] * (cA * (jA[1:] - self.xA) * jA[1:] - P["CAt"][1:]) \
            + (1 - p["d_A"]) * QA[1:]
        eK = m * retK / QK[:-1] - 1
        eA = m * retA / QA[:-1] - 1
        b = p["b"]
        eN = (1 - b) * P["c"][:-1] / (b * (1 - P["N"][:-1])) / P["MPN"][:-1] - 1
        return np.concatenate([eK, eA, eN])

    def guess(self):
        T, ss = self.T, self.ss0
        return np.concatenate([np.full(T, self.xK), np.full(T, self.xA), np.full(T, ss["N"])])

    def solve(self, tol=1e-11, maxit=60):
        z = self.guess()
        R = self.resid(z)
        n = z.size
        for it in range(maxit):
            nr = np.max(np.abs(R))
            if nr < tol:
                break
            J = np.empty((n, n))
            h = 1e-7
            for i in range(n):
                dz = np.zeros(n); dz[i] = h
                J[:, i] = (self.resid(z + dz) - R) / h
            step = np.linalg.solve(J, -R)
            lam = 1.0
            while lam > 1e-4:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, R = zn, Rn
        self.z, self.iters, self.err = z, it, float(np.max(np.abs(R)))
        return self

    def report(self):
        p = self.p
        P = self.paths(self.z)
        GZ = 1 + p["g_Z"]
        r = 1 / P["m"] - 1                                          # r_{t+1}
        w = P["MPN"][:-1]
        y = P["y"][:-1]
        labsh = w * P["N"][:-1] / y
        V = GZ * (P["QK"][:-1] * P["k"][1:] + P["QA"][:-1] * P["A"][1:])
        sM = P["MPA"][:-1] * P["A"][:-1] / P["F"][:-1]
        aiinv = self.PA[:-1] * P["jA"][:-1] * P["A"][:-1] / y
        qA = P["QA"][:-1] / self.PA[:-1]
        return dict(r=r, w=w, labsh=labsh, VY=V / y, sM=sM, y=y, N=P["N"][:-1],
                    qK=P["QK"][:-1], qA=qA, aiinv=aiinv, c=P["c"][:-1])


def run_transitions(p, T=150):
    S = scurve(T)
    out = {}
    for e in (0.5, 1.0, 1.5):
        for kind in ("disembodied", "embodied"):
            if kind == "disembodied":
                Gam = p["Gamma0"] * np.exp(DELTA * S)
                PA = np.full(T + 1, p["PA0"])
            else:
                Gam = np.full(T + 1, p["Gamma0"])
                PA = p["PA0"] * np.exp(-DELTA * S)
            tr = Transition(p, e, Gam, PA, T).solve()
            out[(e, kind)] = tr
            print(f"  e={e:3.1f} {kind:12s} newton iters={tr.iters:2d} max resid={tr.err:.1e}")
    return out


# ----------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------
COL = {"disembodied": "#1f5f9e", "embodied": "#c0512a"}
LAB = {0.5: r"$\varepsilon=0.5$ (complements)", 1.0: r"$\varepsilon=1$ (Cobb-Douglas)",
       1.5: r"$\varepsilon=1.5$ (substitutes)"}


def figures(p, trs, H=40):
    t = np.arange(H)
    specs = [
        ("fig_r", "Real interest rate $r_{t+1}$ (percent)",
         lambda R, ss: 100 * R["r"][:H], lambda ss: 100 * ss["r"]),
        ("fig_w", "Real wage $W/Z$ (percent dev. from initial BGP)",
         lambda R, ss: 100 * (R["w"][:H] / ss["wZ"] - 1), None),
        ("fig_labsh", "Labor share $WN/Y$ (percent)",
         lambda R, ss: 100 * R["labsh"][:H], lambda ss: 100 * ss["labsh"]),
        ("fig_value", "Market value / output $V/Y$",
         lambda R, ss: R["VY"][:H], lambda ss: ss["VY"]),
    ]
    for name, title, fn, fT in specs:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
        for ax, e in zip(axes, (0.5, 1.0, 1.5)):
            for kind in ("disembodied", "embodied"):
                tr = trs[(e, kind)]
                R = tr.report()
                ax.plot(t, fn(R, tr.ss0), color=COL[kind], lw=2, label=kind)
            if fT is not None:
                ax.axhline(fT(tr.ssT), color="0.5", ls=":", lw=1, label="new BGP")
            else:
                ax.axhline(100 * (tr.ssT["wZ"] / tr.ss0["wZ"] - 1), color="0.5", ls=":",
                           lw=1, label="new BGP")
            ax.set_title(LAB[e], fontsize=10)
            ax.set_xlabel("years after announcement")
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=150)
        plt.close(fig)


def tex_tables(p, checks, trs):
    lines = []
    # Table A: CD BGP, permanent progress
    rows = [("No AI progress", 0.0, 0.0), (r"Disembodied, $\gamma_\Gamma=0.10$", 0.10, 0.0),
            (r"Embodied, $\gamma_q=0.10$", 0.0, 0.10)]
    lines.append(r"\begin{tabular}{lcccccccc}\toprule")
    lines.append(r" & $g$ & $r$ & $u_A$ & $W/Z$ & $WN/Y$ & $K/Y$ & $P_AA/Y$ & $V/Y$\\\midrule")
    w_ref = bgp(p, 1.0)["wZ"]
    for lab, gG, gq in rows:
        ss = bgp(p, 1.0, gG, gq)
        lines.append(f"{lab} & {100*ss['g']:.2f} & {100*ss['r']:.2f} & {100*ss['u_A']:.1f} & "
                     f"{ss['wZ']/w_ref:.3f} & {ss['labsh']:.3f} & {ss['KY']:.2f} & "
                     f"{ss['AY']:.3f} & {ss['VY']:.2f}\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_bgp.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Table B: level shift, initial vs final BGP, and transition peaks
    L = [r"\begin{tabular}{lccccccc}\toprule",
         r" & $s_M$ & $WN/Y$ & $W/Z$ & $V/Y$ & \multicolumn{2}{c}{peak $r$ (pct.)} & $W/Z$ at $t=10$\\",
         r" & new BGP & new BGP & new/old & new BGP & disemb. & emb. & disemb./emb.\\\midrule"]
    for e in (0.5, 1.0, 1.5):
        d, m = trs[(e, "disembodied")], trs[(e, "embodied")]
        ss0, ssT = d.ss0, d.ssT
        Rd, Rm = d.report(), m.report()
        L.append(f"{LAB[e].split(' (')[0]} & {ssT['sM']:.3f} & {ssT['labsh']:.3f} & "
                 f"{ssT['wZ']/ss0['wZ']:.3f} & {ssT['VY']:.2f} & {100*Rd['r'].max():.2f} & "
                 f"{100*Rm['r'].max():.2f} & {Rd['w'][10]/ss0['wZ']:.3f} / {Rm['w'][10]/ss0['wZ']:.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_shift.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    # Table C: verification residuals
    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k.replace('_', ' ')} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_checks.tex"), "w") as f:
        f.write("\n".join(C) + "\n")


def main():
    p = BASE
    print("Calibration:", {k: round(v, 4) for k, v in p.items()})
    ss0 = bgp(p, 1.0)
    print(f"Initial BGP: r={ss0['r']:.4f} K/Y={ss0['KY']:.3f} I_K/Y={ss0['IKY']:.3f} "
          f"N={ss0['N']:.3f} V/Y={ss0['VY']:.3f} (approx {ss0['VY_approx']:.3f})")

    checks = {}
    # 1. Exact CD BGP with permanent embodied + disembodied progress
    for gG, gq in ((0.0, 0.0), (0.06, 0.0), (0.0, 0.06), (0.04, 0.05)):
        for k, v in levels_check(p, 1.0, gG, gq).items():
            key = f"CD BGP: {k}"
            checks[key] = max(checks.get(key, 0.0), v)
    # 2. General-e steady states (constant Gamma/P_A)
    for e in (0.5, 1.5):
        for scale in (1.0, 20.0):
            for k, v in levels_check(dict(p, Gamma0=p["Gamma0"] * scale), e, 0.0, 0.0).items():
                key = f"CES steady state: {k}"
                checks[key] = max(checks.get(key, 0.0), v)
    # 3. Growth formula is exact: compare log growth of F with closed form
    # (contained in "F = Y" check, since Y is built from the closed-form G)
    # 4. Baumol limit: W/Z as lambda -> 0 with e < 1
    e = 0.5
    big = bgp(p, e, Gamma=p["Gamma0"] * 1e12)
    checks["Baumol limit W/Z (e=0.5)"] = abs(big["wZ"] / baumol_limit_wZ(p, e) - 1)
    # 5. AK dichotomy: interior BGP fails exactly when AK growth exceeds g_Z
    e = 1.5
    thr = ak_threshold(p, e)
    lo, hi = bgp(p, e, Gamma=0.999 * thr), bgp(p, e, Gamma=1.001 * thr)
    ak_lo, ak_hi = ak_limit(p, e, 0.999 * thr), ak_limit(p, e, 1.001 * thr)
    ok = (lo is not None) and (hi is None) and (ak_lo["g"] < p["g_Z"] < ak_hi["g"])
    checks["AK threshold consistent (e=1.5)"] = 0.0 if ok else 1.0
    checks["AK threshold: g_AK - g_Z at threshold"] = abs(ak_limit(p, e, thr)["g"] - p["g_Z"])

    print("Verification (max abs error):")
    for k, v in checks.items():
        print(f"  {k:55s} {v:.2e}")

    print("Transitions:")
    trs = run_transitions(p)
    for (e, kind), tr in trs.items():
        R = tr.report()
        end = abs(R["w"][-1] / tr.ssT["wZ"] - 1)
        checks[f"transition end W/Z vs closed form, e={e}, {kind}"] = end
    checks["transition Newton residual (max)"] = max(tr.err for tr in trs.values())

    figures(p, trs)
    tex_tables(p, checks, trs)

    summary = dict(
        calibration=p, delta=float(DELTA), initial_bgp=ss0,
        cd_bgp={lab: bgp(p, 1.0, gG, gq) for lab, gG, gq in
                (("none", 0, 0), ("disembodied_0.10", 0.10, 0), ("embodied_0.10", 0, 0.10))},
        ak_threshold_e15=thr,
        ak_example_e15=ak_limit(p, 1.5, 2 * thr),
        baumol_wage_gain_e05=baumol_limit_wZ(p, 0.5) / ss0["wZ"],
        checks=checks,
    )
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for (e, kind), tr in trs.items():
        R = tr.report()
        print(f"  e={e} {kind:12s} r: init {100*tr.ss0['r']:.2f} peak {100*R['r'].max():.2f} "
              f"final {100*tr.ssT['r']:.2f} | W/Z: t0 {R['w'][0]/tr.ss0['wZ']:.3f} "
              f"t10 {R['w'][10]/tr.ss0['wZ']:.3f} t30 {R['w'][30]/tr.ss0['wZ']:.3f} "
              f"BGP {tr.ssT['wZ']/tr.ss0['wZ']:.3f} | sM final {tr.ssT['sM']:.3f}")


if __name__ == "__main__":
    main()
