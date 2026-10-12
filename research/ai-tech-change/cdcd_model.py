"""
Cobb-Douglas production with Cobb-Douglas capital accumulation: exact closed forms.

Pairs with ai_model.py.  There, CES production is paired with the quadratic
adjustment cost 0.5 c (I/K - x)^2 K of Kiley (2000), which delivers closed forms on
the balanced growth path only.  Here the Cobb-Douglas production function is paired
with a Cobb-Douglas accumulation technology (Lucas-Prescott 1971; Hercowitz-Sampson
1991):

    K(t+1) = Abar_K K(t)^(1-delta_K) I_K(t)^delta_K
    A(t+1) = Abar_A A(t)^(1-delta_A) (I_A(t)/P_A(t))^delta_A     (embodied: P_A falls)
    Y      = F(K, Z N, Gamma A) = K^aK (Z N)^aN (Gamma A)^aA,   aK + aN + aA = 1

With the paper's log preferences b ln C + (1-b) ln(1-N), the equilibrium is exact
for any path of (Z, Gamma, P_A), anticipated or not:

    I_j / Y = s_j = beta delta_j alpha_j / (1 - beta (1 - delta_j))
    N       = b aN / (b aN + (1-b)(1 - s_K - s_A))
    Q_j     = I_j / (delta_j j(t+1))            (price of installed capital)
    V / Y   = sum_j s_j / delta_j = sum_j alpha_j / (rho + delta_j),   beta = 1/(1+rho)
    1 + r(t+1) = (1+rho) Y(t+1)/Y(t)
    W       = aN Y / N

so the paths of r and W follow from the log-linear law of motion for (ln K, ln A).

The script
  1. simulates the closed form,
  2. checks it against a generic Newton solve of the nonlinear perfect-foresight
     equilibrium (which does not impose constant saving rates),
  3. shows that with CES production (e != 1) saving rates are no longer constant,
  4. compares the exact CD-CD paths with the quadratic-adjustment-cost model when
     the two have the same local Q-elasticity of investment,
  5. writes figures and LaTeX tables used in ai_tech_change.tex.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ai_model as am

HERE = am.HERE
FIG = am.FIG


# ----------------------------------------------------------------------------
# Calibration: same initial BGP as ai_model.BASE
# ----------------------------------------------------------------------------
def cd_params(p=am.BASE):
    """Map (d_K, d_A, g_Z) into CD-accumulation parameters.

    delta_j = (g + d_j)/(1 + g) and Abar_j = G^(1-delta)/delta^delta make the CD
    accumulation function agree to first order with K' = (1-d)K + I on the initial
    BGP (same level, slope in I, slope in K), so Q = 1 there and the initial BGP is
    identical to ai_model's.
    """
    G = 1.0 + p["g_Z"]
    q = dict(p)
    for j in ("K", "A"):
        dl = (p["g_Z"] + p[f"d_{j}"]) / G
        q[f"delta_{j}"] = dl
        q[f"Abar_{j}"] = G ** (1 - dl) / dl ** dl
    om, a = p["omega"], p["a"]
    q["alpha_K"], q["alpha_N"], q["alpha_A"] = a * om, (1 - a) * om, 1 - om
    q["beta"] = p["B"]
    q["rho"] = 1 / p["B"] - 1
    # local Q-elasticity of I/K is 1/(1-delta); quadratic cost gives 1/(c x)
    q["c_K_match"] = (1 - q["delta_K"]) / (p["g_Z"] + p["d_K"])
    q["c_A_match"] = (1 - q["delta_A"]) / (p["g_Z"] + p["d_A"])
    return q


def closed_form_rates(q):
    be = q["beta"]
    s = {j: be * q[f"delta_{j}"] * q[f"alpha_{j}"] / (1 - be * (1 - q[f"delta_{j}"]))
         for j in ("K", "A")}
    CY = 1 - s["K"] - s["A"]
    b = q["b"]
    N = b * q["alpha_N"] / (b * q["alpha_N"] + (1 - b) * CY)
    VY = s["K"] / q["delta_K"] + s["A"] / q["delta_A"]
    return dict(sK=s["K"], sA=s["A"], N=N, VY=VY, CY=CY,
                VY_rho=sum(q[f"alpha_{j}"] / (q["rho"] + q[f"delta_{j}"]) for j in ("K", "A")))


def transition_matrix(q):
    dK, dA, aK, aA = q["delta_K"], q["delta_A"], q["alpha_K"], q["alpha_A"]
    M = np.array([[1 - dK + dK * aK, dK * aA], [dA * aK, 1 - dA + dA * aA]])
    return M, np.sort(np.linalg.eigvals(M).real)


# ----------------------------------------------------------------------------
# Closed-form simulation in levels
# ----------------------------------------------------------------------------
def initial_states(q, e, Gam0, PA0):
    ss = am.bgp(q, e, Gamma=Gam0, PA=PA0)
    K, A = ss["KY"], ss["AY"] / PA0          # with Z_0 = 1, scale Y_0 below
    y = 1.0
    for _ in range(500):
        K, A = ss["KY"] * y, ss["AY"] * y / PA0
        y = am.prod(K ** q["a"] * ss["N"] ** (1 - q["a"]), Gam0 * A, e, q)[0]
    return K, A, ss


def simulate_closed_form(q, Z, Gam, PA):
    cf = closed_form_rates(q)
    T = len(Z) - 1
    K0, A0, _ = initial_states(q, 1.0, Gam[0], PA[0])
    K, A, Y = np.empty(T + 1), np.empty(T + 1), np.empty(T + 1)
    K[0], A[0] = K0, A0
    for t in range(T + 1):
        X = K[t] ** q["a"] * (Z[t] * cf["N"]) ** (1 - q["a"])
        Y[t] = am.prod(X, Gam[t] * A[t], 1.0, q)[0]
        if t < T:
            K[t + 1] = q["Abar_K"] * K[t] ** (1 - q["delta_K"]) * (cf["sK"] * Y[t]) ** q["delta_K"]
            A[t + 1] = q["Abar_A"] * A[t] ** (1 - q["delta_A"]) * \
                (cf["sA"] * Y[t] / PA[t]) ** q["delta_A"]
    r = (1 + q["rho"]) * Y[1:] / Y[:-1] - 1                  # r_{t+1}
    W = q["alpha_N"] * Y / cf["N"]
    return dict(K=K, A=A, Y=Y, r=r, W=W, N=cf["N"], sK=cf["sK"], sA=cf["sA"], VY=cf["VY"])


# ----------------------------------------------------------------------------
# Generic nonlinear perfect-foresight solver with CD accumulation (any e)
# ----------------------------------------------------------------------------
class CDAccum:
    """Unknowns: s_K(t) = I_K/Y, s_A(t) = I_A/Y, N(t), t = 0..T-1 (levels, no detrending).

    Equilibrium conditions imposed (nothing assumes constant saving rates):
      Q_K(t) = I_K(t)/(delta_K K(t+1)),   Q_A(t) = I_A(t)/(delta_A A(t+1))
      Q_K(t) = m(t+1)[F_K(t+1) + (1-delta_K) Q_K(t+1) K(t+2)/K(t+1)]
      Q_A(t) = m(t+1)[Gamma F_M(t+1) + (1-delta_A) Q_A(t+1) A(t+2)/A(t+1)]
      (1-b) C/(b(1-N)) = F_N,   m(t+1) = B U_C(t+1)/U_C(t),   C = Y - I_K - I_A
    Terminal values at T: final-BGP saving rates and hours.
    """

    def __init__(self, q, e, Z, Gam, PA, terminal):
        self.q, self.e, self.Z, self.Gam, self.PA = q, e, Z, Gam, PA
        self.T = len(Z) - 1
        self.K0, self.A0, self.ss0 = initial_states(q, e, Gam[0], PA[0])
        self.term = terminal          # (sK, sA, N) at T

    def paths(self, z):
        q, e, T = self.q, self.e, self.T
        sK = np.append(z[:T], self.term[0])
        sA = np.append(z[T:2 * T], self.term[1])
        N = np.append(z[2 * T:], self.term[2])
        K, A, Y = np.empty(T + 2), np.empty(T + 2), np.empty(T + 1)
        FX, FM = np.empty(T + 1), np.empty(T + 1)
        K[0], A[0] = self.K0, self.A0
        a = q["a"]
        for t in range(T + 1):
            X = K[t] ** a * (self.Z[t] * N[t]) ** (1 - a)
            Y[t], FX[t], FM[t] = am.prod(X, self.Gam[t] * A[t], e, q)
            K[t + 1] = q["Abar_K"] * K[t] ** (1 - q["delta_K"]) * (sK[t] * Y[t]) ** q["delta_K"]
            A[t + 1] = q["Abar_A"] * A[t] ** (1 - q["delta_A"]) * \
                (sA[t] * Y[t] / self.PA[t]) ** q["delta_A"]
        X = K[:-1] ** a * (self.Z * N) ** (1 - a)
        MPK = FX * a * X / K[:-1]
        MPN = FX * (1 - a) * X / N
        MPA = FM * self.Gam
        C = (1 - sK - sA) * Y
        QK = sK * Y / (q["delta_K"] * K[1:])
        QA = sA * Y / (q["delta_A"] * A[1:])
        b, s = q["b"], q["s"]
        mu = C ** (b * (1 - s) - 1) * (1 - N) ** ((1 - b) * (1 - s))
        m = q["B"] * mu[1:] / mu[:-1]
        return dict(sK=sK, sA=sA, N=N, K=K, A=A, Y=Y, C=C, QK=QK, QA=QA, m=m,
                    MPK=MPK, MPN=MPN, MPA=MPA)

    def resid(self, z):
        q, T = self.q, self.T
        if np.any(z[:2 * T] <= 0):
            return np.full(3 * T, 1e3)
        P = self.paths(z)
        if np.any(P["C"] <= 0) or np.any(P["N"] <= 0) or np.any(P["N"] >= 1):
            return np.full(3 * T, 1e3)
        K, A, QK, QA, m = P["K"], P["A"], P["QK"], P["QA"], P["m"]
        eK = m[:T] * (P["MPK"][1:] + (1 - q["delta_K"]) * QK[1:] * K[2:] / K[1:-1]) / QK[:T] - 1
        eA = m[:T] * (P["MPA"][1:] + (1 - q["delta_A"]) * QA[1:] * A[2:] / A[1:-1]) / QA[:T] - 1
        b = q["b"]
        eN = (1 - b) * P["C"][:T] / (b * (1 - P["N"][:T])) / P["MPN"][:T] - 1
        return np.concatenate([eK, eA, eN])

    def solve(self, z0, tol=1e-11, maxit=60):
        z, R = z0.copy(), self.resid(z0)
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
            while lam > 1e-4:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, R = zn, Rn
        self.z, self.iters, self.err = z, it, float(np.max(np.abs(R)))
        return self


def bgp_rates(q, e, Gam, PA):
    """Saving rates and hours on the BGP for constant Gamma/P_A (Prop. 1 + CD accumulation)."""
    ss = am.bgp(q, e, Gamma=Gam, PA=PA)
    G = 1 + q["g_Z"]
    return ss, ((G - 1 + q["d_K"]) * ss["KY"], (G - 1 + q["d_A"]) * ss["AY"], ss["N"])


# ----------------------------------------------------------------------------
def shock_paths(q, kind, T, delta=am.DELTA):
    t = np.arange(T + 1)
    Z = (1 + q["g_Z"]) ** t
    S = am.scurve(T)
    if kind == "disembodied":
        return Z, q["Gamma0"] * np.exp(delta * S), np.full(T + 1, q["PA0"])
    return Z, np.full(T + 1, q["Gamma0"]), q["PA0"] * np.exp(-delta * S)


def main():
    q = cd_params()
    cf = closed_form_rates(q)
    M, eig = transition_matrix(q)
    ss0 = am.bgp(q, 1.0)
    print(f"delta_K={q['delta_K']:.4f} delta_A={q['delta_A']:.4f} "
          f"matched c_K={q['c_K_match']:.2f} c_A={q['c_A_match']:.2f}")
    print(f"s_K={cf['sK']:.4f} s_A={cf['sA']:.4f} N={cf['N']:.4f} V/Y={cf['VY']:.4f} "
          f"(= sum alpha/(rho+delta) {cf['VY_rho']:.4f}); eigenvalues {eig}")
    checks = {}
    # The CD-CD exact solution coincides with ai_model's initial BGP
    checks["CD-CD saving rates = quadratic-AC BGP"] = max(
        abs(cf["sK"] - ss0["IKY"]), abs(cf["sA"] - ss0["IAY"]))
    checks["CD-CD hours = quadratic-AC BGP"] = abs(cf["N"] - ss0["N"])
    checks["V/Y: sum s/delta = sum alpha/(rho+delta)"] = abs(cf["VY"] - cf["VY_rho"])

    T = 120
    H = 40
    sims, quad = {}, {}
    # quadratic-adjustment-cost model with the same local Q-elasticities
    qq = dict(am.BASE, c_K=q["c_K_match"], c_A=q["c_A_match"])
    for kind in ("disembodied", "embodied"):
        Z, Gam, PA = shock_paths(q, kind, T)
        cfs = simulate_closed_form(q, Z, Gam, PA)
        sims[kind] = cfs
        # generic nonlinear solve, started away from the closed form
        _, term = bgp_rates(q, 1.0, Gam[-1], PA[-1])
        solver = CDAccum(q, 1.0, Z, Gam, PA, term)
        z0 = np.concatenate([np.full(T, 0.9 * term[0]), np.full(T, 1.1 * term[1]),
                             np.full(T, term[2])])
        solver.solve(z0)
        P = solver.paths(solver.z)
        print(f"  CD-CD {kind:12s} generic Newton iters={solver.iters} resid={solver.err:.1e}")
        checks[f"CD-CD {kind}: generic solve s_K, s_A constant"] = float(max(
            np.max(np.abs(P["sK"][:T] - cf["sK"])), np.max(np.abs(P["sA"][:T] - cf["sA"]))))
        checks[f"CD-CD {kind}: generic solve hours"] = float(np.max(np.abs(P["N"][:T] - cf["N"])))
        rgen = 1 / P["m"] - 1
        checks[f"CD-CD {kind}: r path = (1+rho)Y'/Y - 1"] = float(np.max(np.abs(rgen[:T] - cfs["r"][:T])))
        checks[f"CD-CD {kind}: V/Y constant"] = float(np.max(np.abs(
            (P["QK"][:T] * P["K"][1:T + 1] + P["QA"][:T] * P["A"][1:T + 1]) / P["Y"][:T] - cf["VY"])))
        # long run: ln Y gain = (alpha_A/alpha_N) * Delta relative to no-shock baseline
        Zb, Gb, PAb = shock_paths(q, kind, T, delta=0.0)
        base = simulate_closed_form(q, Zb, Gb, PAb)
        gain = np.log(cfs["Y"][-1] / base["Y"][-1])
        checks[f"CD-CD {kind}: ln Y gain at T=120 vs (aA/aN) Delta"] = abs(
            gain - q["alpha_A"] / q["alpha_N"] * am.DELTA)
        cum_r = np.sum(np.log(1 + cfs["r"]) - np.log(1 + base["r"]))
        checks[f"CD-CD {kind}: sum of ln(1+r) deviations = ln Y gain"] = abs(cum_r - gain)
        sims[kind]["base"] = base
        # quadratic model with matched curvature (ai_model, e = 1)
        Gt, PAt = Gam[:T + 1], PA[:T + 1]
        tr = am.Transition(qq, 1.0, Gt, PAt, T).solve()
        quad[kind] = tr
        print(f"  quadratic-matched {kind:12s} iters={tr.iters} resid={tr.err:.1e}")

    # Permanent progress (exact BGP with growth) in the CD-CD economy, generic solve
    for gG, gq in ((0.03, 0.0), (0.0, 0.03)):
        t = np.arange(T + 1)
        Z = (1 + q["g_Z"]) ** t
        Gam = q["Gamma0"] * (1 + gG) ** t
        PA = q["PA0"] * (1 + gq) ** (-t)
        cfs = simulate_closed_form(q, Z, Gam, PA)
        solver = CDAccum(q, 1.0, Z, Gam, PA, (cf["sK"], cf["sA"], cf["N"]))
        solver.solve(np.concatenate([np.full(T, cf["sK"]), np.full(T, cf["sA"]), np.full(T, cf["N"])]))
        g_cf = (1 + q["g_Z"]) * ((1 + gG) * (1 + gq)) ** (q["alpha_A"] / q["alpha_N"]) - 1
        g_sim = cfs["Y"][-1] / cfs["Y"][-2] - 1
        lab = "disembodied" if gG else "embodied"
        checks[f"CD-CD permanent {lab} progress: Newton residual at closed form"] = solver.err
        checks[f"CD-CD permanent {lab} progress: growth at T=120 vs closed form"] = abs(g_sim - g_cf)

    # CES production + CD accumulation: no longer exact (saving rates move)
    ces_var = {}
    for e in (0.5, 1.5):
        kind = "disembodied"
        Z, Gam, PA = shock_paths(q, kind, T)
        _, term = bgp_rates(q, e, Gam[-1], PA[-1])
        _, init = bgp_rates(q, e, Gam[0], PA[0])
        solver = CDAccum(q, e, Z, Gam, PA, term)
        solver.solve(np.concatenate([np.full(T, init[0]), np.full(T, init[1]), np.full(T, init[2])]))
        P = solver.paths(solver.z)
        ces_var[e] = dict(sA_min=float(P["sA"][:T].min()), sA_max=float(P["sA"][:T].max()),
                          sK_min=float(P["sK"][:T].min()), sK_max=float(P["sK"][:T].max()),
                          resid=solver.err)
        print(f"  CES e={e} + CD accumulation: s_A in [{ces_var[e]['sA_min']:.4f}, "
              f"{ces_var[e]['sA_max']:.4f}], s_K in [{ces_var[e]['sK_min']:.4f}, "
              f"{ces_var[e]['sK_max']:.4f}] (resid {solver.err:.1e})")

    # Step (surprise, permanent) shocks: analytic r response vs simulation
    al = np.array([q["alpha_K"], q["alpha_A"]])
    Dl = am.DELTA
    f = {"disembodied": q["alpha_A"] * Dl * np.array([q["delta_K"], q["delta_A"]]),
         "embodied": Dl * np.array([0.0, q["delta_A"]])}
    Ts = 80
    step = {}
    for kind in ("disembodied", "embodied"):
        t = np.arange(Ts + 1)
        Z = (1 + q["g_Z"]) ** t
        Gam = np.full(Ts + 1, q["Gamma0"]); PA = np.full(Ts + 1, q["PA0"])
        if kind == "disembodied":
            Gam = Gam * np.exp(Dl)
        else:
            PA = PA * np.exp(-Dl)
        # start both from the pre-shock BGP states
        cf0 = closed_form_rates(q)
        K0, A0, _ = initial_states(q, 1.0, q["Gamma0"], q["PA0"])
        K, A, Y = np.empty(Ts + 1), np.empty(Ts + 1), np.empty(Ts + 1)
        K[0], A[0] = K0, A0
        for i in range(Ts + 1):
            X = K[i] ** q["a"] * (Z[i] * cf0["N"]) ** (1 - q["a"])
            Y[i] = am.prod(X, Gam[i] * A[i], 1.0, q)[0]
            if i < Ts:
                K[i + 1] = q["Abar_K"] * K[i] ** (1 - q["delta_K"]) * (cf0["sK"] * Y[i]) ** q["delta_K"]
                A[i + 1] = q["Abar_A"] * A[i] ** (1 - q["delta_A"]) * (cf0["sA"] * Y[i] / PA[i]) ** q["delta_A"]
        lnr = np.log((1 + q["rho"]) * Y[1:] / Y[:-1]) - np.log((1 + q["rho"]) * (1 + q["g_Z"]))
        analytic = np.array([al @ np.linalg.matrix_power(M, i) @ f[kind] for i in range(Ts)])
        checks[f"step {kind}: ln(1+r) dev = alpha' M^t f"] = float(np.max(np.abs(lnr - analytic)))
        step[kind] = dict(r=np.exp(lnr + np.log((1 + q["rho"]) * (1 + q["g_Z"]))) - 1,
                          lnY=np.log(Y) - np.log(Y[0]) - t * np.log(1 + q["g_Z"]))
    print("Checks:")
    for k, v in checks.items():
        print(f"  {k:62s} {v:.2e}")

    # ---------------- figures ----------------
    t = np.arange(H)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for kind in ("disembodied", "embodied"):
        S, tr = sims[kind], quad[kind]
        R = tr.report()
        axes[0].plot(t, 100 * S["r"][:H], color=am.COL[kind], lw=2, label=f"{kind}: CD-CD exact")
        axes[0].plot(t, 100 * R["r"][:H], color=am.COL[kind], lw=1.2, ls="--",
                     label=f"{kind}: quadratic, matched $c$")
        wdev = S["W"] / S["base"]["W"] - 1
        axes[1].plot(t, 100 * wdev[:H], color=am.COL[kind], lw=2)
        axes[1].plot(t, 100 * (R["w"][:H] / tr.ss0["wZ"] - 1), color=am.COL[kind], lw=1.2, ls="--")
    axes[0].axhline(100 * am.BASE["r0"], color="0.5", ls=":", lw=1)
    axes[1].axhline(100 * np.expm1(q["alpha_A"] / q["alpha_N"] * am.DELTA), color="0.5", ls=":", lw=1)
    ts = np.arange(30)
    for kind in ("disembodied", "embodied"):
        axes[2].plot(ts, 100 * step[kind]["r"][:30], color=am.COL[kind], lw=2, label=kind)
    axes[2].axhline(100 * am.BASE["r0"], color="0.5", ls=":", lw=1)
    axes[2].set_title("$r_{t+1}$ after a surprise step (percent)", fontsize=10)
    axes[2].legend(fontsize=7, frameon=False)
    axes[0].set_title("S-curve: real interest rate $r_{t+1}$ (percent)", fontsize=10)
    axes[1].set_title("S-curve: real wage (pct. dev. from no-AI path)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("years after announcement"); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_cdcd.{ext}"), dpi=150)
    plt.close(fig)

    # ---------------- single-capital table: drop in required return ----------------
    # Kiley (2000) section 3 experiment: r 7.5% -> 4%, g = 0, d = 0.066
    a, d = 0.282, 0.066
    rows = []
    R0, R1 = 0.075 + d, 0.04 + d
    for e in (0.5, 1.0, 1.5):
        # normalize the CES weight so the capital share is a = 0.282 at r = 7.5% for every e
        alpha_e = a * R0 ** (e - 1)
        rows.append((e, (R0 / R1) ** e - 1, alpha_e * R0 ** (1 - e), alpha_e * R1 ** (1 - e)))
    gordon = 0.075 / 0.04 - 1
    L = [r"\begin{tabular}{lccc}\toprule",
         r" & \% change in $V/Y$ & capital share, $r=7.5\%$ & capital share, $r=4\%$\\\midrule"]
    for e, dv, s0, s1 in rows:
        L.append(f"CES, $\\varepsilon={e:g}$ & {100*dv:.1f} & {s0:.3f} & {s1:.3f}\\\\")
    L.append(f"Gordon model (dividends fixed) & {100*gordon:.1f} & & \\\\")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_ces_r.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ---------------- CD-CD table ----------------
    Rd, Re = sims["disembodied"], sims["embodied"]
    qd, qe = quad["disembodied"].report(), quad["embodied"].report()
    L = [r"\begin{tabular}{lcccc}\toprule",
         r" & \multicolumn{2}{c}{CD-CD (exact)} & \multicolumn{2}{c}{quadratic, matched $c$}\\",
         r" & disembodied & embodied & disembodied & embodied\\\midrule"]
    def peak(x): return 100 * x.max()
    def tpk(x): return int(np.argmax(x))
    L.append(f"peak $r$ (pct.) & {peak(Rd['r']):.2f} & {peak(Re['r']):.2f} & "
             f"{peak(qd['r']):.2f} & {peak(qe['r']):.2f}\\\\")
    L.append(f"year of peak & {tpk(Rd['r'])} & {tpk(Re['r'])} & {tpk(qd['r'])} & {tpk(qe['r'])}\\\\")
    wD = Rd["W"] / Rd["base"]["W"] - 1
    wE = Re["W"] / Re["base"]["W"] - 1
    L.append(f"wage gain at $t=10$ (pct.) & {100*wD[10]:.1f} & {100*wE[10]:.1f} & "
             f"{100*(qd['w'][10]/quad['disembodied'].ss0['wZ']-1):.1f} & "
             f"{100*(qe['w'][10]/quad['embodied'].ss0['wZ']-1):.1f}\\\\")
    L.append(f"wage gain at $t=30$ (pct.) & {100*wD[30]:.1f} & {100*wE[30]:.1f} & "
             f"{100*(qd['w'][30]/quad['disembodied'].ss0['wZ']-1):.1f} & "
             f"{100*(qe['w'][30]/quad['embodied'].ss0['wZ']-1):.1f}\\\\")
    L.append(f"$V/Y$ range & \\multicolumn{{2}}{{c}}{{{cf['VY']:.3f} (constant)}} & "
             f"{qd['VY'][:H].min():.3f}--{qd['VY'][:H].max():.3f} & "
             f"{qe['VY'][:H].min():.3f}--{qe['VY'][:H].max():.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_cdcd.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_checks_cdcd.tex"), "w") as f:
        f.write("\n".join(C) + "\n")

    out = dict(step_peak_r={k: float(v["r"].max()) for k, v in step.items()},
               step_r0={k: float(v["r"][0]) for k, v in step.items()}, params={k: v for k, v in q.items()}, closed_form=cf,
               transition_matrix=M.tolist(), eigenvalues=eig.tolist(),
               ces_with_cd_accumulation=ces_var, checks=checks,
               single_capital_r_drop={str(e): dv for e, dv, _, _ in rows}, gordon=gordon)
    with open(os.path.join(HERE, "results_cdcd.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    for kind in ("disembodied", "embodied"):
        S = sims[kind]
        print(f"  {kind}: peak r {100*S['r'].max():.2f} at t={np.argmax(S['r'])}, "
              f"quad peak {100*quad[kind].report()['r'].max():.2f}")


if __name__ == "__main__":
    main()
