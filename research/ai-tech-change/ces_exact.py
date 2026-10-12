"""
CES production with CES capital accumulation: exact solutions with constant saving rates.

Generalizes the Cobb-Douglas/Cobb-Douglas pairing in cdcd_model.py (the e -> 1 limit).

One capital stock (rho = (e-1)/e, labor ZN exogenous):
    Y^rho      = a_K K^rho + a_L (ZN)^rho
    K(t+1)^rho = Abar^rho [ (1-delta) K^rho + delta I^rho ]
    U          = sum_t beta^t C^rho / rho        (CRRA with IES = e)
Guess V(K) = (kappa/rho) K^rho.  The Bellman equation is solved exactly with
    I/Y = s = Lambda^e,   Lambda = beta Abar^rho delta a_K / (1 - beta Abar^rho (1-delta)),
for any path of Z, anticipated or not.  Cobb-Douglas/Cobb-Douglas with log utility is e = 1.

AI extension (three-input CES with common elasticity e):
    Y^rho      = a_K K^rho + a_L (ZN)^rho + a_A (Gamma A)^rho
    A(t+1)^rho = Abar_A^rho [ (1-delta_A) A^rho + delta_A (I_A/P_A)^rho ]
V_t = (kappa_Kt K^rho + kappa_At A^rho)/rho + h_t solves the Bellman equation exactly;
the saving rates s_Kt, s_At depend only on time (the technology path), never on the
state, and follow from a backward recursion:
    phi_jt   = beta kappa_j,t+1 Abar_j^rho delta_j  (x P_A,t^-rho for AI)
    s_jt     = phi_jt^e / (1 + phi_Kt^e + phi_At^e)
    Omega_t  = (1 + phi_Kt^e + phi_At^e)^(1/e)
    kappa_Kt = Omega_t a_K           + beta Abar_K^rho (1-delta_K) kappa_K,t+1
    kappa_At = Omega_t a_A Gamma_t^rho + beta Abar_A^rho (1-delta_A) kappa_A,t+1
Prices: 1 + r(t+1) = (C(t+1)/C(t))^(1/e) / beta, W = a_L Z (Y/ZN)^(1-rho),
Q_j = 1/(dK_j'/dI_j).

The script verifies every claim against a generic Newton solve of the nonlinear
perfect-foresight equilibrium, shows that exactness fails when the IES differs from e,
and computes embodied vs disembodied AI transitions for e in {0.5, 1.5}.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ai_model as am
import cdcd_model as cd

HERE, FIG = am.HERE, am.FIG


# ----------------------------------------------------------------------------
# Calibration: same initial BGP shares, r, g, depreciation as ai_model.BASE
# ----------------------------------------------------------------------------
def params(e, p=am.BASE, ies=None):
    """Three-input CES economy with CES accumulation, IES = e unless overridden."""
    rho = (e - 1) / e
    ies = e if ies is None else ies
    G = 1 + p["g_Z"]
    r0 = p["r0"]
    beta = G ** (1 / ies) / (1 + r0)          # 1 + r = G^(1/IES) / beta on the BGP
    shares = dict(K=p["a"] * p["omega"], A=1 - p["omega"], L=(1 - p["a"]) * p["omega"])
    R = dict(K=r0 + p["d_K"], A=r0 + p["d_A"])
    # weights a_j from share_j = a_j^e R_j^(1-e) at the initial BGP (Gamma = P_A = 1)
    a = {j: (shares[j] * R[j] ** (e - 1)) ** (1 / e) for j in ("K", "A")}
    a["L"] = shares["L"]                       # normalization: Y_0 = Z_0 N = 1
    q = dict(e=e, rho=rho, ies=ies, beta=beta, G=G, g_Z=p["g_Z"], r0=r0, shares=shares, R=R,
             a=a, d=dict(K=p["d_K"], A=p["d_A"]))
    # accumulation: first-order equal to K' = (1-d)K + I at I/K = g + d, Q = 1 there
    q["delta"], q["Abar_rho"] = {}, {}
    for j in ("K", "A"):
        x = p["g_Z"] + q["d"][j]
        q["delta"][j] = x ** (1 - rho) / (x ** (1 - rho) + 1 - q["d"][j])
        q["Abar_rho"][j] = G ** (rho - 1) * (x ** (1 - rho) + 1 - q["d"][j])
    return q


def output(q, K, L, M):
    rho = q["rho"]
    return (q["a"]["K"] * K ** rho + q["a"]["L"] * L ** rho + q["a"]["A"] * M ** rho) ** (1 / rho)


def initial_state(q):
    """Initial BGP levels with Y_0 = 1, Gamma = P_A = 1, Z_0 N = 1."""
    e = q["e"]
    KY = {j: (q["a"][j] / q["R"][j]) ** e for j in ("K", "A")}   # F_j = a_j (Y/X_j)^(1/e)
    return KY["K"], KY["A"]


# ----------------------------------------------------------------------------
# Closed form: backward recursion for saving rates, forward simulation
# ----------------------------------------------------------------------------
def kappa_step(q, kK, kA, Gam, PA):
    """One step of the recursion; returns (kappa_K,t, kappa_A,t, s_K,t, s_A,t)."""
    e, rho, b = q["e"], q["rho"], q["beta"]
    phiK = b * kK * q["Abar_rho"]["K"] * q["delta"]["K"]
    phiA = b * kA * q["Abar_rho"]["A"] * q["delta"]["A"] * PA ** (-rho)
    den = 1 + phiK ** e + phiA ** e
    Om = den ** (1 / e)
    newK = Om * q["a"]["K"] + b * q["Abar_rho"]["K"] * (1 - q["delta"]["K"]) * kK
    newA = Om * q["a"]["A"] * Gam ** rho + b * q["Abar_rho"]["A"] * (1 - q["delta"]["A"]) * kA
    return newK, newA, phiK ** e / den, phiA ** e / den


def steady_kappa(q, Gam, PA):
    kK = kA = 1.0
    for _ in range(20000):
        nK, nA, sK, sA = kappa_step(q, kK, kA, Gam, PA)
        if abs(nK - kK) + abs(nA - kA) < 1e-15:
            break
        kK, kA = nK, nA
    return nK, nA, sK, sA


def closed_form(q, Z, Gam, PA):
    T = len(Z) - 1
    kK, kA, _, _ = steady_kappa(q, Gam[-1], PA[-1])
    sK, sA = np.empty(T + 1), np.empty(T + 1)
    for t in range(T, -1, -1):
        kK, kA, sK[t], sA[t] = kappa_step(q, kK, kA, Gam[t], PA[t])
    K, A, Y = np.empty(T + 2), np.empty(T + 2), np.empty(T + 1)
    K[0], A[0] = initial_state(q)
    rho = q["rho"]
    for t in range(T + 1):
        Y[t] = output(q, K[t], Z[t], Gam[t] * A[t])
        K[t + 1] = (q["Abar_rho"]["K"] * ((1 - q["delta"]["K"]) * K[t] ** rho
                                          + q["delta"]["K"] * (sK[t] * Y[t]) ** rho)) ** (1 / rho)
        A[t + 1] = (q["Abar_rho"]["A"] * ((1 - q["delta"]["A"]) * A[t] ** rho
                                          + q["delta"]["A"] * (sA[t] * Y[t] / PA[t]) ** rho)) ** (1 / rho)
    return prices(q, dict(sK=sK, sA=sA, K=K, A=A, Y=Y), Z, Gam, PA)


def prices(q, P, Z, Gam, PA):
    e, rho = q["e"], q["rho"]
    Y, K, A, sK, sA = P["Y"], P["K"], P["A"], P["sK"], P["sA"]
    C = (1 - sK - sA) * Y
    r = (C[1:] / C[:-1]) ** (1 / q["ies"]) / q["beta"] - 1
    W = q["a"]["L"] * Z * (Y / Z) ** (1 - rho)                  # N normalized to 1
    labsh = W / Y
    IK, IA = sK * Y, sA * Y
    # Q_j = 1 / (dK'/dI)
    QK = IK ** (1 - rho) * K[1:] ** (rho - 1) / (q["Abar_rho"]["K"] * q["delta"]["K"])
    QA = PA ** rho * IA ** (1 - rho) * A[1:] ** (rho - 1) / (q["Abar_rho"]["A"] * q["delta"]["A"])
    V = QK * K[1:] + QA * A[1:]
    sM = q["a"]["A"] * (Gam * A[:-1] / Y) ** rho
    return dict(P, C=C, r=r, W=W, labsh=labsh, QK=QK, QA=QA, VY=V / Y, sM=sM)


# ----------------------------------------------------------------------------
# Generic nonlinear perfect-foresight solver (no constant-saving assumption)
# ----------------------------------------------------------------------------
class Generic:
    def __init__(self, q, Z, Gam, PA, terminal):
        self.q, self.Z, self.Gam, self.PA = q, Z, Gam, PA
        self.T = len(Z) - 1
        self.term = terminal

    def paths(self, z):
        q, T, rho = self.q, self.T, self.q["rho"]
        sK = np.append(z[:T], self.term[0])
        sA = np.append(z[T:], self.term[1])
        K, A, Y = np.empty(T + 2), np.empty(T + 2), np.empty(T + 1)
        K[0], A[0] = initial_state(q)
        for t in range(T + 1):
            Y[t] = output(q, K[t], self.Z[t], self.Gam[t] * A[t])
            K[t + 1] = (q["Abar_rho"]["K"] * ((1 - q["delta"]["K"]) * K[t] ** rho
                                              + q["delta"]["K"] * (sK[t] * Y[t]) ** rho)) ** (1 / rho)
            A[t + 1] = (q["Abar_rho"]["A"] * ((1 - q["delta"]["A"]) * A[t] ** rho
                                              + q["delta"]["A"] * (sA[t] * Y[t] / self.PA[t]) ** rho)) ** (1 / rho)
        return dict(sK=sK, sA=sA, K=K, A=A, Y=Y)

    def resid(self, z):
        q, T, rho, e = self.q, self.T, self.q["rho"], self.q["e"]
        if np.any(z <= 0):
            return np.full(2 * T, 1e3)
        P = self.paths(z)
        Y, K, A, sK, sA = P["Y"], P["K"], P["A"], P["sK"], P["sA"]
        C = (1 - sK - sA) * Y
        if np.any(C <= 0):
            return np.full(2 * T, 1e3)
        m = q["beta"] * (C[1:] / C[:-1]) ** (-1 / q["ies"])
        Gam, PA = self.Gam, self.PA
        a = q["a"]
        MPK = a["K"] * (Y / K[:-1]) ** (1 - rho)
        MPA = a["A"] * Gam ** rho * (Y / A[:-1]) ** (1 - rho)
        IK, IA = sK * Y, sA * Y
        QK = IK ** (1 - rho) * K[1:] ** (rho - 1) / (q["Abar_rho"]["K"] * q["delta"]["K"])
        QA = PA ** rho * IA ** (1 - rho) * A[1:] ** (rho - 1) / (q["Abar_rho"]["A"] * q["delta"]["A"])
        # dK''/dK' at fixed I' = Abar^rho (1-delta) K'^(rho-1) K''^(1-rho)
        dKK = q["Abar_rho"]["K"] * (1 - q["delta"]["K"]) * K[1:-1] ** (rho - 1) * K[2:] ** (1 - rho)
        dAA = q["Abar_rho"]["A"] * (1 - q["delta"]["A"]) * A[1:-1] ** (rho - 1) * A[2:] ** (1 - rho)
        eK = m[:T] * (MPK[1:] + QK[1:] * dKK) / QK[:T] - 1
        eA = m[:T] * (MPA[1:] + QA[1:] * dAA) / QA[:T] - 1
        return np.concatenate([eK, eA])

    def solve(self, z0, tol=1e-11, maxit=80):
        z, R = z0.copy(), self.resid(z0)
        n = z.size
        for it in range(maxit):
            nr = np.max(np.abs(R))
            if nr < tol:
                break
            J = np.empty((n, n))
            for i in range(n):
                dz = np.zeros(n); dz[i] = 1e-7 * max(1.0, abs(z[i]))
                J[:, i] = (self.resid(z + dz) - R) / dz[i]
            step = np.linalg.solve(J, -R)
            lam = 1.0
            while lam > 1e-5:
                zn = z + lam * step
                Rn = self.resid(zn)
                if np.max(np.abs(Rn)) < nr:
                    break
                lam *= 0.5
            z, R = zn, Rn
        self.z, self.iters, self.err = z, it, float(np.max(np.abs(R)))
        return self


def shock(q, kind, T, delta=am.DELTA, step=False):
    t = np.arange(T + 1)
    Z = q["G"] ** t
    S = np.ones(T + 1) if step else am.scurve(T)
    S = S * delta
    if kind == "disembodied":
        return Z, np.exp(S), np.ones(T + 1)
    return Z, np.ones(T + 1), np.exp(-S)


def main():
    checks = {}
    T = 120
    results = {}
    for e in (0.5, 1.5):
        q = params(e)
        # closed-form BGP saving rate: Lambda^e (one capital) and two-capital fixed point
        kK, kA, sK, sA = steady_kappa(q, 1.0, 1.0)
        KY, AY = initial_state(q)
        g = q["g_Z"]
        checks[f"e={e}: steady s_K = (g+d_K) K/Y"] = abs(sK - (g + q["d"]["K"]) * KY)
        checks[f"e={e}: steady s_A = (g+d_A) A/Y"] = abs(sA - (g + q["d"]["A"]) * AY)
        # one-capital formula check (drop AI by giving it a tiny weight)
        q1 = params(e)
        q1["a"] = dict(q1["a"], A=1e-300)
        _, _, s1, _ = steady_kappa(q1, 1.0, 1.0)
        Lam = (q["beta"] * q["Abar_rho"]["K"] * q["delta"]["K"] * q["a"]["K"]
               / (1 - q["beta"] * q["Abar_rho"]["K"] * (1 - q["delta"]["K"])))
        checks[f"e={e}: one-capital s = Lambda^e"] = abs(s1 - Lam ** e)
        results[e] = {}
        for kind in ("disembodied", "embodied"):
            for step in (False, True):
                Z, Gam, PA = shock(q, kind, T, step=step)
                cf = closed_form(q, Z, Gam, PA)
                _, _, sKT, sAT = steady_kappa(q, Gam[-1], PA[-1])
                gen = Generic(q, Z, Gam, PA, (sKT, sAT))
                z0 = np.concatenate([np.full(T, 0.95 * sKT), np.full(T, 1.05 * sAT)])
                gen.solve(z0)
                P = gen.paths(gen.z)
                lab = f"e={e} {kind} {'step' if step else 'S-curve'}"
                checks[f"{lab}: generic solve = closed-form saving rates"] = float(max(
                    np.max(np.abs(P["sK"][:T] - cf["sK"][:T])),
                    np.max(np.abs(P["sA"][:T] - cf["sA"][:T]))))
                print(f"  {lab:32s} Newton iters={gen.iters} resid={gen.err:.1e} "
                      f"s_A: {cf['sA'][0]:.4f} -> {cf['sA'][T]:.4f}")
                results[e][(kind, step)] = cf
        # Z shock alone: saving rates exactly constant (state-independent)
        t = np.arange(T + 1)
        Z = q["G"] ** t * np.exp(0.1 * (t >= 5))
        cf = closed_form(q, Z, np.ones(T + 1), np.ones(T + 1))
        checks[f"e={e}: labor-technology shock leaves s_K, s_A constant"] = float(
            np.ptp(cf["sK"]) + np.ptp(cf["sA"]))
        _, _, sKT, sAT = steady_kappa(q, 1.0, 1.0)
        gen = Generic(q, Z, np.ones(T + 1), np.ones(T + 1), (sKT, sAT)).solve(
            np.concatenate([np.full(T, 0.95 * sKT), np.full(T, 1.05 * sAT)]))
        P = gen.paths(gen.z)
        checks[f"e={e}: ... confirmed by generic solve"] = float(np.max(np.abs(P["sK"][:T] - sKT)))

    # Exactness requires IES = e: with log utility and e = 0.5, saving rates move
    viol = {}
    for e in (0.5, 1.5):
        qv = params(e, ies=1.0)
        t = np.arange(T + 1)
        Z = qv["G"] ** t * np.exp(0.1 * (t >= 5))
        # initial BGP saving rates are unchanged (BGP formulas); solve generic model
        KY, AY = initial_state(qv)
        s0 = ((qv["g_Z"] + qv["d"]["K"]) * KY, (qv["g_Z"] + qv["d"]["A"]) * AY)
        gen = Generic(qv, Z, np.ones(T + 1), np.ones(T + 1), s0).solve(
            np.concatenate([np.full(T, s0[0]), np.full(T, s0[1])]))
        P = gen.paths(gen.z)
        viol[e] = dict(sK_min=float(P["sK"][:T].min()), sK_max=float(P["sK"][:T].max()),
                       resid=gen.err)
        print(f"  IES=1, e={e}: s_K in [{viol[e]['sK_min']:.4f}, {viol[e]['sK_max']:.4f}] "
              f"(BGP {s0[0]:.4f}); resid {gen.err:.1e}")

    print("Checks:")
    for k, v in checks.items():
        print(f"  {k:66s} {v:.2e}")

    # ------------------------------------------------------------------ figure
    H = 40
    t = np.arange(H)
    q1 = cd.cd_params()
    cd_sims = {}
    for kind in ("disembodied", "embodied"):
        Z, Gam, PA = cd.shock_paths(q1, kind, T)
        cd_sims[kind] = cd.simulate_closed_form(q1, Z, Gam, PA)
        Zb, Gb, PAb = cd.shock_paths(q1, kind, T, delta=0.0)
        cd_sims[kind]["base"] = cd.simulate_closed_form(q1, Zb, Gb, PAb)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for col, e in enumerate((0.5, 1.0, 1.5)):
        for kind in ("disembodied", "embodied"):
            if e == 1.0:
                S = cd_sims[kind]
                r, w = S["r"], S["W"] / S["base"]["W"]
            else:
                S = results[e][(kind, False)]
                q = params(e)
                r = S["r"]
                w = S["W"] / (q["a"]["L"] * q["G"] ** np.arange(T + 1) * 1.0)
            axes[0, col].plot(t, 100 * r[:H], color=am.COL[kind], lw=2, label=kind)
            axes[1, col].plot(t, 100 * (w[:H] - 1), color=am.COL[kind], lw=2)
        axes[0, col].axhline(100 * am.BASE["r0"], color="0.5", ls=":", lw=1)
        axes[0, col].set_title(am.LAB[e] + (", IES = 1" if e == 1 else f", IES = {e}"), fontsize=10)
        axes[1, col].set_xlabel("years after announcement")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("real interest rate $r_{t+1}$ (pct.)")
    axes[1, 0].set_ylabel("real wage $W/Z$ (pct. dev.)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Exact solutions, CES production with CES accumulation: 20-fold S-curve fall in the cost of AI",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_cesexact.{ext}"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ table
    L = [r"\begin{tabular}{lcccccc}\toprule",
         r" & \multicolumn{2}{c}{$I_A/Y$} & \multicolumn{2}{c}{peak $r$ (pct.), S-curve} & \multicolumn{2}{c}{$r_1$ (pct.), surprise step}\\",
         r" & initial & final & disemb. & emb. & disemb. & emb.\\\midrule"]
    for e in (0.5, 1.5):
        R = results[e]
        d, m = R[("disembodied", False)], R[("embodied", False)]
        ds, ms = R[("disembodied", True)], R[("embodied", True)]
        sA0 = steady_kappa(params(e), 1.0, 1.0)[3]
        L.append(f"$\\varepsilon={e:g}$ & {sA0:.3f} & "
                 f"{d['sA'][-1]:.3f} & {100*d['r'].max():.2f} & {100*m['r'].max():.2f} & "
                 f"{100*ds['r'][0]:.2f} & {100*ms['r'][0]:.2f}\\\\")
    S1d, S1e = cd_sims["disembodied"], cd_sims["embodied"]
    with open(os.path.join(HERE, "results_cdcd.json")) as f:
        cdj = json.load(f)
    L.insert(4, f"$\\varepsilon=1$ & {cd.closed_form_rates(q1)['sA']:.3f} & "
                f"{cd.closed_form_rates(q1)['sA']:.3f} & {100*S1d['r'].max():.2f} & "
                f"{100*S1e['r'].max():.2f} & {100*cdj['step_r0']['disembodied']:.2f} & "
                f"{100*cdj['step_r0']['embodied']:.2f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_cesexact.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    C = [r"\begin{tabular}{lc}\toprule Check & max abs. error\\\midrule"]
    for k, v in checks.items():
        C.append(f"{k.replace('...', 'and')} & {v:.1e}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    with open(os.path.join(HERE, "table_checks_cesexact.tex"), "w") as f:
        f.write("\n".join(C) + "\n")

    out = dict(checks=checks, ies_violation=viol,
               params={str(e): {k: v for k, v in params(e).items()} for e in (0.5, 1.5)},
               summary={str(e): {f"{k[0]}_{'step' if k[1] else 'scurve'}":
                                 dict(peak_r=float(v["r"].max()), r1=float(v["r"][0]),
                                      sA0=float(v["sA"][0]), sAT=float(v["sA"][-1]),
                                      wage_T=float(v["W"][-1] / (params(e)["a"]["L"] * params(e)["G"] ** T)))
                                 for k, v in results[e].items()} for e in (0.5, 1.5)})
    with open(os.path.join(HERE, "results_cesexact.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
