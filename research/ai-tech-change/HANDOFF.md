# AI-driven technical change: project handoff

## Goal
Use Kiley (2004), "Stock Prices and Fundamentals: A Macroeconomic Perspective,"
Journal of Business 77(4): 909-936 (working paper: FEDS 2000-05,
https://www.federalreserve.gov/pubs/feds/2000/200005/200005pap.pdf) as a blueprint.
That paper uses CES production and investment adjustment costs, with simple
closed-form balanced-growth solutions. Extend it to AI-driven technical change
and derive closed forms for the flex-price equilibrium real interest rate and
real wage.

## First step in a new session
Download the FEDS 2000-05 PDF and copy its exact notation, adjustment-cost form,
and closed-form expressions for q and value/output. The previous session could
not download it because its environment had Trusted network access.

## Planned model (draft)
- Traditional bundle: X = K^a (Z L)^(1-a).
- Output: CES in X and effective AI, with weight w and elasticity sigma:
  Y = [w X^((s-1)/s) + (1-w)(Gamma A)^((s-1)/s)]^(s/(s-1)).
  sigma < 1 means AI complements the bundle; sigma > 1 means it substitutes.
- Disembodied AI progress: Gamma grows at gamma_G.
- Embodied AI progress (Greenwood-Hercowitz-Krusell): the price of AI
  efficiency units, p, falls at rate gamma_q.
- Adjustment costs: (phi/2)(I/K)^2 K. Then q = 1 + phi*i on the balanced
  growth path, with i = g + delta.
- User cost of traditional capital:
  u_K = (1 + phi_K i_K)(r + d_K) - (phi_K/2) i_K^2.
- User cost of AI:
  u_A = [(1+r)(1+gamma_q) - (1-d_A)](1 + phi_A i_A) - (phi_A/2) i_A^2.
  This includes an obsolescence term from embodied progress.
- Interest rate: 1 + r = (1+g)^theta / beta.
- Effective AI cost: lambda = p u_A / Gamma.
- Labor share: (1-a)[1 - (1-w)^s lambda^(1-s)].
- Wage, from the unit-cost condition:
  P_X^(1-s) = w^(-s)[1 - (1-w)^s lambda^(1-s)],
  w_L = Z(1-a)[P_X (a/u_K)^a]^(1/(1-a)).
- Cobb-Douglas case (sigma = 1): exact balanced growth path with
  g = g_z + (1-w)(gamma_G + gamma_q) / ((1-a) w).
- sigma > 1: the economy tends toward an AK regime (closed form).
- sigma < 1: Baumol-type limit.
- Numerics: perfect-foresight transition paths for an embodied vs. a
  disembodied S-curve AI shock, checked against the closed forms.

## Status (session 2, 2026-10-12)
Done: paper downloaded and read; model derived, verified, written up.
- `ai_tech_change.tex` / `.pdf`: research note (closed forms, limits, transitions).
- `ai_model.py`: closed forms, verification against equilibrium conditions,
  perfect-foresight transitions, figures (`figures/`), tables (`table_*.tex`),
  `results.json`. Run `python3 ai_model.py` (numpy, matplotlib), then pdflatex twice.

Corrections to the draft above, after reading FEDS 2000-05:
- The working paper's production is Cobb-Douglas K-N; CES appears only in this
  extension (the AI aggregator). The published JoB version was not checked (paywalled).
- Paper's adjustment cost is 0.5c(I/K - x)^2 K with x = g + d, so adjustment costs
  vanish on the BGP and Q = 1 exactly. The user costs are therefore
  R_K = r + d_K and u_A = (1+r)(1+gamma_q) - (1-d_A), with no adjustment-cost terms.
- CES is written in share form, F = [w^(1/e) X^rho + (1-w)^(1/e) M^rho]^(1/rho),
  so s_M = (1-w) lambda^(1-e) and P_X^(1-e) = (1-s_M)/w.
- Paper's notation: s is utility curvature, so the AI elasticity is epsilon (e), and the
  weight is omega. 1 + r = (1+g)^theta / B with theta = 1 - b(1-s).

Possible next steps: calibrate e, d_A, gamma_q to data; nested task-based CES
for labor; discount-rate (equity premium) experiments as in Section 3 of the paper.
