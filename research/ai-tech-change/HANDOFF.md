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

## Status (session 2, second request)
The user says the published JoB article has two one-capital closed-form pairings:
CES production + adjustment-cost function, and Cobb-Douglas production + Cobb-Douglas
accumulation (K' = Abar K^(1-delta) I^delta). The published text was not available
(JSTOR, not in Drive), so both were derived from scratch:
- Pairing (i): closed forms on the BGP only (K/Y = alpha R^-e, V/Y ~ alpha/(r+d)^e).
- Pairing (ii), with log utility: exact for any path. s_j = B delta_j alpha_j/(1-B(1-delta_j)),
  V/Y = sum alpha_j/(rho+delta_j) (constant), 1+r' = (1+rho)Y'/Y, W = aN Y/N, and a
  log-linear VAR in (ln K, ln A). Embodied and disembodied progress have the same
  cumulative effect on ln(1+r); the timing differs (step versus S-curve reverses the ranking).
- `cdcd_model.py`: closed form, a generic Newton check, matched-curvature comparison with
  the quadratic model, and a CES + CD-accumulation counterexample.
If the user supplies the published PDF, align notation and parameterization with it.

## Status (session 2, third request)
Clarification from the user: in the JoB article, CES production pairs with a CES
investment/accumulation equation, giving an exact solution with a constant saving
rate. The CD/CD pair is the special case. Derived and verified (`ces_exact.py`):
- K'^rho = Abar^rho[(1-delta)K^rho + delta I^rho], Y^rho = a_K K^rho + a_L (ZN)^rho,
  U = C^rho/rho (IES = e; in paper's notation b(1-s) = rho), hours fixed.
  V = kappa K^rho/rho; s = Lambda^e, Lambda = B Abar^rho delta a_K/(1 - B Abar^rho (1-delta)).
- AI: three-input CES with common e, plus a CES accumulation for A with I_A/P_A. Saving
  rates are time-only functions from a backward recursion in kappa_K, kappa_A (state-free,
  invariant to Z). The nested CD bundle inside CES is exact only at e = 1.
- Exactness fails with IES != e (checked numerically).
Still unverified against the published text: notation and whether the article also
imposes IES = e.

## Workflow rule from the user
Once an analytic (closed-form) case is confirmed, stop and report back before
building any AI extension of it. The user has their own ideas for the AI cases.

## Status (session 2, fourth request): three cases, AI vs other progress
The user's design: same long-run output gain for every shock. Disembodied (Gamma in
production) vs embodied (P in accumulation). Labor-augmenting Z as the benchmark.
- Case 1 (one K, e = 0.5) and Case 2 (one K, e = 1.5): exact with IES = e. Saving rates
  come from a time-only backward recursion. Closed-form long run: ln W = gamma_Y/e,
  dln(labor share) = -rho gamma_Y, shock size (Gamma/P)^(e-1) = [1-(1-s_K)e^(-rho gamma_Y)]/s_K.
- Case 3 (K-L bundle CD or CES 0.5, AI substitute e2 = 1.5, AI share 3%): no closed form
  (X^rho2 is not linear in K^rho1 unless e1 = e2). Solved numerically with log utility.
  A CD bundle makes K-augmenting progress equivalent to Z (Uzawa).
- Files: cases.py, ai_cases.tex/.pdf, figures fig_case12_exact, fig_case12_log, fig_case3,
  fig_case3_ces, and tables table_case*_*.tex, table_checks_cases.tex, results_cases.json.
Next: wait for the user's direction (see the workflow rule above).

## Status (session 2, fifth request): separate comparison note
`ai_growth_note.tex`/`.pdf` and `growth_cases.py`. Baseline: CD K-L bundle, AI a CES
substitute (e = 1.5), log utility, fixed hours. Linear accumulation vs. adjustment costs
(c_j = 1/x_j, I/K-Q elasticity of one). Disembodied (Gamma_j) vs. embodied (P_j), for AI and for K.
- Case 1: surprise level jumps (same 10% long-run Y). Case 2: permanent faster growth
  (+0.5 pp g). AI growth uses the e = 1 variant, because e > 1 has no BGP with ongoing
  AI progress. Case 3: S-curve centered on year 10, announced at t = 0.
- Closed forms: BGP (Prop 1). Level shift: dlnW = gamma_Y (K) vs gamma_Y/[1+(1-a)(e-1)] (AI).
- Key dynamics: embodied progress raises r more on impact. Ongoing or anticipated embodied
  progress with linear accumulation first LOWERS r (firms wait for cheaper capital);
  adjustment costs mostly remove that.

## Status (session 2, sixth request): note narrowed to jump + S-curve
The user dropped the faster-growth case (no BGP with e > 1) and linear accumulation.
`ai_growth_note`/`growth_cases.py` were renamed to `ai_capital_note.tex/.pdf` and
`capital_cases.py`. Now: adjustment costs only (c_j = 1/x_j), same elasticities,
same 10% long-run output gain; summary of r, W and labor share in `table_capital_cases.tex`
and `fig_capital_cases`. Added: half-jump/half-S-curve case ("mixed") and output (net of adjustment costs).

## Status (session 2): saving models (not yet in a note)
`saving_models.py`: same production as capital_cases (e = 1.5 baseline, d_A = d_K = 0.066,
no adjustment costs, ramp S-curve complete by year 10). Saving models: Ramsey, Solow (fixed s),
perpetual youth (p = 0.025, flat labor income), Diamond OLG (30-year periods, long run only).
Long run: closed-form fixed points in r. Key result: Delta r = 0 for regular-capital progress
in every model, and for AI progress when e = 1; for AI with e = 1.5 it is Solow +1.09 pp,
Diamond +0.44, PY +0.04, Ramsey 0. Outputs: fig_saving_longrun, fig_saving_trans_{A,K},
table_saving_*.tex, results_saving_models.json.

## Status (session 2): consolidated write-up
`ai_capital_note.tex/.pdf` is now the comprehensive note: model, closed-form long run,
Ramsey dynamics in Setting A (d_A = 0.20, adjustment costs) and Setting B (equal d, no
adjustment costs), the elasticity grid, and saving models (Ramsey, perpetual youth, annual
OLG with 60 cohorts, Diamond, Solow). The S-curve is now the cosine ramp complete by year 10
everywhere (capital_cases default changed). The annual OLG is solved by Levenberg-Marquardt
from the Ramsey path; its long run is read at T-100 (truncation artifact in the final cohorts).
Run: `python3 capital_cases.py; python3 capital_cases.py --equal-d-noadj; python3 saving_models.py`
(about 6 minutes), then pdflatex x3.
