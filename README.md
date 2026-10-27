# VERA — Verifiable Analytics Agent

VERA is the hackathon build for the Meta Global AI Developer Hackathon.

## Pre-existing infrastructure

VERA uses the pre-existing deterministic Data Analysis Engine baseline (, 95 native analytical capabilities) as technical infrastructure.

## Built during the hackathon

All VERA-facing agent workflows, Meta Model API integration, verification, replanning, visualization, UI/API work, and analytical extensions in  are new hackathon work.

Current hackathon analytical extensions:

- #96 `one_way_anova`
- #97 `mann_whitney_u_test`
- #98 `kruskal_wallis_test`
- #99 `multiple_testing_correction`
- #100 `extreme_value_tail_risk`
- #101 `bayesian_structural_causal_impact`
- #102 `directed_transfer_entropy`

The pre-existing engine and the hackathon extensions remain explicitly separated for auditability.
