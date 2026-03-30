RG-TS Simulation Suite
======================

Files
-----
- `rgts_simulation_suite.py`: Colab-ready Python script.

What it does
------------
Implements the paper's Robust Gibbs-Thompson Sampling (RG-TS) for heavy-tailed bandits
using a generalized Gibbs posterior with Catoni's psi and ULA sampling, and compares it with:
- GaussianTS
- CatoniUCB
- UCB1

Scenarios
---------
1. Pareto_InfiniteVariance
2. Whale_Contamination
3. Lognormal_Skew

Outputs created when you run the script
---------------------------------------
- `rgts_sim_outputs/tables/*.csv`
- `rgts_sim_outputs/figures/*.png`
- `rgts_sim_outputs.zip`

How to run in Colab
-------------------
1. Upload `rgts_simulation_suite.py`
2. Run:
   `!python rgts_simulation_suite.py`

Switch between quick and larger runs
------------------------------------
Inside the script, change:
`RUN_PROFILE = "fast"`
to
`RUN_PROFILE = "paper"`

Notes
-----
- The script is intentionally honest: it does not hard-code RG-TS to win in every scenario.
- Its strongest advantage should appear in whale-contamination and strongly heavy-tailed settings.
- If you want to tune RG-TS further, the main knobs are:
  `rgts_eta`, `rgts_M`, `alpha_coef`, `alpha_max`, and `prior_sd`.
