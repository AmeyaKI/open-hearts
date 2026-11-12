# Experiments

Every benchmark and experiment in this project is a script in this folder. Results are written to `results/` (not committed) and are reproducible by seed. What each experiment found is written up in [`../docs/RESEARCH_LOG.md`](../docs/RESEARCH_LOG.md), and the opponents used are documented in [`../docs/BENCHMARK_REGISTRY.md`](../docs/BENCHMARK_REGISTRY.md).

Most scripts are run from the repository root, for example `.venv/bin/python experiments/<script>.py`. Some long experiments take hours, and the OpenSpiel matches need a separate environment (see the docstring at the top of each script).

## Start here

| Script | What it does |
|---|---|
| [`relay.py`](relay.py) | Terminal tool for logging real Hearts games as they are played at a table, with an advisor mode that suggests cards for your seat. |
| [`run_xinxin_match.py`](run_xinxin_match.py) | Plays TallyHearts against xinxin, the academic reference Hearts bot, in OpenSpiel. |
| [`run_cbench.py`](run_cbench.py) | Plays TallyHearts against OpenSpiel's generic ISMCTS bot inside OpenSpiel's own Hearts game. |
| [`run_exploit_eval.py`](run_exploit_eval.py) | Measures exploitability: how many extra points a dedicated student of TallyHearts can take from it. |
| [`run_ablation2.py`](run_ablation2.py) | Compares belief levels (uniform, voids, full) inside honest search against heuristic opponents. |

## Hidden-card inference (guessing)

| Script | What it does |
|---|---|
| [`run_guessing.py`](run_guessing.py) | Scores how well belief tables guess the hidden hands at each evidence level, using recorded heuristic games. |
| [`run_guessing2.py`](run_guessing2.py) | Adds choice-aware guessing, which uses how opponents chose their cards, and scores it against the constraint-only curves. |
| [`run_guessing5.py`](run_guessing5.py) | Repeats the guessing comparison against held-out opponent personalities the bot has never seen. |
| [`run_robustness.py`](run_robustness.py) | Tests how strict choice-filtering degrades when opponents sometimes play random cards instead of the assumed policy. |
| [`run_survival.py`](run_survival.py) | Measures how many candidate worlds survive strict choice filtering at each trick boundary. |
| [`run_sampler_stats.py`](run_sampler_stats.py) | Measures the failure rate and attempt counts of the hidden-hand sampler on recorded games. |
| [`run_sir_oracle.py`](run_sir_oracle.py) | Tests whether the earlier loss of the profiled-oracle row came from importance weighting rather than from the information itself. |

## Search ablations

| Script | What it does |
|---|---|
| [`run_ablation.py`](run_ablation.py) | First ablation: search at three belief levels plus heuristic and random baselines, then a sample-count sweep. |
| [`run_ablation_parallel.py`](run_ablation_parallel.py) | Runs the first ablation across many cores, with identical per-game seeding to the serial version. |
| [`run_noleak.py`](run_noleak.py) | Control run for uniform beliefs with a sampler that does not use observed voids. |
| [`run_bprobe.py`](run_bprobe.py) | Reruns the belief-level ablation against randomized, less predictable opponents. |
| [`run_ablation2.py`](run_ablation2.py) | Compares belief levels inside honest search against heuristic opponents, with a bridge row to the first search implementation. |
| [`run_entropy_curve.py`](run_entropy_curve.py) | Measures what opponent reading is worth as opponents become more or less predictable. |
| [`run_world_sweep.py`](run_world_sweep.py) | Checks whether the shipped search size is still the right number of sampled worlds after the speedup. |
| [`run_grouping_ablation.py`](run_grouping_ablation.py) | Paired check that grouping equivalent cards leaves search strength unchanged. |
| [`bench_grouping.py`](bench_grouping.py) | Timing and candidate-count report for equivalent-card grouping. |

## Learned models (value net, profiler)

| Script | What it does |
|---|---|
| [`gen_value_data.py`](gen_value_data.py) | Generates self-play position and outcome shards for training the value network. |
| [`train_value.py`](train_value.py) | Trains the value network and scores it against honest baselines. |
| [`run_value_probe.py`](run_value_probe.py) | Checks whether value-truncated search agrees with full playouts and how much cheaper it is. |
| [`run_ablation4.py`](run_ablation4.py) | Tests whether a learned value network beats heuristic playouts inside honest search. |
| [`gen_population_data.py`](gen_population_data.py) | Generates one row per real card choice from a population of opponent personalities, for training the profiler. |
| [`train_profiler.py`](train_profiler.py) | Trains the opponent profiler and scores it on held-out personalities. |
| [`measure_adaptation.py`](measure_adaptation.py) | Measures how much of the adaptation headroom a mixture of profiles captures on personalities it has not seen. |

## Opponent populations and league

| Script | What it does |
|---|---|
| [`run_league.py`](run_league.py) | Round-robin league of bot versions, each playing one seat against a trio of another, on identical deals. |
| [`run_ablation5.py`](run_ablation5.py) | Main population ablation: several search variants against 20 held-out personality trios. |
| [`run_ablation7.py`](run_ablation7.py) | Plays the champion against 50 sealed held-out expert opponents, with oracle diagnostics. |
| [`run_ablation7c.py`](run_ablation7c.py) | Tests whether imagining expert opponents in rollouts, instead of the beginner heuristic, reduces points taken. |
| [`expert_diagnostics.py`](expert_diagnostics.py) | Reports train-only diagnostics for the expert opponent population, such as rule firing rates and entropy. |

## External benchmarks (OpenSpiel ISMCTS, xinxin)

| Script | What it does |
|---|---|
| [`run_cbench.py`](run_cbench.py) | Matches against OpenSpiel's ISMCTS bot; needs a separate environment with `pyspiel` installed. |
| [`run_xinxin_match.py`](run_xinxin_match.py) | Matches against xinxin; needs a source-built OpenSpiel with xinxin and the local patch below. |
| [`xinxin_knoshooting.patch`](xinxin_knoshooting.patch) | Local patch to OpenSpiel's xinxin bot enabling its own no-shooting-the-moon rule, so it optimizes the objective we score. |
| [`tune_ismcts.py`](tune_ismcts.py) | Tunes OpenSpiel ISMCTS settings on separate tuning deals, to give that opponent a fair chance. |
| [`cbench/`](cbench/) | Package of OpenSpiel bridge code: `adapter.py` links our engine to OpenSpiel's Hearts game, `evaluators.py` adds Hearts-aware evaluators for ISMCTS. |

## Robustness and exploitability

| Script | What it does |
|---|---|
| [`run_exploit_eval.py`](run_exploit_eval.py) | Measures exploitability of the frozen champion by seating one exploiter that models the champion in its imagination. |
| [`run_deviation_probe.py`](run_deviation_probe.py) | Locates which decisions in the hand let the exploiter take its extra points. |
| [`probe_model_agreement.py`](probe_model_agreement.py) | Measures how often the exploiter's miniature copy of the champion picks the same card as the real champion. |

## Live play

| Script | What it does |
|---|---|
| [`relay.py`](relay.py) | Terminal relay for real games, with house rules of no passing and no moon rescoring, plus an advise mode for your seat. |

## Utilities (watchdog, probes, data generation)

| Script | What it does |
|---|---|
| [`watchdog.py`](watchdog.py) | Watches long-run checkpoint files from outside the run and only logs stalls, never killing anything. |
| [`probe_blocks.py`](probe_blocks.py) | Measures whether smaller checkpoint blocks reduce idle-core tail time in parallel runs. |
