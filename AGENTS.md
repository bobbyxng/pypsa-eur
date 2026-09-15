# pypsa-eur (bobbyxng fork) — Agent Guide

This is a fork of [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) (upstream), on branch
`resilient-islands`, currently based on upstream `master` as of **2026-09-14**, which includes
the streamlined-workflow restructure ([#1838](https://github.com/PyPSA/pypsa-eur/pull/1838)).
Pinned deliberately at that merge, not continuously tracking `master`. `origin` points at this fork
(`git@github.com:bobbyxng/pypsa-eur.git`); `upstream` points at `PyPSA/pypsa-eur.git` for
whenever a future release is deliberately adopted. This submodule is driven from the parent
[PyPSARO](../../../README.md) repo — see that repo's `AGENTS.md`/`MEMORY.md` for the actual
research context (the ARO/C&CG resilience workflow this fork's networks feed into). If you're
working here without that context loaded, go read it first.

## Critical rules

- **Never add a `Co-Authored-By: Claude` (or any AI co-author) trailer to a commit, ever** —
  same standing rule as the parent PyPSARO repo.
- **Never auto-commit or auto-push** without an explicit ask, every time.
- **This branch's base is intentionally locked.** New work goes on top of it as ordinary
  commits on `resilient-islands`; don't merge or rebase onto upstream `master` casually.
  Moving to a newer upstream state is a deliberate, occasional action, not an automatic sync.
  When you do move, read upstream's `doc/migration.md` first — the #1838 restructure needed a
  full pass over the fork's rules, configs and tests, not just conflict resolution.
- **SPDX headers are required on every file** (`REUSE.toml` maps file globs to licenses: code
  → MIT, docs/images → CC-BY-4.0, generated/config files → CC0-1.0). Pre-commit's
  `reuse-lint-file` hook enforces this — don't hand-roll a header, copy the pattern from a
  neighboring file of the same type.
- If anything here is ever meant to go back upstream as a PR, read `doc/contributing.md`
  first — it has an explicit "AI-based Contributions" policy (keep PR descriptions
  human/concise, mark verbose AI-generated content in collapsed `<details>`, keep changes
  focused). Not restated here; that policy is upstream's, not this fork's.

## Repo map

- `Snakefile` — top-level rules: `all` (default target), `create_scenarios`, `purge`,
  `dump_graph_config`, `rulegraph`/`filegraph`, `doc`, `sync`/`sync_dry`.
- `rules/*.smk` — `build_electricity.smk` (base network, shapes, renewable/hydro profiles,
  demand, clustering), `build_sector.smk` (heat/gas/industry/biomass/transport/CO2),
  `compose.smk` (`compose_network`, the single rule that replaced the old
  `add_electricity`/`prepare_network`/`prepare_sector_network`/`add_existing_baseyear`/
  `add_brownfield` chain, plus `get_compose_inputs`), `solve.smk` (`solve_network` for all
  foresight modes, `solve_operations_network`), `aro.smk` (fork-only, see below),
  `collect.smk` (aggregation targets — `cluster_networks`, `compose_networks`,
  `solve_networks`, `plot_balance_maps`), `postprocess.smk` (summary/plotting),
  `retrieve.smk` (~30 `retrieve_*` rules for raw input data), `development.smk`
  (maintenance/comparison rules), `common.smk` (shared helpers: `merge_configs`,
  `scenario_config`, `config_provider`, `get_scenarios`, `navigate_config`).

  The stages are `base → simplified → clustered → composed_{horizon} → solved_{horizon}`.
  Only `{horizon}` and `{run}` remain as wildcards; everything the old
  `{clusters}`/`{opts}`/`{sector_opts}` wildcards carried now lives in the config.
- `scripts/*.py` — 113 files. Central ones: `add_electricity.py`,
  `build_electricity_demand.py`, `build_electricity_demand_base.py`, `prepare_network.py`,
  `prepare_sector_network.py`, `solve_network.py`.
- `scripts/lib/validation/config/` — the actual config **source of truth** (see below), not
  `config.default.yaml`.
- `test/` — pytest suite (`test_base_network.py`, `test_compose_network.py`,
  `test_config_schema.py`, `test_co2_budget.py`, `test_make_summary.py`, and the fork's
  `test_prepare_aro_network.py`); distinct from `config/test/`, which holds minimal configs
  for Snakemake integration-test runs (`config.electricity.yaml`, `config.overnight.yaml`,
  `config.myopic.yaml`, `config.perfect.yaml`, the fork's `config.prepare_aro.yaml` /
  `config.prepare_aro_sector.yaml`, etc.), not unit tests.
- `doc/` — MkDocs site (`pypsa-eur.readthedocs.io`), not Sphinx.

## Config/schema — this is generated, and generated the opposite way from PyPSARO's own

**`config/config.default.yaml` and `config/schema.default.json` are generated artifacts, not
hand-authored** — the actual source of truth is a pydantic model registry under
`scripts/lib/validation/config/` (~30 files, e.g. `electricity.py`, `sector.py`,
`solving.py`, each registering a `ConfigUpdater` that composes into a `ConfigSchema`).
`validate_config(config)` builds the composite schema and instantiates it against the loaded
config, raising `pydantic.ValidationError` on a bad key/value.

**Do not hand-edit `config.default.yaml` or `schema.default.json`.** After changing anything
under `scripts/lib/validation/config/`, regenerate them:

```
pixi run -e test generate-config    # = pytest test/test_config_schema.py --fix
```

`test_config_schema.py` asserts these two files are byte-identical to what regeneration would
produce, and fails with exactly this instruction if they've drifted — this is the guard rail,
don't bypass it by editing the generated files directly.

There is no `config/config.yaml` by default — it's gitignored, optional, user-created
(precedence: `--config` CLI > `--configfile` > `config/config.yaml` >
`config.default.yaml`/`plotting.default.yaml`). PyPSARO instead drives this submodule via
`--configfile` pointing at PyPSARO's own `config/pypsa-eur/*.yaml` files — see the `pypsa-eur`
task in PyPSARO's `pixi.toml`. `config/examples/` has six full example configs as reference
if you need to see broader option combinations than PyPSARO's own configs use.

## Build / test / lint

Tasks are split across pixi environments/features — note the `-e` flag, it's not optional
for most of these:

| task | command | env |
|---|---|---|
| unit tests | `pixi run -e test unit-tests` (`pytest test`) | test |
| integration tests | `pixi run -e test integration-tests` (Snakemake runs against `config/test/*.yaml`) | test |
| integration tests, streamlined | `pixi run -e test integration-tests-streamlined` (the electricity-only myopic/perfect configs added by #1838) | test |
| regenerate config schema | `pixi run -e test generate-config` | test |
| clean test outputs | `pixi run -e test clean-tests` | test |
| everything above | `pixi run -e test all-tests` | test |
| build docs | `pixi run -e doc build-docs` | doc |
| reset local outputs | `pixi run reset` (interactive; wipes logs/resources/benchmarks/results/.snakemake) | default |
| regenerate doc DAG diagrams | `pixi run update-dags` | default |
| regenerate env lockfiles | `pixi run sync-locks` | default |

Lint/format is pre-commit-driven: ruff (lint+format), codespell (scripts/doc), snakefmt,
Jupyter notebook output cleanup, YAML pretty-format, and `reuse-lint-file`. No mypy/type
checking here.

**CI exists but is master-scoped — it will not run automatically on this branch.** 7 GitHub
workflows live in `.github/workflows/` (`test.yaml`, `release.yaml`, `codeql.yaml`,
`security-scan.yaml`, `update-lockfile.yaml`, `validate.yaml`, `push-images.yaml`), but
`test.yaml`'s triggers are scoped to push/PR against `master`. Don't assume "CI will catch
it" on `resilient-islands` — run the pixi test tasks above yourself before committing
anything non-trivial.

## Fork-specific changes vs. upstream

- **`rules/aro.smk` + `scripts/prepare_aro_network.py` + `scripts/lib/validation/config/aro.py`**
  add an `aro.heat` config section and a `prepare_aro_network` rule that layers an exogenous
  electrified-heat load onto an electricity-only network, as separate `Load` components on the
  existing AC buses (no heat buses, no links, no storage). It reads
  `resources/networks/composed_{horizon}.nc` (built with `sector.enabled: false`) and writes
  `composed_aro_{horizon}.nc`, which is what PyPSARO's `run.network` points at — keeping the
  content assumptions here rather than in PyPSARO. Deliberately additive — new files only,
  plus a one-line `include:` in `Snakefile` and one field in
  `scripts/lib/validation/config/_schema.py` — so a future move onto a newer upstream state
  re-applies it with minimal conflicts. The demand construction is *imported* from
  `prepare_sector_network` (`build_heat_demand`) and `definitions/heat_system` (`HeatSystem`)
  rather than duplicated; only `add_heat`'s bus/link creation is replaced. See the parent repo's
  `memory/heat-electrification-design.md` for the modelling rationale.

  The heat resources it consumes (`cop_profiles_{horizon}.nc`, `district_heat_share_{horizon}.csv`,
  `hourly_heat_demand_total.nc`, …) are produced by rules that are *not* gated on
  `sector.enabled`, which is what makes an electricity-only run able to request them.

  `test_prepare_aro_network.py` needs two runs, not one: since #1838 there is a single
  `composed_{horizon}.nc` per run whose content depends on `sector.enabled`, so
  `config/test/config.prepare_aro.yaml` (electricity-only, the real shape) supplies the network
  under test and `config/test/config.prepare_aro_sector.yaml` (sector-coupled) supplies
  `add_heat`'s own heat loads as ground truth. Keep the two files identical apart from
  `sector.enabled`, `run.name` and the `aro` block.

- **Memory floor.** Every rule that upstream gives an explicit `mem_mb` below 6000 is raised to
  6000 here (and the `nprocesses * N` expressions are wrapped in `max(..., 6000)`), for
  compatibility with the apptainer container. A handful are raised further on measured need
  (`build_ship_raster` 12000, `build_transmission_projects` and
  `add_transmission_projects_and_dlr` 8000). Rules that upstream leaves without a `mem_mb` are
  left alone — they fall back to `--default-resources`. After any upstream merge, re-check with
  `grep -rn "mem_mb=" rules/`.

- **`container:` in `Snakefile` + `container.image` config key** wire up Snakemake's own
  apptainer/singularity per-job wrapping, and hard-fail when `--software-deployment-method
  apptainer` is requested without an image configured (Snakemake would otherwise silently run
  every job uncontainerized).
- **`scripts/_helpers.py`'s `get_scenarios()`** resolves `run.scenarios.file` against
  PyPSARO's root (found by walking up from `workflow.basedir` looking for `.gitmodules`)
  instead of the process's CWD, and `Snakefile` passes `workflow.basedir` into it. This is
  what lets PyPSARO reference `run.scenarios.file` with a clean, PyPSARO-root-relative path
  regardless of whether pypsa-eur is invoked through PyPSARO's wrapper or directly from
  within this submodule. If you add more fork-specific behavioral changes, note them here —
  once there's more than a couple, consider a `memory/`-style breakdown matching PyPSARO's
  own convention rather than letting this section grow unbounded.
