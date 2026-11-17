<!-- SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur> -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# pypsa-eur (bobbyxng fork) — Agent Guide

This is a fork of [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) (upstream), on branch
`resilient-islands`, currently based on upstream release **v2026.09.0** (2026-09-25, merge base
`c8209014`), which includes the streamlined-workflow restructure
([#1838](https://github.com/PyPSA/pypsa-eur/pull/1838)). Pinned deliberately at that release, not
continuously tracking `master`; single upstream fixes needed earlier are cherry-picked and listed
under "Fork-specific changes" below. `origin` points at this fork
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

- **`extract_electrified_heat_shares`** (`rules/aro.smk` +
  `scripts/extract_electrified_heat_shares.py`) reads a *solved* sector-coupled network
  (PyPSARO's `europe-heat-reference` run) and writes
  `results/{run}/heat_shares/electrified_heat_shares_{horizon}.json`: per-country
  heat-pump / resistive shares of building heat supply, to be copied by hand into
  `aro.heat.shares` (same hand-copy pattern as `calibrate_district_heating_potential`). Not
  read by any rule. Anchor: on the old europe-pre-run network it reproduces DE 0.845/0.040 and
  ES 0.946/0.054. Unit tests: `test/test_extract_electrified_heat_shares.py`.

- **Endogenous heat** (`aro.heat.mode: endogenous`, `add_endogenous_heat` in
  `scripts/prepare_aro_network.py`): instead of fixed electric Loads, two heat systems per node
  -- `decentral` (pypsa-eur's rural + urban decentral merged) and `urban central` -- with heat
  Loads served by air heat pumps, resistive heaters, water tanks and heat vents, plus an
  `urban central H2 CHP` priced from `"central gas CHP"`, and, when `aro.methanol` is on, an
  `urban central methanol CHP` on the local methanol tank (`_add_methanol_chp`; same
  `reserve_only` access and fuel pricing as the methanol turbines; techno-economics exactly
  upstream's generic `urban central {fuel} CHP` minus its CO2 port). Wiring, cost keys and TES
  convention mirror `add_heat`; the demand split is shared with the exogenous mode through
  `heat_system_inputs`, so both modes serve identical heat. Tests:
  `test/test_prepare_aro_network_endogenous.py` against the `config.prepare_aro_endogenous.yaml`
  build and the sector twin; `test/test_aro_methanol_chp.py` for the methanol CHP.

- **Fixed 2050 nuclear fleet** (`rules/aro.smk` `build_aro_nuclear_fleet` +
  `scripts/build_aro_nuclear_fleet.py` → committed `data/aro_nuclear_fleet_2050.csv`): the
  ARO runs' exogenous nuclear fleet, built from ppm unit by unit with the adopted rules
  (E1-E3, N; parent `memory/nuclear-treatment.md`). New units one step short of confirmed
  carry `Status == "pending"`. Consumed through a new `electricity.custom_powerplants_file`
  option (default: upstream's `data/custom_powerplants.csv`, so nothing else changes); the
  run must also drop ppm's nuclear rows via `powerplants_filter`, because custom rows are
  appended, never substituted. The same rule writes `data/aro_nuclear_p_max_pu.csv`
  (upstream's per-country availability plus a row for every fleet country without history,
  i.e. PL: per-year median over the run's modelled countries, UA excluded; `add_electricity` would otherwise leave it at 1.0), used via
  `conventional: nuclear: p_max_pu`. Tests: `test/test_build_aro_nuclear_fleet.py`.

- **Availability from ENTSO-E outages** (`conventional: unavailability`, meant to go upstream
  as a PR, so it is written upstream-shaped and edits upstream files): `retrieve.smk`
  `retrieve_entsoe_unavailability` (API per country and year, token from the environment,
  `data/versions.csv` row `entsoe_unavailability`, source `build`; busy windows are split
  because entsoe-py silently stops at 5000 documents), `build_electricity.smk`
  `build_entsoe_actual_generation` (hourly output per unit, 16.1.A, from monthly File
  Library files in `data/entsoe_actualgeneration/`: the API serves this one day per request)
  and `build_conventional_unavailability`, and `attach_unavailability` in `add_electricity.py`,
  fed through `input_conventional`. The cleaning is the hard part and each step fixes a
  measured defect: latest revision only, identical notices once (Paks republishes), the two
  reporting levels combined per plant (Ringhals), closed units dropped from
  `data/entsoe_unit_closures.csv` (hand-curated; the British list is stale since Brexit),
  losses capped by actual output and counted only in hours with output data (Leibstadt,
  Paks), and country-years kept only if >= 90% of their loss is in such hours and the loss
  is 0.5-1.1x the IAEA PRIS table's (Slovak reporting only from 2021, British output data
  ends 2021). The `historical` method maps ENTSO-E units to power plants through
  `data/entsoe_unit_mapping.csv` (empty so far) plus ppm's own ENTSO-E codes. Tests:
  `test/test_build_conventional_unavailability.py`.
- **Cherry-picked from upstream ahead of the base:** [#2387](https://github.com/PyPSA/pypsa-eur/pull/2387)
  (`d02406c4`, OSM network: AC-tagged relations no longer imported as DC links, GW ratings
  converted to MW, Overpass errors and timeouts handled, DC links in `make_network_comparison`).
  Needed for PyPSARO's `europe-partitioning` build, which builds the OSM network from raw data
  (`data: osm: source: build`) down to 30 kV. Only the release-note line for #2387 was taken;
  the other upstream notes in that hunk belong to commits not on this branch. On the next
  deliberate upstream update this commit is already contained and drops out.
- **`clean_osm_data.py`: route relations without a way member are dropped** before
  `_create_line`. Below 220 kV OSM has relations whose members are only nodes (GB
  `relation/18581906`, 33 kV), which crashed the build with `KeyError: ['geometry']`. Not yet
  upstream; a candidate for a PR, and on the next upstream update check whether upstream fixed it
  differently.
- **`clean_osm_data.py` `_clean_date`: dashes stripped from both ends, years outside 1-9999
  set to NaT.** Irish 38-110 kV assets tagged `start_date=mid-2000s` became the year -2000,
  which crashed the GeoJSON export (`'int' object has no attribute 'year'`). Same PR candidate.
- **`clean_osm_data.py`: member ways of DC relations (`frequency=0`) are dropped** like those of
  AC relations. Their ways often have no frequency tag, defaulted to 50 Hz and came in as an AC
  line next to the DC link: the Great Belt link (`relation/5487095`) appeared as a 400 kV AC line
  Funen-Zealand (`merged_way/207775098-400+2`), AC-coupling DK1 and DK2. The prebuilt archive 0.7
  does not have the duplicate. Same PR candidate.
- **`build_osm_network.py`: voltage-dependent station tolerance** (`osm_network_release:
  station_tolerance: {high_voltage, low_voltage, threshold}`; unset = one 500 m `BUS_TOL` for
  all, as upstream). With one 500 m buffer and single-linkage union, a build down to 30 kV
  chained sub-220 kV substations into stations up to 16.7 km wide (Upper Austria, Berlin
  8.6 km with 110 and 380 kV), with transformers between all their voltages and two 380 kV
  substations joined through 110 kV ones. With a threshold, `_create_station_seeds` forms
  stations in two passes (>= threshold with `high_voltage`, below with `low_voltage`), then
  `_attach_low_to_high_voltage_stations` merges each low-voltage station touching a
  high-voltage one into the nearest (by PoI) only; high-voltage polygons stay untouched, so the
  >= threshold grid does not depend on `low_voltage` (cutting attached stations out of a second
  high-voltage station instead left holes and cost AT 7% of its >= 220 kV line-km). AT/DE test
  (500/200 m, threshold 220): largest station 5.1 km, 96-97% of sub-220 kV line-km reach a
  hub (500 m everywhere: 98%); smaller `low_voltage` leaves line ends unconnected, 300 m chains
  again. `_extend_lines_to_buses` now draws lines that the station cut leaves empty (lying
  inside two adjacent stations) straight between their buses instead of crashing. PR candidate.
- **`base_network.py` `_remove_unconnected_components`: sparse adjacency from the branches**
  instead of `n.adjacency_matrix()`, which fills a dense bus x bus DataFrame even with
  `return_dataframe=False` (8 bytes per bus pair). A build down to 30 kV has ~52,000 buses:
  ~22 GB, OOM-killed (the first 30 kV build peaked at 15 GB). Now ~1 GB. Same components, same
  result. `simplify_network` and `cluster_network` still call `adjacency_matrix`; fine at
  transmission-only bus counts, a wall again if they ever see a sub-220 kV network. PR candidate.
- **Outage-island partition** (`rules/aro.smk`: `build_aro_partition` collects the per-country
  `build_aro_islands`, which in one pass sets k, builds the load grid, the islands, snaps them
  to municipalities (toggle `aro.partition.snap_to_lau`) and runs the copperplate check; config
  `aro.partition`). Ported from PyPSARO's
  prototype (`~/projects/partition-test`, `memory: outage-island-partitioning` in the parent
  repo) to run on a build with lower voltages (PyPSARO's `europe-partitioning` config); outputs
  in `resources/<run>/aro_partition/`. Differences to the prototype: load spread as the model
  spreads it (electricity as `build_electricity_demand_base`: Energy Atlas raster for EU27,
  DESNZ for GB, NUTS3 GDP/population otherwise; heat by NUTS3 population; weighted by the
  electricity and heat peaks) on 1 km points instead of NUTS3 0.6/0.4 through the 0.3 deg
  population layout (which is itself only NUTS3 population spread by area); municipalities from
  pypsa-eur's own `lau_regions` (GISCO LAU 2019) for every country instead of LAU 2021; no HVDC
  way filter (fixed in `clean_osm_data`); no Litgrid comparison (stays in the prototype).
  The per-country coincident peaks (electricity + electrical heat load, at the run's
  `clustering.temporal` resolution) that set k and the load weights are built from raw inputs
  by `build_aro_country_peaks`: upstream's heat rules reused via `use rule ... with:` on
  NUTS2 regions (`aro.partition.peak_regions`; `build_aro_peaks_*`, outputs in `aro_partition/peaks/`), combined with
  `prepare_aro_network`'s own heat functions. No network is clustered and no pre-run build is
  read, so the partition is reproducible from scratch and never circular; its demand settings
  must equal the pre-run's (PyPSARO's `pixi run check-partition-config`). Pick: least cut
  (`selection`; near-optimal balanced and medoid as options) within `size_bounds`
  [0.75, 1.33] x mean. The least cut drifts to a bound, so the bounds set the balance;
  Litgrid's [0.5, 1.5] is the sensitivity (Spain 17/17/7/7 GW, Berlin alone). Snapped islands rebalanced back into the
  bounds (`lau_respect_bounds`). No island spans a sea crossing: a land mass across the sea
  is its own island if it holds >= `separate_land_mass_min_gw` and the model has a link to it
  (>= 220 kV bus, DC landing, or a link in `data/transmission_projects/aro`), else it joins the
  nearest island (Gotland, Shetland, Zeeuws-Vlaanderen); below `size_bounds[0]` x the country's
  mean island it comes on top of k (Balearics, Corsica, Crete, Sicily). Needs `pymetis` and
  `unidecode`. `build_island_partition` (only defined with `aro.partition.export`) copies
  `islands.geojson` to that path; a pre-run reads it through `clustering:
  custom_busshapes_file`, with `mode: custom_busshapes` and `exact_busshapes: true`: `base_network` then cuts the bus regions
  along the islands (`build_custom_admin_shapes`), and `simplify_network`/`cluster_network`
  treat them like administrative regions (stubs removed within them, buses mapped per
  country), so the clustered regions equal the islands. Upstream `custom_busshapes` merges
  whole country-level Voronoi cells by bus location instead (median 15% area mismatch). An
  island without a bus gets a virtual one at its centroid (Crete, whose only link at or above
  220 kV, the Attica-Crete HVDC Ariadne, is missing from OSM and added through
  `transmission_projects.include.aro`).
- **`prepare_osm_network_release.py` memory:** the 30 kV `europe-partitioning` build was
  OOM-killed (26.5 GB) while *loading* `base.nc`: xarray decodes the line-geometry WKT strings
  into a fixed-width array as wide as the longest one (~95,000 characters x 47,100 lines
  ~ 18 GB). `load_network_lean` reads the `*_geometry` variables (except `shapes`) with netCDF4
  instead: 0.8 GB, 2 s; the whole job now runs in ~35 s. Any other script that loads a network
  with full-resolution OSM geometries will hit the same wall. Smaller map changes alongside:
  line-shape simplification on the map is configurable (`osm_network_release:
  map_line_simplify`, default 30 m = upstream), map coordinates are rounded to 5 decimals
  (~1 m), and the view centre comes from the bounds. The CSV exports are unchanged.
- **Custom data versions** (`data/custom_versions.csv`, the soft-fork route of
  `doc/data_sources.md`): extra rows appended to `data/versions.csv` through
  `data.version_files` in a run's config, so upstream's table stays untouched. Currently one
  row: technology-data `costs` pinned to the commit of PR #291 (nuclear fuel price fix), used
  by PyPSARO's `europe-pre-run` config. Tagged `supported` only, never `latest`, so runs on
  `version: latest` are unaffected. Replace with the release once it contains the fix.

- **Shared resources stay inside the run prefix** (`Snakefile` `SHARED_DIR`, `shared_dir` in
  `scripts/_helpers.py` `get_run_path`): with `run.shared_resources.policy: base` (or `true`)
  upstream writes shared files to the top level of `resources/`, shared across EVERY prefix,
  so two configs with different clustering silently overwrite each other's
  `networks/clustered.nc`. Here they go to `resources/{prefix}/`. No prefix -> upstream
  behaviour. `base` also needs `exclude: ["costs_"]` (compose_network asks for the shared
  cost path, process_cost_data writes the per-run one). Test:
  `test/test_shared_resources_prefix.py`.

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
