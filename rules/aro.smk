# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# Fork-specific rules for the PyPSARO ARO/C&CG resilience workflow. Kept in their own file,
# additive to upstream: a new file re-applies without conflict when this branch is merged onto a
# newer upstream tag, whereas edits inside compose_network.py / prepare_sector_network.py would
# collide whenever upstream touches those central scripts.
#
# Paths follow the streamlined workflow (upstream #1838): the only wildcard left here is
# {horizon}; {clusters}/{opts}/{sector_opts} are gone and scenario content lives in the config.
# The input network is the *composed* one (compose_network folds the former add_electricity +
# prepare_network chain into one rule), so `aro` is a post-compose, pre-solve layer.


# The manifest, not `directory("data/fallahnejad")`, is the declared output: Snakemake
# wipes a directory output before re-running, which would discard 382 MB on every retry
# and defeat the script's own skip-what-exists resume.
rule retrieve_fallahnejad_dh:
    output:
        manifest="data/fallahnejad/manifest.json",
    log:
        "logs/retrieve_fallahnejad_dh.log",
    retries: 2
    threads: 1
    resources:
        mem_mb=6000,
    message:
        "Retrieving Fallahnejad et al. (2024) district-heating potential data"
    script:
        scripts("retrieve_fallahnejad_dh.py")


# Not part of any default target, and its output is copied into the config by hand rather
# than read by a rule: the config stays a static, reviewable artifact and no ordinary run
# depends on the 400 MB retrieval above.
rule calibrate_district_heating_potential:
    input:
        fallahnejad="data/fallahnejad/manifest.json",
        pop_layout=resources("pop_layout.csv"),
        pop_weighted_energy_totals=resources("pop_weighted_energy_totals.csv"),
        pop_weighted_heat_totals=resources("pop_weighted_heat_totals.csv"),
        heating_efficiencies=resources("heating_efficiencies.csv"),
        district_heat_share=resources("district_heat_share.csv"),
    output:
        potential=resources("dh_potential_calibrated_{horizon}.json"),
    log:
        logs("calibrate_district_heating_potential_{horizon}.log"),
    threads: 1
    resources:
        mem_mb=6000,
    params:
        energy_totals_year=config_provider("energy", "energy_totals_year"),
        sector=config_provider("sector"),
    message:
        "Calibrating district-heating potential to Fallahnejad et al. (2024) for {wildcards.horizon}"
    script:
        scripts("calibrate_district_heating_potential.py")


rule prepare_aro_network:
    input:
        network=resources("networks/composed_{horizon}.nc"),
        hourly_heat_demand_total=resources("hourly_heat_demand_total.nc"),
        cop_profiles=resources("cop_profiles_{horizon}.nc"),
        pop_weighted_energy_totals=resources("pop_weighted_energy_totals.csv"),
        pop_weighted_heat_totals=resources("pop_weighted_heat_totals.csv"),
        heating_efficiencies=resources("heating_efficiencies.csv"),
        district_heat_share=resources("district_heat_share_{horizon}.csv"),
        pop_layout=resources("pop_layout.csv"),
        costs=resources("costs_{horizon}_processed.csv"),
    output:
        network=resources("networks/composed_aro_{horizon}.nc"),
    log:
        logs("prepare_aro_network_{horizon}.log"),
    benchmark:
        benchmarks("prepare_aro_network/{horizon}")
    threads: 1
    resources:
        mem_mb=8000,
    params:
        aro_heat=config_provider("aro", "heat"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
        sector=config_provider("sector"),
        # The input network is already temporally aggregated (compose_network folds that in),
        # so the hourly heat inputs must be collapsed with the SAME operator upstream used --
        # mean for averaging/segmentation, point-sample for representative. Inferring it from
        # the snapshot index instead silently broke on drop_leap_day years.
        # memory: aro-heat-temporal-alignment
        clustering_temporal=config_provider("clustering", "temporal"),
    message:
        "Adding exogenous electrified-heat load to composed network for {wildcards.horizon}"
    script:
        scripts("prepare_aro_network.py")


rule prepare_aro_networks:
    input:
        expand(
            resources("networks/composed_aro_{horizon}.nc"),
            horizon=config["planning_horizons"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting ARO-prepared network files"
