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
