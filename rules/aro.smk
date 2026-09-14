# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# Fork-specific rules for the PyPSARO ARO/C&CG resilience workflow. Kept in their own file,
# additive to upstream: a new file re-applies without conflict when this branch is reset onto a
# newer upstream tag, whereas edits inside prepare_network.py / prepare_sector_network.py would
# collide whenever upstream touches those central scripts.


rule prepare_aro_network:
    input:
        network=resources("networks/base_s_{clusters}_elec_{opts}.nc"),
        hourly_heat_demand_total=resources(
            "hourly_heat_demand_total_base_s_{clusters}.nc"
        ),
        cop_profiles=resources("cop_profiles_base_s_{clusters}_{planning_horizons}.nc"),
        pop_weighted_energy_totals=resources(
            "pop_weighted_energy_totals_s_{clusters}.csv"
        ),
        pop_weighted_heat_totals=resources("pop_weighted_heat_totals_s_{clusters}.csv"),
        heating_efficiencies=resources("heating_efficiencies.csv"),
        district_heat_share=resources(
            "district_heat_share_base_s_{clusters}_{planning_horizons}.csv"
        ),
        pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
        costs=resources("costs_{planning_horizons}_processed.csv"),
    output:
        network=resources(
            "networks/base_s_{clusters}_elec_{opts}_aro_{planning_horizons}.nc"
        ),
    log:
        logs(
            "prepare_aro_network_base_s_{clusters}_elec_{opts}_{planning_horizons}.log"
        ),
    benchmark:
        benchmarks(
            "prepare_aro_network/base_s_{clusters}_elec_{opts}_{planning_horizons}"
        )
    threads: 1
    resources:
        mem_mb=8000,
    params:
        aro_heat=config_provider("aro", "heat"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
        sector=config_provider("sector"),
    message:
        "Adding exogenous electrified-heat load to base_s_{wildcards.clusters}_elec_{wildcards.opts} for {wildcards.planning_horizons}"
    script:
        scripts("prepare_aro_network.py")


rule prepare_aro_networks:
    input:
        expand(
            resources(
                "networks/base_s_{clusters}_elec_{opts}_aro_{planning_horizons}.nc"
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting ARO-prepared network files"
