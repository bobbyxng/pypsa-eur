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


# The fixed 2050 nuclear fleet as a custom powerplant list (rules E1-E3/N in the script's
# docstring). Writes into data/ and is committed, so the fleet is reviewable row by row and
# frozen against later ppm releases. A run uses it via `electricity.custom_powerplants_file`
# plus a `powerplants_filter` that drops ppm's own nuclear rows, and the availability table
# via `conventional: nuclear: p_max_pu` (adds PL, which upstream's table lacks and would
# leave at 1.0). memory: nuclear-treatment
rule build_aro_nuclear_fleet:
    input:
        powerplants=rules.retrieve_powerplants.output["powerplants"],
        p_max_pu="data/nuclear_p_max_pu.csv",
    output:
        fleet="data/aro_nuclear_fleet_2050.csv",
        p_max_pu="data/aro_nuclear_p_max_pu.csv",
    log:
        "logs/build_aro_nuclear_fleet.log",
    threads: 1
    resources:
        mem_mb=6000,
    params:
        # The availability median for countries without history runs over these only.
        countries=config_provider("countries"),
    message:
        "Building the fixed 2050 nuclear fleet for the ARO runs"
    script:
        scripts("build_aro_nuclear_fleet.py")


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


# Same hand-copy pattern as the calibration above: reads a SOLVED sector-coupled network
# (PyPSARO's europe-heat-reference run) and writes per-country heat-pump / resistive shares
# for `aro.heat.shares`. Not read by any rule. memory: heat-electrification-design
rule extract_electrified_heat_share:
    input:
        network=RESULTS + "networks/solved_{horizon}.nc",
    output:
        shares=RESULTS + "heat_shares/electrified_heat_shares_{horizon}.json",
    log:
        logs("extract_electrified_heat_share_{horizon}.log"),
    threads: 1
    resources:
        mem_mb=8000,
    message:
        "Extracting heat-pump / resistive heat shares for {wildcards.horizon}"
    script:
        scripts("extract_electrified_heat_shares.py")


rule extract_electrified_heat_shares:
    input:
        expand(
            RESULTS + "heat_shares/electrified_heat_shares_{horizon}.json",
            horizon=config["planning_horizons"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting electrified heat shares"


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
        # Produced by `build_salt_cavern_potentials` in build_sector.smk, which is NOT gated
        # on `sector.enabled` -- same property that lets this rule read the heat resources
        # from an electricity-only run. Without it, carrier `H2` keeps attach_stores' uncapped
        # *underground* cost at every node (dump/todos.md sec -3).
        h2_cavern=resources("salt_cavern_potentials.csv"),
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
        aro_h2_turbine=config_provider("aro", "h2_turbine"),
        aro_methanol=config_provider("aro", "methanol"),
        aro_dc_losses=config_provider("aro", "dc_losses"),
        aro_fix_cross_border_transmission=config_provider(
            "aro", "fix_cross_border_transmission"
        ),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
        sector=config_provider("sector"),
        # The input network is already temporally aggregated (compose_network folds that in),
        # so the hourly heat inputs must be collapsed with the SAME operator upstream used --
        # mean for averaging/segmentation, point-sample for representative. Inferring it from
        # the snapshot index instead silently broke on drop_leap_day years.
        # memory: aro-heat-temporal-alignment
        clustering_temporal=config_provider("clustering", "temporal"),
    message:
        "Preparing ARO network for {wildcards.horizon} (heat load, H2 storage/turbines, optional methanol)"
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


# ---- Outage-island partition (PyPSARO: memory: outage-island-partitioning) ------------------
# Islands = model nodes per country, formed from supply areas around the >=220/<220 kV hubs on
# the OSM grid down to 30 kV, borders drawn where the grid below 220 kV is weakest, snapped to
# municipalities (LAU). Runs on a build with lower voltages (PyPSARO's europe-partitioning
# config); the result is promoted to data/busshapes/ by hand and read by a pre-run through
# `clustering: mode: custom_busshapes`. Kept out of the pre-run's DAG on purpose: a partition
# is a frozen design decision, it must not change silently with upstream inputs.


def aro_partition_countries(w):
    exclude = set(config_provider("aro", "partition", "exclude")(w))
    return [c for c in config_provider("countries")(w) if c not in exclude]


# ---- Country peaks from raw inputs: the heat chain on regions independent of the islands ----
# The peaks that size the partition (k per country, electricity/heat weights) are built here
# from scratch, never read from a pre-run build. Upstream's heat rules are reused unchanged on
# NUTS regions (`aro.partition.peak_regions`) instead of the network's regions (they read
# nothing but the region shapes),
# writing to aro_partition/peaks/, so no network is clustered and the islands never feed back.
# The demand settings (aro.heat, district heating, snapshots, clustering.temporal, costs) must
# equal the pre-run's: PyPSARO's `pixi run check-partition-config` checks that.
ARO_PEAKS = "aro_partition/peaks/"


def aro_partition_horizon(w):
    horizons = config_provider("planning_horizons")(w)
    return str(horizons[0] if isinstance(horizons, list) else horizons)


rule build_aro_peak_regions:
    input:
        country_shapes=resources("country_shapes.geojson"),
        nuts3_shapes=resources("nuts3_shapes.geojson"),
    output:
        regions=resources(ARO_PEAKS + "regions.geojson"),
    params:
        countries=config_provider("countries"),
        level=config_provider("aro", "partition", "peak_regions"),
    run:
        import geopandas as gpd

        if params.level == "country":
            regions = gpd.read_file(input.country_shapes)
            regions = regions[regions["name"].isin(params.countries)]
        else:
            # NUTS codes start with the country (GR, GB here), which the heat chain reads
            # as the region's country (`name[:2]`)
            nuts = gpd.read_file(input.nuts3_shapes)
            nuts = nuts[nuts.country.isin(params.countries)]
            col = {"nuts1": "level1", "nuts2": "level2", "nuts3": "level3"}[
                params.level
            ]
            regions = nuts[[col, "country", "geometry"]].dissolve(col).reset_index()
            regions = regions.rename(columns={col: "name"})
            bad = regions["name"].str[:2] != regions["country"]
            if bad.any():
                raise ValueError(
                    f"NUTS codes not prefixed by country: {list(regions.name[bad])}"
                )
        regions[["name", "geometry"]].to_file(output.regions)


use rule build_clustered_population_layouts as build_aro_peaks_pop_layout with:
    input:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
        onshore_regions=resources(ARO_PEAKS + "regions.geojson"),
        cutout=lambda w: input_cutout(w),
    output:
        clustered_pop_layout=resources(ARO_PEAKS + "pop_layout.csv"),
    log:
        logs("aro_peaks/build_clustered_population_layouts.log"),
    benchmark:
        benchmarks("aro_peaks/build_clustered_population_layouts")


use rule build_daily_heat_demand as build_aro_peaks_daily_heat_demand with:
    input:
        pop_layout=resources("pop_layout_total.nc"),
        onshore_regions=resources(ARO_PEAKS + "regions.geojson"),
        cutout=lambda w: input_cutout(
            w, config_provider("sector", "heat_demand_cutout")(w)
        ),
    output:
        heat_demand=resources(ARO_PEAKS + "daily_heat_demand_total.nc"),
    log:
        logs("aro_peaks/build_daily_heat_demand_total.log"),
    benchmark:
        benchmarks("aro_peaks/build_daily_heat_demand_total")


use rule build_hourly_heat_demand as build_aro_peaks_hourly_heat_demand with:
    input:
        heat_profile="data/heat_load_profile_BDEW.csv",
        heat_demand=resources(ARO_PEAKS + "daily_heat_demand_total.nc"),
    output:
        heat_demand=resources(ARO_PEAKS + "hourly_heat_demand_total.nc"),
        heat_dsm_profile=resources(ARO_PEAKS + "residential_heat_dsm_profile.csv"),
    log:
        logs("aro_peaks/build_hourly_heat_demand_total.log"),
    benchmark:
        benchmarks("aro_peaks/build_hourly_heat_demand_total")


use rule build_temperature_profiles as build_aro_peaks_temperature_profiles with:
    input:
        pop_layout=resources("pop_layout_total.nc"),
        onshore_regions=resources(ARO_PEAKS + "regions.geojson"),
        cutout=lambda w: input_cutout(
            w, config_provider("sector", "heat_demand_cutout")(w)
        ),
    output:
        temp_soil=resources(ARO_PEAKS + "temp_soil_total.nc"),
        temp_air=resources(ARO_PEAKS + "temp_air_total.nc"),
    log:
        logs("aro_peaks/build_temperature_profiles_total.log"),
    benchmark:
        benchmarks("aro_peaks/build_temperature_profiles_total")


use rule build_central_heating_temperature_profiles as build_aro_peaks_central_heating_temperature_profiles with:
    input:
        temp_air_total=resources(ARO_PEAKS + "temp_air_total.nc"),
        onshore_regions=resources(ARO_PEAKS + "regions.geojson"),
    output:
        central_heating_forward_temperature_profiles=resources(
            ARO_PEAKS + "central_heating_forward_temperature_profiles_{horizon}.nc"
        ),
        central_heating_return_temperature_profiles=resources(
            ARO_PEAKS + "central_heating_return_temperature_profiles_{horizon}.nc"
        ),
    log:
        logs("aro_peaks/build_central_heating_temperature_profiles_{horizon}.log"),
    benchmark:
        benchmarks("aro_peaks/build_central_heating_temperature_profiles_{horizon}")


use rule build_ptes_operations as build_aro_peaks_ptes_operations with:
    input:
        central_heating_forward_temperature_profiles=resources(
            ARO_PEAKS + "central_heating_forward_temperature_profiles_{horizon}.nc"
        ),
        central_heating_return_temperature_profiles=resources(
            ARO_PEAKS + "central_heating_return_temperature_profiles_{horizon}.nc"
        ),
        onshore_regions=resources(ARO_PEAKS + "regions.geojson"),
    output:
        ptes_direct_utilisation_profiles=resources(
            ARO_PEAKS + "ptes_direct_utilisation_profiles_{horizon}.nc"
        ),
        ptes_top_temperature_profiles=resources(
            ARO_PEAKS + "ptes_top_temperature_profiles_{horizon}.nc"
        ),
        ptes_e_max_pu_profiles=resources(
            ARO_PEAKS + "ptes_e_max_pu_profiles_{horizon}.nc"
        ),
    log:
        logs("aro_peaks/build_ptes_operations_{horizon}.log"),
    benchmark:
        benchmarks("aro_peaks/build_ptes_operations_{horizon}")


def input_aro_peaks_heat_source_temperature(w):
    # upstream's input_heat_source_temperature, pointed at the country-region files
    return {
        k: v.replace(resources(""), resources(ARO_PEAKS))
        for k, v in input_heat_source_temperature(w).items()
    }


use rule build_cop_profiles as build_aro_peaks_cop_profiles with:
    input:
        unpack(input_aro_peaks_heat_source_temperature),
        central_heating_forward_temperature_profiles=resources(
            ARO_PEAKS + "central_heating_forward_temperature_profiles_{horizon}.nc"
        ),
        central_heating_return_temperature_profiles=resources(
            ARO_PEAKS + "central_heating_return_temperature_profiles_{horizon}.nc"
        ),
        temp_soil_total=resources(ARO_PEAKS + "temp_soil_total.nc"),
        temp_air_total=resources(ARO_PEAKS + "temp_air_total.nc"),
        temp_ptes_total=resources(
            ARO_PEAKS + "ptes_top_temperature_profiles_{horizon}.nc"
        ),
        onshore_regions=resources(ARO_PEAKS + "regions.geojson"),
    output:
        cop_profiles=resources(ARO_PEAKS + "cop_profiles_{horizon}.nc"),
    log:
        logs("aro_peaks/build_cop_profiles_{horizon}.log"),
    benchmark:
        benchmarks("aro_peaks/build_cop_profiles_{horizon}")


use rule build_population_weighted_energy_totals as build_aro_peaks_population_weighted_energy_totals with:
    input:
        energy_totals=resources("{kind}_totals.csv"),
        clustered_pop_layout=resources(ARO_PEAKS + "pop_layout.csv"),
    output:
        resources(ARO_PEAKS + "pop_weighted_{kind}_totals.csv"),
    log:
        logs("aro_peaks/build_population_weighted_{kind}_totals.log"),
    benchmark:
        benchmarks("aro_peaks/build_population_weighted_{kind}_totals")


use rule build_district_heat_share as build_aro_peaks_district_heat_share with:
    input:
        district_heat_share=resources("district_heat_share.csv"),
        clustered_pop_layout=resources(ARO_PEAKS + "pop_layout.csv"),
    output:
        district_heat_share=resources(ARO_PEAKS + "district_heat_share_{horizon}.csv"),
    log:
        logs("aro_peaks/build_district_heat_share_{horizon}.log"),
    benchmark:
        benchmarks("aro_peaks/build_district_heat_share_{horizon}")


rule build_aro_peaks_snapshots:
    output:
        network=resources(ARO_PEAKS + "snapshots.nc"),
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    run:
        # process_cost_data reads a network only for its modelled years (snapshot weightings);
        # an empty one with the run's hourly snapshots keeps the 30 kV network from being
        # simplified (dense adjacency, memory wall) just to annualise costs.
        import pypsa
        from scripts._helpers import get_snapshots

        n = pypsa.Network()
        n.set_snapshots(get_snapshots(params.snapshots, params.drop_leap_day))
        n.export_to_netcdf(output.network)


use rule process_cost_data as build_aro_peaks_costs with:
    input:
        network=resources(ARO_PEAKS + "snapshots.nc"),
        costs=lambda w: rules.retrieve_cost_data.output["costs"].format(
            horizon=cost_year(w)
        ),
        custom_costs=config_provider("costs", "custom_cost_fn"),
    output:
        resources(ARO_PEAKS + "costs_{horizon}_processed.csv"),
    log:
        logs("aro_peaks/build_cost_data_{horizon}.log"),
    benchmark:
        benchmarks("aro_peaks/build_cost_data_{horizon}")


rule build_aro_country_peaks:
    input:
        electricity_demand=resources("electricity_demand.csv"),
        hourly_heat_demand_total=resources(ARO_PEAKS + "hourly_heat_demand_total.nc"),
        cop_profiles=lambda w: resources(
            ARO_PEAKS + f"cop_profiles_{aro_partition_horizon(w)}.nc"
        ),
        pop_weighted_energy_totals=resources(
            ARO_PEAKS + "pop_weighted_energy_totals.csv"
        ),
        pop_weighted_heat_totals=resources(ARO_PEAKS + "pop_weighted_heat_totals.csv"),
        heating_efficiencies=resources("heating_efficiencies.csv"),
        district_heat_share=lambda w: resources(
            ARO_PEAKS + f"district_heat_share_{aro_partition_horizon(w)}.csv"
        ),
        pop_layout=resources(ARO_PEAKS + "pop_layout.csv"),
        costs=lambda w: resources(
            ARO_PEAKS + f"costs_{aro_partition_horizon(w)}_processed.csv"
        ),
    output:
        peaks=resources("aro_partition/country_peaks.csv"),
    log:
        logs("build_aro_country_peaks.log"),
    threads: 1
    resources:
        mem_mb=8000,
    params:
        countries=config_provider("countries"),
        horizon=aro_partition_horizon,
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        load_scaling=config_provider("load", "scaling_factor"),
        aro_heat=config_provider("aro", "heat"),
        sector=config_provider("sector"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
        clustering_temporal=config_provider("clustering", "temporal"),
    message:
        "Coincident electricity + electrical heat peak per country"
    script:
        scripts("build_aro_country_peaks.py")


rule build_aro_islands:
    input:
        peaks=resources("aro_partition/country_peaks.csv"),
        buses=resources("osm/release/buses.csv"),
        lines=resources("osm/release/lines.csv"),
        transformers=resources("osm/release/transformers.csv"),
        converters=resources("osm/release/converters.csv"),
        added_links="data/transmission_projects/aro/new_links.csv",
        country_shapes=resources("country_shapes.geojson"),
        nuts3_shapes=resources("nuts3_shapes.geojson"),
        raster=rules.retrieve_electricity_demand_energy_atlas.output["tif"],
        gb_excel=rules.retrieve_desnz_electricity_consumption.output["xlsx"],
        gb_geojson=rules.retrieve_ons_lad.output["geojson"],
        lau=rules.retrieve_lau_regions.output["zip"],
    output:
        islands=resources("aro_partition/{country}/islands.geojson"),
        islands_grid=resources("aro_partition/{country}/islands_grid.geojson"),
        supply_areas=resources("aro_partition/{country}/supply_areas.csv"),
        lau_assignment=resources("aro_partition/{country}/lau_assignment.csv"),
        copperplate=resources("aro_partition/{country}/copperplate.csv"),
        map=resources("aro_partition/{country}/map.png"),
        map_grid=resources("aro_partition/{country}/map_grid.png"),
        summary=resources("aro_partition/{country}/summary.md"),
    log:
        logs("build_aro_islands_{country}.log"),
    threads: 1
    resources:
        mem_mb=8000,
    params:
        countries=aro_partition_countries,
        partition=config_provider("aro", "partition"),
        distribution_key=config_provider("load", "distribution_key"),
    message:
        "Outage islands of {wildcards.country}"
    script:
        scripts("build_aro_islands.py")


rule build_aro_partition:
    input:
        peaks=resources("aro_partition/country_peaks.csv"),
        islands=lambda w: expand(
            resources("aro_partition/{country}/islands.geojson"),
            country=aro_partition_countries(w),
        ),
        copperplate=lambda w: expand(
            resources("aro_partition/{country}/copperplate.csv"),
            country=aro_partition_countries(w),
        ),
        country_shapes=resources("country_shapes.geojson"),
    output:
        islands=resources("aro_partition/islands.geojson"),
        summary=resources("aro_partition/summary.csv"),
    log:
        logs("build_aro_partition.log"),
    threads: 1
    resources:
        mem_mb=4000,
    params:
        countries=aro_partition_countries,
        partition=config_provider("aro", "partition"),
    message:
        "Collecting the outage islands of all countries"
    script:
        scripts("build_aro_partition.py")
