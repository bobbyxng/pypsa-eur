# Validation of islanded networks
## Import and helpers
### Import core packages
import logging
import numpy as np
import pandas as pd
import pypsa
import plotly.graph_objects as go
logger = logging.getLogger(__name__)

### Helpers
def extract_capacity(n: pypsa.Network) -> pd.DataFrame:
    """Extract a tidy table of optimal capacities for all extendable assets."""
    df = n.statistics.optimal_capacity(nice_names=False).reset_index()
    df.columns = ["component", "name", "capacity"]
    return df


def extract_energy_balance(
    n: pypsa.Network,
    bus_carrier="AC",
    ) -> pd.DataFrame:
    """Extract a tidy table of energy balance statistics."""
    df = n.statistics.energy_balance(nice_names=False, bus_carrier=bus_carrier)
    df.rename("energy", inplace=True)
    return df.reset_index()


def extract_energy_balance_time(
    n: pypsa.Network,
    bus_carrier="AC",
    ) -> pd.DataFrame:
    """Extract a tidy table of energy balance statistics."""
    df = n.statistics.energy_balance(nice_names=False, groupby_time=False, bus_carrier=bus_carrier)
    return df.reset_index()


def calculate_total_costs(n: pypsa.Network, exclude_grid: bool = False) -> float:
    """Calculate total costs (capex + opex), optionally excluding AC lines and DC links."""
    costs = pd.concat([n.statistics.capex(nice_names=False), n.statistics.opex(nice_names=False)], axis=1, keys=["capex", "opex"])
    if exclude_grid:
        costs = costs[~(
            ((costs.index.get_level_values("component") == "Line") &
             (costs.index.get_level_values("carrier") == "AC")) |
            ((costs.index.get_level_values("component") == "Link") &
             (costs.index.get_level_values("carrier") == "DC"))
        )]
    return costs.fillna(0).sum().sum()


def create_bar_plot(
    pivot_df, 
    x_col, 
    y_cols, 
    labels, 
    title, 
    yaxis_title,
    height=600,
    width=800,
):
    """Create a grouped bar plot with multiple scenarios."""
    fig = go.Figure()
    for y_col, label in zip(y_cols, labels):
        fig.add_trace(go.Bar(
            x=pivot_df[x_col], 
            y=pivot_df[y_col], 
            name=label,
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_col.title(),
        yaxis_title=yaxis_title,
        barmode="group",
        hovermode="x unified",
        template="plotly_white",
        height=height,
        width=width,
    )
    return fig


def create_stacked_bar_plot(
    df, 
    scenario_col, 
    value_col, 
    carrier_col, 
    carrier_colors, 
    scenario_labels, 
    title, 
    yaxis_title,
    height=800,
    width=600,
):
    """Create a stacked bar plot with carriers colored by carrier_colors dict.
    
    Positive values are stacked upward (generation), negative values downward (demand).
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-form dataframe with scenario, carrier, and value columns
    scenario_col : str
        Column name containing scenario identifiers
    value_col : str
        Column name containing values to plot
    carrier_col : str
        Column name containing carrier names
    carrier_colors : dict
        Dictionary mapping carrier names to colors
    scenario_labels : dict
        Dictionary mapping scenario identifiers to display labels
    title : str
        Plot title
    yaxis_title : str
        Y-axis label
    """
    fig = go.Figure()
    
    # Define the order of scenarios
    scenario_order = ["grid", "maxpu0", "removed", "outage_summer", "outage_winter", "stochastic"]
    
    # Get unique carriers
    carriers = df[carrier_col].unique()
    
    # Create a complete dataframe with all scenario-carrier combinations
    # Group by scenario and carrier to aggregate any duplicates
    df_grouped = df.groupby([scenario_col, carrier_col], as_index=False)[value_col].sum()
    
    # Separate positive and negative values
    for carrier in carriers:
        carrier_data = df_grouped[df_grouped[carrier_col] == carrier].copy()
        
        # Split into positive and negative
        positive_data = carrier_data[carrier_data[value_col] >= 0].copy()
        negative_data = carrier_data[carrier_data[value_col] < 0].copy()
        
        color = carrier_colors.get(carrier, "#cccccc")
        
        # Add positive values (generation)
        if not positive_data.empty:
            # Ensure all scenarios are present
            x_vals = []
            y_vals = []
            for scenario in scenario_order:
                x_vals.append(scenario_labels.get(scenario, scenario))
                matching = positive_data[positive_data[scenario_col] == scenario]
                if not matching.empty:
                    y_vals.append(matching[value_col].values[0])
                else:
                    y_vals.append(0)
            
            fig.add_trace(go.Bar(
                name=carrier,
                x=x_vals,
                y=y_vals,
                marker=dict(color=color),
                legendgroup=carrier,
                showlegend=True,
            ))
        
        # Add negative values (demand)
        if not negative_data.empty:
            # Ensure all scenarios are present
            x_vals = []
            y_vals = []
            for scenario in scenario_order:
                x_vals.append(scenario_labels.get(scenario, scenario))
                matching = negative_data[negative_data[scenario_col] == scenario]
                if not matching.empty:
                    y_vals.append(matching[value_col].values[0])
                else:
                    y_vals.append(0)
            
            fig.add_trace(go.Bar(
                name=carrier,
                x=x_vals,
                y=y_vals,
                marker=dict(color=color),
                legendgroup=carrier,
                showlegend=True,
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title=yaxis_title,
        barmode="relative",  # This stacks positive up and negative down
        hovermode="x unified",
        template="plotly_white",
        height=height,
        width=width,
    )
    
    return fig


def create_stacked_area_plot(
    df, 
    time_col,
    scenario_col, 
    value_col, 
    carrier_col, 
    carrier_colors, 
    scenario_labels, 
    title, 
    yaxis_title,
    height=600,
    width=2000,
):
    """Create stacked area plots over time for multiple scenarios in subplots.
    
    Positive values are stacked upward (generation), negative values downward (demand).
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-form dataframe with time, scenario, carrier, and value columns
    time_col : str
        Column name containing time/snapshot identifiers
    scenario_col : str
        Column name containing scenario identifiers
    value_col : str
        Column name containing values to plot
    carrier_col : str
        Column name containing carrier names
    carrier_colors : dict
        Dictionary mapping carrier names to colors
    scenario_labels : dict
        Dictionary mapping scenario identifiers to display labels
    title : str
        Plot title
    yaxis_title : str
        Y-axis label
    height : int
        Height of the figure in pixels
    width : int
        Width of the figure in pixels
    """
    from plotly.subplots import make_subplots
    
    # Define the order of scenarios
    scenario_order = ["grid", "maxpu0", "removed", "outage_summer", "outage_winter", "stochastic"]
    n_scenarios = len(scenario_order)
    
    # Create subplots - one column per scenario
    fig = make_subplots(
        rows=1, 
        cols=n_scenarios,
        subplot_titles=[scenario_labels.get(s, s) for s in scenario_order],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )
    
    # Get unique carriers
    carriers = sorted([c for c in df[carrier_col].unique() if c in carrier_colors])
    
    # Ensure time column is datetime type for consistent merging
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Get all unique time points across all scenarios (sorted)
    all_times = sorted(df[time_col].unique())
    
    # Process each scenario
    for col_idx, scenario in enumerate(scenario_order, start=1):
        scenario_data = df[df[scenario_col] == scenario].copy()
        
        if scenario_data.empty:
            continue
        
        # Process each carrier
        for carrier in carriers:
            carrier_data = scenario_data[scenario_data[carrier_col] == carrier].copy()
            
            if carrier_data.empty:
                continue
            
            # Create a complete time series with all time points
            # This ensures proper stacking and hover alignment
            complete_series = pd.DataFrame({time_col: pd.to_datetime(all_times)})
            carrier_data = complete_series.merge(
                carrier_data[[time_col, value_col]], 
                on=time_col, 
                how='left'
            ).fillna(0)
            
            # Sort by time
            carrier_data = carrier_data.sort_values(time_col)
            
            # Get values
            x_vals = carrier_data[time_col]
            y_vals = carrier_data[value_col]
            
            color = carrier_colors.get(carrier, "#cccccc")
            showlegend = True # (col_idx == 1)
            
            # Split into positive and negative by creating separate series
            y_positive = y_vals.where(y_vals >= 0, 0)
            y_negative = y_vals.where(y_vals < 0, 0)
            
            # Add positive trace if there are any positive values
            if (y_positive != 0).any():
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_positive,
                        name=carrier,
                        mode='lines',
                        line=dict(width=0),
                        stackgroup='positive',
                        fillcolor=color,
                        legendgroup=carrier,
                        showlegend=showlegend,
                        hovertemplate=f'{carrier}<br>%{{y:.2f}}<extra></extra>',
                    ),
                    row=1, col=col_idx
                )
            
            # Add negative trace if there are any negative values
            if (y_negative != 0).any():
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_negative,
                        name=carrier,
                        mode='lines',
                        line=dict(width=0),
                        stackgroup='negative',
                        fillcolor=color,
                        legendgroup=carrier,
                        showlegend=showlegend,
                        hovertemplate=f'{carrier}<br>%{{y:.2f}}<extra></extra>',
                    ),
                    row=1, col=col_idx
                )
    
    # Update layout
    fig.update_layout(
        height=height,
        width=width,
        template="plotly_white",
        hovermode="x unified",
        title_text=title,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Update all x-axes
    for i in range(1, n_scenarios + 1):
        fig.update_xaxes(title_text="Time", row=1, col=i)
    
    # Update y-axis only for the first subplot

    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)    
    
    return fig

## Data prep
### Load the base network, adjust loads, and add load-shedding generators.

n_elec_path = "resources/resilient-islands/networks/base_s_128_elec_.nc"
solver_name = "gurobi"
solver_options = {"OutputFlag": 0}  # minimize solver log output

# Load network
n = pypsa.Network(n_elec_path)

# Base styling and carriers
n.carriers.loc["", "color"] = "#aaaaaa"
n.add("Carrier", "load shedding", color="#aa0000")

# Increase load
n.loads_t.p_set *= 1.7

# Add load-shedding generators
spatial_nodes = n.buses.index.tolist()
ls_names = [f"{bus} load shedding" for bus in spatial_nodes]

n.add(
    "Generator",
    pd.Index(ls_names),
    bus=pd.Series(spatial_nodes, index=ls_names),
    p_nom_extendable=True,
    p_nom_max=np.inf,
    capital_cost=0.1,
    marginal_cost=10000.0,
    carrier="load shedding",
)

# Resample
n = n.cluster.temporal.resample("168h")

## Optimisation
### Prepare scenario copies for baseline and islanded configurations. Run optimisations for baseline, s_max_pu = 0 / p_max_pu = 0, and fully islanded cases.

#### N1
print("Running baseline optimisation...")
n1 = n.copy()
n1.optimize(solver_name=solver_name, solver_options=solver_options)

#### N2
print("Running islanded optimisation (s_max_pu = 0 and p_max_pu = 0)...")
n2 = n.copy()
if not n.lines.empty:
    n2.lines["s_max_pu"] = 0
if not n2.links.empty:
    dc_links = n2.links[n2.links.carrier == "DC"].index
    n2.links.loc[dc_links, ["p_max_pu", "p_min_pu"]] = 0

n2.optimize(solver_name=solver_name, solver_options=solver_options)

#### N3
print("Running islanded optimisation (lines and links removed)...")
n3 = n.copy()
if not n.lines.empty:
    n3.remove("Line", n3.lines.index)
if not n3.links.empty:
    n3.remove("Link", n3.links[n3.links.carrier == "DC"].index)

n3.optimize(solver_name=solver_name, solver_options=solver_options)

#### N4
print("Running islanded optimisation (AC lines and DC links zero availability in summer: Q2+Q3)...")
n4 = n.copy()

snapshots = n4.snapshots
quarters = np.array_split(snapshots, 4)
q1, q2, q3, q4 = quarters
outage_summer = q2.union(q3)

if not n4.lines.empty:
    ac_lines = n4.lines.index[n4.lines.carrier == "AC"]
    lines_avail = pd.DataFrame(index=snapshots, columns=ac_lines, dtype=float)
    lines_avail[:] = n4.lines.loc[ac_lines, "s_max_pu"]
    lines_avail.loc[outage_summer, :] = 0
    n4.lines_t.s_max_pu = lines_avail

if not n4.links.empty:
    dc_links = n4.links.index[n4.links.carrier == "DC"]
    links_avail = pd.DataFrame(index=snapshots, columns=dc_links, dtype=float)
    links_avail[:] = n4.links.loc[dc_links, "p_max_pu"]
    links_avail.loc[outage_summer, :] = 0
    n4.links_t.p_max_pu = links_avail
    n4.links_t.p_min_pu = -links_avail

n4.optimize(solver_name=solver_name, solver_options=solver_options)

#### N5
print("Running islanded optimisation (AC lines and DC links zero availability in winter: Q1+Q4)...")
n5 = n.copy()

snapshots = n5.snapshots
quarters = np.array_split(snapshots, 4)
q1, q2, q3, q4 = quarters
outage_winter = q1.union(q4)

if not n5.lines.empty:
    ac_lines = n5.lines.index[n5.lines.carrier == "AC"]
    lines_avail = pd.DataFrame(index=snapshots, columns=ac_lines, dtype=float)
    lines_avail[:] = n5.lines.loc[ac_lines, "s_max_pu"]
    lines_avail.loc[outage_winter, :] = 0
    n5.lines_t.s_max_pu = lines_avail

if not n5.links.empty:
    dc_links = n5.links.index[n5.links.carrier == "DC"]
    links_avail = pd.DataFrame(index=snapshots, columns=dc_links, dtype=float)
    links_avail[:] = n5.links.loc[dc_links, "p_max_pu"]
    links_avail.loc[outage_winter, :] = 0
    n5.links_t.p_max_pu = links_avail
    n5.links_t.p_min_pu = -links_avail

n5.optimize(solver_name=solver_name, solver_options=solver_options)

#### N6
print("Running stochastic optimisation (summer vs winter outage scenarios with 50/50 probability)...")
n6 = n.copy()

snapshots = n6.snapshots
quarters = np.array_split(snapshots, 4)
q1, q2, q3, q4 = quarters
t_summer = q2.union(q3)
t_winter = q1.union(q4)

# Get AC lines and DC links before setting scenarios
ac_lines = n6.lines.index[n6.lines.carrier == "AC"] if not n6.lines.empty else []
dc_links = n6.links.index[n6.links.carrier == "DC"] if not n6.links.empty else []

# Set up stochastic scenarios - this broadcasts all data across scenarios
n6.set_scenarios({"summer_outage": 0.5, "winter_outage": 0.5})

if len(ac_lines) > 0:
    # Build multi-index columns for line availabilities
    scenarios = ["summer_outage", "winter_outage"]
    columns = pd.MultiIndex.from_product(
        [scenarios, ac_lines],
        names=["scenario", "name"]
    )

    # Create empty DataFrame to hold line availabilities
    dfn6 = pd.DataFrame(index=n6.snapshots, columns=columns, dtype=float)

    # Fill in static line availabilities for each scenario
    for scen in scenarios:
        dfn6.loc[:, scen] = n6.lines.loc[(scen, ac_lines), "s_max_pu"].to_numpy()

    # Set zero availabilities during outages
    dfn6.loc[t_summer, ("summer_outage", ac_lines)] = 0.0
    dfn6.loc[t_winter, ("winter_outage", ac_lines)] = 0.0

    # Assign to lines_t.s_max_pu
    n6.lines_t.s_max_pu = dfn6

if len(dc_links) > 0:
    columns = pd.MultiIndex.from_product(
        [scenarios, dc_links],
        names=["scenario", "name"]
    )

    # Create empty DataFrame to hold link availabilities
    dfn6_p_max_pu = pd.DataFrame(index=n6.snapshots, columns=columns, dtype=float)
    dfn6_p_min_pu = pd.DataFrame(index=n6.snapshots, columns=columns, dtype=float)

    # Fill in static link availabilities for each scenario
    for scen in scenarios:
        dfn6_p_max_pu.loc[:, scen] = n6.links.loc[(scen, dc_links), "p_max_pu"].to_numpy()
        dfn6_p_min_pu.loc[:, scen] = n6.links.loc[(scen, dc_links), "p_min_pu"].to_numpy()
    
    # Set zero availabilities during outages
    dfn6_p_max_pu.loc[t_summer, ("summer_outage", dc_links)] = 0.0
    dfn6_p_min_pu.loc[t_summer, ("summer_outage", dc_links)] = 0.0
    dfn6_p_max_pu.loc[t_winter, ("winter_outage", dc_links)] = 0.0
    dfn6_p_min_pu.loc[t_winter, ("winter_outage", dc_links)] = 0.0

    # Assign to links_t.p_max_pu and p_min_pu
    n6.links_t.p_max_pu = dfn6_p_max_pu
    n6.links_t.p_min_pu = dfn6_p_min_pu

# Note: Dual assignment fails with multi-indexed stochastic scenarios
# Catch the error since the solution is already assigned before dual assignment
n6.optimize(solver_name=solver_name, solver_options=solver_options)

n6.statistics.opex()

## Metrics
total_costs = pd.DataFrame({
    "scenario": [
        "Grid exists",
        "Grid outage (full year, s_max_pu=0)",
        "Grid outage (full year, lines removed)",
        "Grid outage (Summer)",
        "Grid outage (Winter)",
        "Grid outage (Stochastic: Summer/Winter)",
    ],
    "total_cost": [
        np.round(calculate_total_costs(n1) / 1e9, 3),
        np.round(calculate_total_costs(n2, exclude_grid=False) / 1e9, 3),
        np.round(calculate_total_costs(n3, exclude_grid=True) / 1e9, 3),
        np.round(calculate_total_costs(n4, exclude_grid=False) / 1e9, 3),
        np.round(calculate_total_costs(n5, exclude_grid=False) / 1e9, 3),
        np.round(calculate_total_costs(n6, exclude_grid=False) / 1e9, 3),
    ],
})
print("\nTotal annual costs by scenario:")
print(total_costs.to_string(index=False))

