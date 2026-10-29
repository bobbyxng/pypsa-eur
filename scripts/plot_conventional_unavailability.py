# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot the availability of conventional generators per country and calendar month built by
``build_conventional_unavailability``, next to the static ``p_max_pu`` it replaces.

One heatmap per carrier: a row per country with units of the carrier in the power plant
list, a column per month and a last column with the static value for the modelled year.
All panels share one colour scale from the lowest to the highest value shown, so countries
(and carriers) compare directly; darker means more available. Row labels give the number
of reference years behind a country's profile, or ``median`` for the fallback.

For ``method: historical`` the country rows are averaged per calendar month.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_conventional_unavailability import PPM_FUELTYPES

logger = logging.getLogger(__name__)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
MONTHS += ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# One hue, light to dark; the lightest step stays clear of the white cell gaps.
CMAP = LinearSegmentedColormap.from_list(
    "availability",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
TEXT_DARK, TEXT_LIGHT = "#333333", "#ffffff"


def monthly_table(unavailability: pd.DataFrame, carrier: str) -> pd.DataFrame:
    """Country rows of one carrier as a country x calendar month table."""
    df = unavailability[
        (unavailability.carrier == carrier) & unavailability.unit.isna()
    ]
    period = pd.to_numeric(df.period, errors="coerce")
    if period.isna().any():  # historical: snapshots instead of months
        period = pd.to_datetime(df.period).dt.month
    return df.assign(month=period.astype(int)).pivot_table(
        index="country", columns="month", values="p_max_pu"
    )


def static_values(fn: str, year: int) -> pd.Series:
    """The static table's column for the modelled year, as add_electricity picks it."""
    df = pd.read_csv(fn, index_col=0)
    try:
        df.columns = df.columns.astype(int)
        return df[year]
    except (ValueError, TypeError, KeyError):
        return df.iloc[:, -1]


def fleet_countries(powerplants: pd.DataFrame, carrier: str) -> list[str]:
    fueltypes = PPM_FUELTYPES.get(carrier, [])
    return sorted(
        powerplants.loc[powerplants.Fueltype.isin(fueltypes), "Country"].unique()
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("plot_conventional_unavailability")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    unavailability = pd.read_csv(snakemake.input.unavailability)
    powerplants = pd.read_csv(snakemake.input.powerplants, index_col=0)
    year = pd.Timestamp(snakemake.params.snapshots["start"]).year

    panels = {}
    for carrier in snakemake.params.carriers:
        table = monthly_table(unavailability, carrier)
        countries = [
            c for c in fleet_countries(powerplants, carrier) if c in table.index
        ]
        if not countries:
            logger.info(f"No {carrier} units with an availability profile; skipping.")
            continue
        table = table.loc[countries].reindex(columns=range(1, 13))
        if f"static_{carrier}" in snakemake.input.keys():
            static = static_values(snakemake.input[f"static_{carrier}"], year)
            table["static"] = static.reindex(countries)
        rows = unavailability[
            (unavailability.carrier == carrier) & unavailability.unit.isna()
        ]
        info = rows.groupby("country")[["source", "years"]].first().loc[countries]
        labels = [
            f"{c}  (median)"
            if info.at[c, "source"] == "median"
            else f"{c}  ({info.at[c, 'years']:.0f} yr)"
            for c in countries
        ]
        order = np.argsort(table.iloc[:, :12].mean(axis=1).to_numpy())
        panels[carrier] = (table.iloc[order], [labels[i] for i in order])

    # One scale for every cell shown, so colours compare across countries and carriers.
    values = np.concatenate([t.to_numpy().ravel() for t, _ in panels.values()] or [[]])
    vmin, vmax = (np.nanmin(values), np.nanmax(values)) if len(values) else (0, 1)

    heights = [max(len(t), 1) for t, _ in panels.values()] or [1]
    fig, axes = plt.subplots(
        len(heights),
        1,
        figsize=(8.2, 1.2 + 0.4 * sum(heights)),
        height_ratios=heights,
        squeeze=False,
    )
    if not panels:
        axes[0, 0].text(0.5, 0.5, "No availability profiles", ha="center")
        axes[0, 0].set_axis_off()

    for ax, (carrier, (table, labels)) in zip(axes[:, 0], panels.items()):
        grid = table.to_numpy(dtype=float)
        has_static = "static" in table
        if has_static:  # an empty column sets the static value apart
            grid = np.insert(grid, 12, np.nan, axis=1)
        ax.imshow(grid, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        norm = (grid - vmin) / (vmax - vmin) if vmax > vmin else grid * 0
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=TEXT_LIGHT if norm[i, j] > 0.5 else TEXT_DARK,
                )
        ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        xlabels = MONTHS + (["", f"Static\n({year})"] if has_static else [])
        ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"{carrier.capitalize()} availability per country and month (p_max_pu)",
            loc="left",
            fontsize=10,
            color=TEXT_DARK,
        )

    params = snakemake.params.unavailability
    periods = params["reference_periods"]
    note = (
        f"ENTSO-E outages, {params['method']}; reference periods "
        f"{', '.join(map(str, periods['include']))}"
        + (
            f" without {', '.join(map(str, periods['exclude']))}"
            if periods["exclude"]
            else ""
        )
        + "."
    )
    fig.text(0.01, 0.01, note, fontsize=6.5, color="#666666", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(snakemake.output.plot, bbox_inches="tight")
