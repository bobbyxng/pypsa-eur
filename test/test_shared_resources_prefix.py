# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Fork change: shared resources stay inside the run prefix (`shared_dir`)."""

from scripts._helpers import get_run_path

RDIR = "my-prefix/{run}/"


def test_base_shares_inside_prefix():
    """Shared files go under the prefix; per-run files keep the run directory."""
    kwargs = dict(
        dir="resources/",
        rdir=RDIR,
        shared_resources="base",
        exclude_from_shared=["costs_"],
        shared_dir="my-prefix/",
    )
    assert get_run_path("networks/clustered.nc", **kwargs) == (
        "resources/my-prefix/networks/clustered.nc"
    )
    assert get_run_path("networks/composed_{horizon}.nc", **kwargs) == (
        "resources/my-prefix/{run}/networks/composed_{horizon}.nc"
    )
    assert get_run_path("costs_2050_processed.csv", **kwargs) == (
        "resources/my-prefix/{run}/costs_2050_processed.csv"
    )


def test_default_shared_dir_is_upstream_behaviour():
    """Without a prefix, shared files land at the top level as upstream."""
    assert (
        get_run_path("networks/clustered.nc", "resources/", "{run}/", "base", [])
        == "resources/networks/clustered.nc"
    )
    assert (
        get_run_path("x.csv", "resources/", "{run}/", True, [], shared_dir="p/")
        == "resources/p/x.csv"
    )
