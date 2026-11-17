# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Transmission projects configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#transmission_projects_cf
"""

from typing import Literal

from pydantic import BaseModel, Field

from scripts.lib.validation.config._base import ConfigModel


class _IncludeConfig(ConfigModel):
    """Configuration for `transmission_projects.include` settings."""

    tyndp2020: bool = Field(
        True,
        description="Whether to integrate the TYNDP 2020 dataset.",
    )
    nep: bool = Field(
        True,
        description="Whether to integrate the German network development plan dataset.",
    )
    aro: bool = Field(
        False,
        description="Whether to integrate the fork's links missing from OSM (`data/transmission_projects/aro`): the Attica-Crete HVDC (Ariadne, 1000 MW, in operation since 2025, status `completed`).",
    )
    manual: bool = Field(
        True,
        description="Whether to integrate the manually added transmission projects. They are taken from the previously existing links_tyndp.csv file.",
    )


class TransmissionProjectsConfig(BaseModel):
    """Configuration for `transmission_projects` settings."""

    enable: bool = Field(
        True,
        description="Whether to integrate this transmission projects or not.",
    )
    include: _IncludeConfig = Field(
        default_factory=_IncludeConfig,
        description="Name of the transmission projects. They should be unique and have to be provided in the `data/transmission_projects` folder.",
    )
    skip: list[str] = Field(
        default_factory=lambda: ["upgraded_lines", "upgraded_links"],
        description="Type of lines to skip from all transmission projects. Possible values are: `upgraded_lines`, `upgraded_links`, `new_lines`, `new_links`.",
    )
    status: list[str] | dict[str, list[str]] = Field(
        default_factory=lambda: ["under_construction", "in_permitting", "confirmed"],
        description="Status to include into the model as list or as dict with name of project and status to include. Possible values for status are `under_construction`, `in_permitting`, `confirmed`, `planned_not_yet_permitted`, `under_consideration`.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="Projects left out, e.g. ones already in the base network that the duplicate check misses because the project's coordinates are off: ElecLink (TYNDP2020_36, in operation since 2022) sits at IFA's endpoints, 0.2 degrees from its own OSM link, beyond the check's 0.1.",
    )
    replaces: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Projects that replace existing links on the same route, mapped to those links: the existing links are removed when the project is added. For replacements the duplicate check misses because the capacity changes, e.g. SACOI3 (TYNDP2020_29/30, 400 MW) replacing SACOI2 (300 MW).",
    )
    new_link_capacity: Literal["zero", "keep"] = Field(
        "zero",
        description="Whether to set the new link capacity to the provided capacity or set it to zero.",
    )
