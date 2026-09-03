# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: CC0-1.0

# Bakes the pypsa-eur pixi env into an image so `pixi install` never has to
# touch BeeGFS again — same rationale and pattern as the parent repo's own
# Dockerfile (../../../Dockerfile), but this submodule has its own separate
# pixi env, so it gets its own image (pypsa-eur-env.sif, not pypsaro-env.sif).

FROM ghcr.io/prefix-dev/pixi:0.78.0 AS build
WORKDIR /app
COPY pixi.toml pixi.lock ./

# --locked aborts on a stale lock instead of silently re-solving. No stub
# package needed here (unlike the parent repo's `aro`) -- this submodule has
# no pypi-dependencies/editable-installed local package; its own scripts/
# are read directly by Snakemake, same as workflow/ in the parent repo.
RUN pixi install --locked -e default \
 && rm -rf /root/.cache/rattler /root/.cache/pip

FROM debian:trixie-slim AS runtime
WORKDIR /app
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default

# Static ENV, not a sourced pixi shell-hook script -- deliberately diverging
# from upstream's own docker/dev-env/Dockerfile, which uses shell-hook +
# ENTRYPOINT. That pattern silently breaks Snakemake's *automatic* per-job
# apptainer wrapping (it runs `sh -c '<rule command>'` directly, with zero
# awareness of any custom activation script -- confirmed by reading
# snakemake/deployment/singularity.py in the parent repo's own env). Static
# Docker ENV is auto-applied by Singularity before any command runs
# regardless of what shell wraps it, so it covers both manual invocation and
# Snakemake's own per-job wrapping uniformly. Values confirmed empirically
# (built the `build` stage, ran `pixi shell-hook` inside it, diffed env
# before/after sourcing it) -- five conda activate.d scripts fire here
# (gdal, libarrow, libglib, libxml2-split, proj4; libarrow's had no visible
# env effect), a larger set than the parent repo's three since this env
# pulls GDAL directly (fiona/rasterio/geopandas), not just via PROJ.
ENV CONDA_PREFIX=/app/.pixi/envs/default \
    PATH=/app/.pixi/envs/default/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LC_ALL=C \
    CPL_ZIP_ENCODING=UTF-8 \
    GDAL_DATA=/app/.pixi/envs/default/share/gdal \
    GDAL_DRIVER_PATH=/app/.pixi/envs/default/lib/gdalplugins \
    GSETTINGS_SCHEMA_DIR=/app/.pixi/envs/default/share/glib-2.0/schemas \
    PROJ_DATA=/app/.pixi/envs/default/share/proj \
    PROJ_NETWORK=ON \
    XML_CATALOG_FILES="file:///app/.pixi/envs/default/etc/xml/catalog file:///etc/xml/catalog"

# libsubid5 is unrelated to pypsa-eur's own dependencies -- it's what the
# *cluster's* singularity/apptainer binary needs to run when bind-mounted
# into this container (for the driver-runs-inside-the-container case), same
# fix as the parent repo's Dockerfile and same reason.
RUN apt-get update && apt-get install -y --no-install-recommends libsubid5 \
 && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
