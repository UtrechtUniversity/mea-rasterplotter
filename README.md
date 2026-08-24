# mea-rasterplotter

Scripts for extracting spike timings from Axion MultiElectrode Arrays (MEA) `.spk` files, exporting them to CSV, and visualizing the data with rasterplots.

## Status

This project is incomplete and is in development. The notebook UIs are still a little rough.

## Prerequisites

- Python
- MATLAB for the Axion `.spk` spike timing extraction, using the vendored
  [AxionFileLoader](https://github.com/axionbio/AxionFileLoader) MATLAB loader.
- Recommended: [Git](https://git-scm.com/) and [UV](https://docs.astral.sh/uv/).

## Setup

### Download

Open a console or Windows Command Prompt at the desired location and clone the repository:

```sh
# Clone mea-rasterplotter
git clone --recurse-submodules https://github.com/UtrechtUniversity/mea-rasterplotter.git
# Or, if already cloned, add the AxionFileLoader submodule:
git submodule update --init
```

If you do not have Git available, download ZIP files of both `mea-rasterplotter` and `AxionFileLoader` from GitHub (located under the `<> Code` button), and extract them so you have this directory layout:

```sh
mea-rasterplotter/
├── README.md
...
└── vendor
    └── AxionFileLoader
```

### Install dependencies

If you have [uv](https://docs.astral.sh/uv/) installed (recommended):

```sh
# Create/update environment with dependencies
uv sync
```

Alternatively, use pip (included with Python):

```sh
# Navigate inside the repository dir:
cd mea-rasterplotter
# Example for Windows:
# Install and activate a Python environment:
py -m venv .venv
.venv\Scripts\activate
# Install the dependencies in the environment:
py -m pip install -r requirements.txt
```

### Updates

To get updates, fetch the latest changes with Git, and install dependencies if they were updated:

```sh
cd /path/to/mea-rasterplotter
git pull # The default branch is develop
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Use

The functionality can be used via two notebooks:
- `process_spk.py` for extracting spike timings from .spk and saving to a CSV file.
- `rasterplot.py` for loading a CSV file and visualizing the data in a rasterplot.

Open Marimo from an activated venv:

```sh
# Use `marimo run` to run a notebook as an app, hiding the code cells:
marimo run process_spk.py
marimo run rasterplot.py
# Use `marimo edit` to open marimo in edit mode
marimo edit
```

Marimo will open in a browser window.

## Development

```sh
# Install optional dependencies
uv sync --extra marimo-ai
```

```sh
# Upgrade uv.lock dependencies
uv lock --upgrade
# Export updates to requirements.txt
uv export --format requirements.txt --output-file requirements.txt
```
