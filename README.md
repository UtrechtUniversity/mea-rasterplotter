# mea-rasterplotter

Scripts for extracting spike timings from Axion MultiElectrode Arrays (MEA) `.spk` files, exporting them to CSV, and visualizing the data with rasterplots.

## Status

This project is incomplete and is in development. The notebook UIs are still a little rough.

## Prerequisites

- Python
- MATLAB for the default Axion `.spk` processing path, using the vendored
  [AxionFileLoader](https://github.com/axionbio/AxionFileLoader) MATLAB loader.
- Optional support for [GNU Octave](https://octave.org/) instead of MATLAB.
  This is meant for development/testing, and requires switching
  `mea-rasterplotter/vendor/AxionFileLoader` to the `feature/octave` branch
  before using the Octave notebook path.
- Recommended: [Git](https://git-scm.com/) and [UV](https://docs.astral.sh/uv/).

## Setup

### Download

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
# Install and activate an environment
py -m venv .venv
.venv\Scripts\activate
# Install the dependencies
py -m pip install -r requirements.txt
```

## Use

The functionality can be used via two notebooks:
- `process_spk.py` for extracting spike timings from .spk and saving to a CSV file.
- `rasterplot.py` for loading a CSV file and visualizing the data in a rasterplot.

Open Marimo from an activated venv:

```sh
marimo edit
```

Marimo will open in a browser window.

## Development

```sh
# Install optional dependencies
uv sync --extra marimo-ai
```
