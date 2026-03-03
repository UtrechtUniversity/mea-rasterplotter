# mea-rasterplotter

## Prerequisites

- MATLAB for using the [AxionFileLoader](https://github.com/axionbio/AxionFileLoader) Matlab loader.

  Or in testing/development: [GNU Octave](https://octave.org/) with modified AxionFileLoader.
- [UV](https://docs.astral.sh/uv/) Python package manager.

## Installation

```sh
# Clone mea-rasterplotter
git clone --recurse-submodules https://github.com/UtrechtUniversity/mea-rasterplotter.git
# Or, if already cloned, add the AxionFileLoader submodule:
git submodule update --init

# Create/update environment
uv sync
```
