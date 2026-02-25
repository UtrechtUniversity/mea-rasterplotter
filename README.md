# mea-rasterplotter

## Prerequisites

- [GNU Octave](https://octave.org/) for using the Matlab loader [AxionFileLoader](https://github.com/axionbio/AxionFileLoader).
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
