# mea-rasterplotter

## Prerequisites

- MATLAB for the default Axion `.spk` processing path, using the vendored
  [AxionFileLoader](https://github.com/axionbio/AxionFileLoader) MATLAB loader.

- Optional developer-only support for [GNU Octave](https://octave.org/).
  This is experimental and requires switching
  `mea-rasterplotter/vendor/AxionFileLoader` to the `feature/octave` branch
  before using the Octave notebook path.
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
