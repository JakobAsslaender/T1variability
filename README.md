| **T₁ Variability Paper**      | **Corresponding Code**            |
|:----------------------------- |:--------------------------------- |
| [![][arXiv-img1]][arXiv-url1] | [![][docsv0.2-img]][docsv0.2-url] |
| unpublished edits             | [![][docsdev-img]][docsdev-url]   |

This code reproduces all results in the paper [Magnetization transfer explains most of the T1 variability in the MRI literature][arXiv-url1]. Please refer to the [Documentation][docs-url] for a detailed description of the code.

The code is written in the open-source language Julia and is structured as follows:
- `T1_mapping_methods.jl` implements the pulse sequence simulations and the mono-exponential fitting routines of each T₁-mapping method.
- `Fit_qMT_to_literatureT1.jl` is the main script that performs the fit of the qMT models to the variable literature T₁ estimates.
- `helper_functions.jl` contains implementations of RF pulses, their propagators, and some helper functions, which are less relevant for understanding the simulations.
- `Project.toml` and `Manifest.toml` contain information about the packages used by the simulation, facilitating their automated installation.

Julia can be downloaded from https://julialang.org or, on Unix systems, by simply calling

    `curl -fsSL https://install.julialang.org | sh`

from the command line.

In order to run the simulations, place all five files in the same folder, `cd` into this folder, and call

    `julia -iq --threads=auto Fit_qMT_to_literatureT1.jl`

For a more interactive interface, the code can be called from Visual Studio Code with the Julia extension.

[docsdev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docsdev-url]: https://jakobasslaender.github.io/T1variability/dev/

[docsv0.2-img]: https://img.shields.io/badge/docs-v0.2-blue.svg
[docsv0.2-url]: https://jakobasslaender.github.io/T1variability/v0.2/

[arXiv-img1]: https://img.shields.io/badge/arXiv-2409.05318-blue.svg
[arXiv-url1]: https://arxiv.org/pdf/2409.05318v1
