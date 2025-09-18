| **Publication**        | **Links to Publications**                               | **Corresponding Code**                                              |
|:-----------------------|:------------------------------------------------------- |:------------------------------------------------------------------- |
| T₁ Variability         | [![][MRM-img1]][MRM-url1] [![][arXiv-img1]][arXiv-url1] | [![][docsv1.0-img]][docsv1.0-url] [![][docsv2.0-img]][docsv2.0-url] |
| T₁ Sensitivity         | [![][arXiv-img2]][arXiv-url2]                           | [![][docsv2.0-img]][docsv2.0-url]                                   |
| unpublished edits      |                                                         | [![][docsdev-img]][docsdev-url]                                     |

# T₁ Variability Paper
This code reproduces all results in the paper [Magnetization transfer explains most of the T₁ variability in the MRI literature][arXiv-url1]. Please refer to the [Documentation][docsv1.0-url] for a detailed description of the code.

The code is written in the open-source language Julia and is structured as follows:
- `T1_mapping_methods.jl` implements the pulse sequence simulations and the mono-exponential fitting routines of each T₁-mapping method.
- `Fit_qMT_to_literatureT1.jl` is the main script that performs the fit of the qMT models to the variable literature T₁ estimates.
- `helper_functions.jl` contains implementations of RF pulses, their propagators, and some helper functions, which are less relevant for understanding the simulations.
- `Project.toml` and `Manifest.toml` contain information about the packages used by the simulation, facilitating their automated installation.

# T₁ Sensitivity Paper
This code also reproduces all results in the paper [Sensitivity of literature T₁ mapping methods to the underlying magnetization transfer parameters][arXiv-url2]. Please refer to the [Documentation][docsv2.0-url] for a detailed description of the code.

In addition to the scripts described above, this analysis uses the following code:
- `Derivatives.jl` is the main script that calculates the derivatives and performs the mixed effects model fit.
- `Derivatives_HelperFunctions.jl` contains implementations of the Shapley regression and other helper functions.

# Running the code
Julia can be downloaded from https://julialang.org or, on Unix systems, by simply calling

    `curl -fsSL https://install.julialang.org | sh`

from the command line.

To run the simulations, place all files in the same folder, `cd` into this folder, and call

    `julia -iq --threads=auto --project=. Fit_qMT_to_literatureT1.jl`

for reproducing the T₁ variability paper and

    `julia -iq --threads=auto --project=. Derivaties.jl`

for reproducing the T₁ sensitivity paper.

For a more interactive interface, the code can be called from Visual Studio Code with the Julia extension.

[docsdev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docsdev-url]: https://jakobasslaender.github.io/T1variability/dev/

[docsv1.0-img]: https://img.shields.io/badge/docs-v1.0-blue.svg
[docsv1.0-url]: https://jakobasslaender.github.io/T1variability/v1.0/

[arXiv-img1]: https://img.shields.io/badge/arXiv-2409.05318-blue.svg
[arXiv-url1]: https://arxiv.org/pdf/2409.05318v1

[MRM-img1]: https://img.shields.io/badge/doi-10.1002/mrm.30451-blue.svg
[MRM-url1]: https://doi.org/10.1002/mrm.30451

[docsv2.0-img]: https://img.shields.io/badge/docs-v2.0-blue.svg
[docsv2.0-url]: https://jakobasslaender.github.io/T1variability/v2.0/

[arXiv-img2]: https://img.shields.io/badge/arXiv-2509.13644-blue.svg
[arXiv-url2]: https://arxiv.org/pdf/2509.13644
