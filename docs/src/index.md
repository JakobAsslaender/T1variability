## Magnetization transfer explains most of the T₁ variability in the MRI literature

The code in the first section reproduces all results in the paper [Magnetization transfer explains most of the T₁ variability in the MRI literature](https://arxiv.org/pdf/2409.05318). The code is written in the open-source language Julia and is structured as follows:

- [T₁-mapping methods](@ref): This script contains functions to simulate the MR signal of each T₁-mapping method and fit a mono-exponential T₁ to the simulated data.
- [Global fit](@ref): This script performs the fits of the quantitative magnetization transfer parameters to the variable T₁ estimates.
- [Helper functions](@ref): This script contains implementations of RF pulses and some helper functions, which are less relevant for understanding the simulations. This script also loads all required packages.

## Sensitivity of literature T₁ mapping methods to the underlying magnetization transfer parameters

The code in the second section reproduces all results in the follow-up paper [Sensitivity of literature T₁ mapping methods to the underlying magnetization transfer parameters](https://arxiv.org/pdf/2509.13644). The code is also written in the open-source language Julia and is structured as follows:

- [Sensitivity analysis](@ref): This script calculates the derivatives, analyzes, and plots them.
- [Helper functions for the sensitivity analysis](@ref): This script contains functions for the calculation of the coefficient of determination `R²`. This script also loads all required packages.