# Magnetization transfer explains most of the T₁ variability in the MRI literature

This code reproduces all results in the paper [Magnetization transfer explains most of the T₁ variability in the MRI literature](https://arxiv.org/pdf/2409.05318). The code is written in the open-source language Julia and is structured as follows:

- [T₁-mapping methods](@ref): This script contains functions to simulate the MR signal of each T₁-mapping method and fit a mono-exponential T₁ to the simulated data.
- [Global fit](@ref): This script performs the fits of the quantitative magnetization transfer parameters to the variable T₁ estimates.
- [Helper functions](@ref): This script contains implementations of RF pulses and some helper functions, which are less relevant for understanding the simulations. This script also loads all required packages.