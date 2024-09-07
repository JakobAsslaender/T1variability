# Magnetization transfer explains most of the T₁ variability in the MRI literature

This code reproduces all results in the paper [Magnetization transfer explains most of the T₁ variability in the MRI literature](https://TODO.org). The code is written in the open-source language Julia and is structured as follows:

- [T₁ Mapping methods](@ref) contains functions that simulate the MR signal of each T₁-mapping method and fit a mono-exponential T₁ to the simulated data.
- [Global Fit](@ref) performs the the fits of the quantitative magnetization transfer parameters to the variable T₁ estimates.
- [Helper functions](@ref) contains implementations of RF pulses and some helper functions, which are less relevant for understanding the simulations. This script also loads all required packages.


```@contents
Pages=[
        "build_literate/Fit_qMT_to_literatureT1.md",
        "build_literate/helper_functions.md",
]
Depth = 2
```

