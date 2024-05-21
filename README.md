This code reproduces all results in the paper "Magnetization transfer explains most of the T1 variability in the MRI literature". The code is written in the open-source language Julia and is structured as follows:
- Fit_qMT_to_literatureT1.jl is the main script and implements the pulse sequence simulations and the mono-exponential fitting routines. It also calls the fit of the qMT models to the variable literature T1 estimates.
- helper_functions.jl contains implementations of RF pulses and some helper functions, which are less relevant for understanding the simulations
- Project.toml and Manifest.toml contain information about the packages used by the simulation, facilitating their automated installation.

Julia can be downloaded from the website https://julialang.org or, on Unix systems, installed by simply calling
    `curl -fsSL https://install.julialang.org | sh`
from the command line.

In order to run the simulations, place all four files in the same folder and call
    `julia -iq --threads=auto some_path/Fit_qMT_to_literatureT1.jl`
replacing `some_path` with the path to the files. For a more interactive interface, the code can be called from Visual Studio Code with the Julia extension.

Upon publication of the paper, this script will be provided on GitHub and will be rendered in HTML and Jupyter Notebook formats for readability and interactivity. The latter will be integrated with binder, allowing the users to modify the code in a browser and run it in the cloud without any local installations. An example of this interface can be found at https://jakobasslaender.github.io/MRIgeneralizedBloch.jl/stable/build_literate/Simulation_Pulse/.