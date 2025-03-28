# pln_iefc_jatis

This is the data and code associated with the paper: "Implicit Electric Field Conjugation with a Photonic Lantern Nuller"

The main directory contains the Jupyter notebook that reproduces the figures from the paper. The associated lab data is in the 'data' folder.

# ABOUT THE SIMULATIONS: PLEASE READ

The simulations referred to in the discussion section can be found under 'simulations.' The notebook titles should hopefully be self-explanatory (iEFC with the real lantern modes, iEFC with a perfect lantern with phase only pupil aberrations, iEFC with a perfect lantern with phase-and-amplitude pupil aberrations, EFC with the real lantern modes, and a propagation of real and ideal lantern modes back onto the pupil).

The underlying control code used in the pln_iefc_sims.py is the same code as was used on the testbed: a combination of inherited bench code for controlling the DM and functions from the [lina](https://github.com/uasal/lina/blob/main/lina) package to conduct iEFC.

I wanted to use the exact same code for consistency, which meant adding in a simulation of the lantern in place of reading the camera. Because I hadn't been injecting any additional aberrations in the testbed, the easiest way to add pupil errors in the simulation ended up being just adding a global WFE rather than passing it through all the layers of function calls, which is why three copies of the code (each with different pupil-plane WFE) are provided. (There was probably a better way to do this, but this was the simplest.)

Also, the WFE is randomly generated with "make_power_law_error" function from HCIPy, which doesn't take a seed, so you'll get a slightly different result each time you restart the notebook.

The functions used in the simulations provided have been tested and are working. It's probably best to assume that any functions not used by the notebooks have not been rigorously tested. Feel free to reach out to me if you're interested in them or have other questions!