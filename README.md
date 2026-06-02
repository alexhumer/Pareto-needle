# Pareto-needle
This repository is supplementary material to the paper [Humer2026] on the Pareto-optimization of hypodermic needles. It provides the code to run the optimization, raw data of the published results (res_all.pkl.1) and a Python script to visualize 3d-projections of the Pareto-front. 

## Target group:
The tool is provided to engineers and scientists working on optimization of biomedical devices. For understanding the code, basic math and Python programming is required.

## Getting started:
To run the optimization loop use the following command
```bash
python3 compute_cannula.py 
```
which will output the Pareto-optimal designs as a pickle file (default: res_all.pkl). 

To visualize the results (including the results provided with this repository), run the file
```bash
python3 postproc_all.py 
```

## Sources:
[Humer2026] Humer, A., Ehrenhofer, A., Wallmersperger, T., Krommer, M.: Pareto-optimization of hypodermic needles with respect to flow, buckling and piercing. Acta Mech (2026). https://doi.org/10.1007/s00707-025-04543-y

## Final notes:
The software is referenced in Zenodo.
[![DOI](https://zenodo.org/badge/943326362.svg)](https://doi.org/10.5281/zenodo.14975196)
