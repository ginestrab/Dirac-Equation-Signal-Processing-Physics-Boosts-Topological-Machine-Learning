# Dirac Equation Signal Processing


This repository contains the notebooks, functions and datasets necessary to reproduce results published in:

[1] Wang, R., Tian, Y., Liò, P., Bianconi, G. (2025) 'Dirac-Equation Signal Processing: Physics Boosts Topological Machine Learning'.

This code is distributed by the authors in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

If you use any of these codes please cite the following papers:
[1] Wang, R., Tian, Y., Liò, P., Bianconi, G. (2025) 'Dirac-Equation Signal Processing: Physics Boosts Topological Machine Learning'.
[2] Bianconi, G., (2021). The topological Dirac equation of networks and simplicial complexes. Journal of Physics: Complexity, 2(3), p.035022.

# Content:

## Drift file:
If using this code, please cite

[3] Schaub, M.T., Benson, A.R., Horn, P., Lippner, G., and Jadbabaie, A. (2020) ‘Random walks on simplicial complexes and the normalized Hodge 1-Laplacian’, SIAM Review, 62(2), pp. 353–391. doi:10.1137/18m1201019.

[4] Roddenberry, T.M., Glaze, N. and Segarra, S., (2021), July. Principled simplicial neural networks for trajectory prediction. In International Conference on Machine Learning (pp. 9020-9029). PMLR.

[5] https://github.com/nglaze00/SCoNe_GCN/tree/master

[5] Lumpkin, Rick; Centurioni, Luca (2019). Global Drifter Program quality-controlled 6-hour interpolated data from ocean surface drifting buoys. NOAA National Centers for Environmental Information. Dataset. https://doi.org/10.25921/7ntx-z961.

Files:

dataBuoys-coords.jld2: dataset in jld2 format dataBuoys.jld2: dataset in jld2 format that includes the coordinates of the hexagons buoy_data.py: script to read the above files and return python arrays necessary to the analysis Buoys Mass-general noise-to-share.ipynb: notebook containing the analysis conducted in paper [1]

## Fungi file:

If using this code, please cite

[1] Wang, R., Tian, Y., Liò, P., Bianconi, G. (2025) 'Dirac-Equation Signal Processing: Physics Boosts Topological Machine Learning'.

Files:
Fungi.xlsx: dataset on fungi from A. Benson, Austin Benson Data Repository, https://www.cs.cornell.edu/~arb/data/, accessed: 11-27-2024.

## General_Function.py:
Contains all functions, including the signal processing algorithms used in the paper, as well as the generation of synthetic signals and noise.

## NGF file:

The NGF code is at the repository https://github.com/ginestrab/Network-Geometry-with-Flavor

If using this code, please cite

[6] G. Bianconi and C. Rahmede "Network geometry with flavor:from complexity to quantum geometry" Physical Review E 93, 032315 (2016).

Files:

NGF_edgelist.edges: dataset containing list of all edges in the NGF network. 

