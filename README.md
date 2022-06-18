# Long-Term-GWL-Simulations

doi of this repo:  
[![DOI](https://zenodo.org/badge/349114094.svg)](https://zenodo.org/badge/latestdoi/349114094) 

doi of groundwater dataset:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4683879.svg)](https://doi.org/10.5281/zenodo.4683879)


This repository enables you to reproduce the results and apply the long-term groundwater level simulation methodology of:     
##### **Wunsch, A., Liesch, T. & Broda, S. Deep learning shows declining groundwater levels in Germany until 2100 due to climate change. Nat Commun 13, 1221 (2022). https://doi.org/10.1038/s41467-022-28770-2**  

Contact: [**andreas.wunsch@kit.edu**](andreas.wunsch@kit.edu)  

ORCIDs of authors:   
* A. Wunsch:  [**0000-0002-0585-9549**](https://orcid.org/0000-0002-0585-9549)   
* T. Liesch:  [**0000-0001-8648-5333**](https://orcid.org/0000-0001-8648-5333)   
* S. Broda:  [**0000-0001-6858-6368**](https://orcid.org/0000-0001-6858-6368)   

For a detailed description please refer to the publication.
Please adapt all absolute loading/saving and software paths within the scripts to make them running. We further use the following packages:

*  [**BayesianOptimization**](https://github.com/fmfn/BayesianOptimization)
*  [**SHAP**](https://github.com/slundberg/shap)
*  [**unumpy**](https://github.com/Quansight-Labs/unumpy)


### Data
All groundwater level data in its original form is available free of charge from the respective websites of the local authorities. However, we used data interpolated based on previous knowledge and therefore publish the used data [**HERE**](https://doi.org/10.5281/zenodo.4683879). Many thanks at this point for the support to the responsible state authorities, who provided us with the data and allowed a publication.

All climate data necessary to train the models (HYRAS) is available for free online: https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/
Please note that we partly used earlier versions of the HYRAS data, which may cause slight differences in training results. We are currently checking if we are allowed to provide our training forcing data. 

We cannot provide the clima projection forcing data. The  respective model data is available from [**EURO-CORDEX**](https://esgf-data.dkrz.de/projects/esgf-dkrz/) or after Downscaling with EPISODES directly (and for free for non-comercial purposes) from [**DWD**](https://www.dwd.de/DE/klimaumwelt/klimaforschung/klimaprojektionen/fuer_deutschland/fuer_dtld_rcp-datensatz_node.html).
