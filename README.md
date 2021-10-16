# ASF Jupyter Notebooks

Jupyter notebooks provide an easy way to play with data recipes and visualize data. They also provide a low-barrier entry point into general Python development. [See official documentation](https://jupyter.org/) for more info about the overlying technology used.

This repo contains Jupyter notebooks developed to explore Synthetic Aperature Radar (SAR) data hosted at ASF in an user-friendly way. To explore what data are available to manipulate and analyze, visit https://search.asf.alaska.edu.

Many of the notebooks contained in this repo are built to run within certain [conda environments](https://github.com/ASFOpenSARlab/opensarlab-envs). Since some required software cannot be found within conda, system installations may be needed. [These docker files](https://github.com/ASFOpenSARlab/opensarlab-docker) (particularly the dockerfiles) may be useful as a guide for system software requirements.

To help users create conda environments, [this notebook](https://github.com/ASFOpenSARlab/opensarlab-envs/blob/main/Create_OSL_Conda_Environments.ipynb) has been created and most notebooks have the needed conda environment referenced in their metadata. Users may find it easier to build their conda environments within OpenSARlab using the above mentioned notebook at the following path: `/home/jovyan/conda_environments/Create_OSL_Conda_Environments.ipynb`.

Any questions and improvement recommendations can be directed to **uaf-jupyterhub-asf+notebooks@alaska.edu**.
