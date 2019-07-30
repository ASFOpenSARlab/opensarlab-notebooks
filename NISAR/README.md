## NISAR Prototype Notebooks

**This directory is intended for members of the NISAR Science Team only.**

"Using advanced radar imaging that will provide an unprecedented, detailed view of Earth, the NASA-ISRO Synthetic Aperture Radar, or [NISAR](https://nisar.jpl.nasa.gov/nisarmission/), satellite is designed to observe and take measurements of some of the planet's most complex processes, including ecosystem disturbances, ice-sheet collapse, and natural hazards such as earthquakes, tsunamis, volcanoes and landslides."
To support the mission, members of the NISAR science team are developing high level products able to process the enormous amount of SAR data the mission is expected to produce.
Prototypes of the products are being created and shared using Jupyter Notebooks ran on ASF's Jupyter Hub Platform.
While this repo is open to the public, this directory and it's contents are intended for use only by NISAR Science Team members.

### Structure of Directory

The structure of the directory is to be decided by the NISAR Science Team. Note that any changes in this repo are semi-permanent.

### Example Recipes

Example SAR recipes can be found in the [SAR Training section](https://github.com/asfadmin/asf-jupyter-notebooks/tree/master/SAR_Training) of the repo. Modified versions of these recipes can be saved within the NISAR directory via a Pull Request.

### Adding Files To The GitHub Repo

There are two ways to semi-permanently add files to this directory:

#### Email files to ASF

1. Download from _opensarlab_ the wanted notebooks and accompanying files
1. Using the email used during signup, send an email to ASF with the following information:

    ```
    TO: uaf-jupyterhub-asf@alaska.edu
    SUBJECT: NISAR Science Team GitHub PR
    BODY: [Add any instruction as needed. For example, "Please add the following files to the NISAR directory within https://github.com/asfadmin/asf-jupyter-notebooks." Remember that any changes will be semi-permanent.]
    ```

1. An administrator from ASF will review the changes.
   If there are any concerns with the PR, the admin will communicate via email.
   If the PR looks good, the admin will merge the request. Confirmation will be sent via an email.


#### Submit a GitHub Pull Request
For those comfortable with using Git, submit your changes as a pull request:

1. git checkout master
1. git pull origin master
1. git checkout -b *my_custom_branch_name*
1. **Add changes**
1. git status  # to see changes
1. git add .
1. git commit -m "Update files for the NISAR directory"
1. git push origin master

Go to https://github.com/asfadmin/asf-jupyter-notebooks, select the *my_custom_branch_name* branch and create a Pull Request.

Using the email used during signup, send an email to ASF with the following information:

```
    TO: uaf-jupyterhub-asf@alaska.edu
    SUBJECT: NISAR Science Team GitHub PR
    BODY: [copy-paste the URL to the Pull Request here] has been created. Please review and merge.
```

An administrator from ASF will review the changes.
If there are any concerns with the PR, the admin will communicate via email.
If the PR looks good, the admin will merge the request. Confirmation will be sent via an email.
