## NISAR Prototype Notebooks

**This directory is intended for members of the NISAR Science Team only.**

[NISAR](https://nisar.jpl.nasa.gov/) is an up-and-coming "dedicated U.S. and Indian InSAR mission, in partnership with ISRO, optimized for studying hazards and global environmental change".
Members of the NISAR science team are developing high level products able to process the enormous amount of SAR data the mission is expected to produce.
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
1. Attach the files to an email with the following information

    ```
    TO: uaf-jupyterhub-asf@alaska.edu
    SUBJECT: NISAR Science Team GitHub PR
    BODY: [Add any instruction as needed. For example, "Please add the following files to the NISAR directory within https://github.com/asfadmin/asf-jupyter-notebooks." Remember that any changes will be semi-permanent.]
    ```

1. An administrator from ASF will review the changes and merge the request.


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

Using the email address used during OpenSARLab signup, send an email:

```
    TO: uaf-jupyterhub-asf@alaska.edu
    SUBJECT: NISAR Science Team GitHub PR
```

An administrator from ASF will review the changes and merge the request.
