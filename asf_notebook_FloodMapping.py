# asf_notebook_FloodMapping.py
# Alex Lewandowski
# 9-16-2020
# Module of Alaska Satellite Facility OpenSARLab Jupyter Notebook helper functions
# Minimal version of asf_notebook.py containing only the functions needed for FloodMappingFromSARImages.ipynb
# 

import os

class NoHANDLayerException(Exception):
    """
    Raised when expecting path to HAND layer but none found
    """
    pass
    
def input_path(prompt):        
    print(f"Current working directory: {os.getcwd()}") 
    print(prompt)
    return input()


def handle_old_data(data_dir, contents):
    print(f"\n********************** WARNING! **********************")
    print(f"The directory {data_dir} already exists and contains:")
    for item in contents:
        print(f"• {item.split('/')[-1]}")
    print(f"\n\n[1] Delete old data and continue.")
    print(f"[2] Save old data and add the data from this analysis to it.")
    print(f"[3] Save old data and pick a different subdirectory name.")
    while True:
        try:
            selection = int(input("Select option 1, 2, or 3.\n"))
        except ValueError:
             continue
        if selection < 1 or selection > 3:
             continue
        return selection
