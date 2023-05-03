# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:15:05 2023

@author: lukas
"""

import os

folder_path = os.getcwd()
old_string = "preliminary"
new_string = "local"

#%%
# list all files in the folder
for filename in os.listdir(folder_path):
    # check if the file matches the pattern
    if filename.startswith("v_2040_") and filename.endswith(".nc") and old_string in filename:
        # construct the new file name
        new_filename = filename.replace(old_string, new_string)
        # rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))


#%%

# import os

# folder_path = r"C:\Users\lukas\Documents\GitHub\Masters_Thesis_NorthSeaEnergyIsland\model\v_2030_local_nac\results"  # replace with your folder path
# old_string = "preliminary"
# new_string = "local"
# nac_string = "_nac"

# # list all files in the folder
# for filename in os.listdir(folder_path):
#     # check if the file matches the pattern with "preliminary"
#     if filename.startswith("v_2030_") and filename.endswith(".nc") and old_string in filename:
#         # construct the new file name
#         new_filename = filename.replace(old_string, new_string)
#         # rename the file
#         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    
#     # check if the file matches the pattern without "preliminary"
#     if filename.startswith("v_2030_") and filename.endswith(".nc") and old_string not in filename:
#         # check if the file has the "nac" part
#         if nac_string not in filename:
#             # construct the new file name with the "nac" part added
#             new_filename = filename.replace(new_string, new_string + nac_string)
#             # rename the file
#             os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        
