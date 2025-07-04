# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:41:20 2024

@author: nicky
"""

import uproot 
import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD:NCodes/OldCodes/mass_frequency_plotting.py
file = uproot.open("/root/Mu3eProject/RawData/signal1_1_1944629_execution_1_run_num_836827_vertex.root")
=======
file = uproot.open("C:/Users/m4joh/OneDrive/University/Mu3e/Raw Simulation Data/signal1_0_1944629_execution_1_run_num_407942_vertex.root")
>>>>>>> f2b7602 (Raw Data Folder Name Change):N Codes/mass_frequency_plotting.py
vertices = file["vertex"].arrays()


######## Playing around with uproot calling commands #######
print(vertices.fields)
#print()
#print(vertices.true_px1.show)

# number_list = [] # List to count through and ammend the number of mass events

# for i in range(3800, 4500):
#     print(len(vertices[i].mmass))

#############################################################################


## Code to collect all the mass events into a single array ################

overall_mass_array = np.empty(0) # Initializes an empty array to populate with the masses

for i in range(len(vertices)):           # Iterate through frames and count mass events, and add
   frame_mass_array = (vertices[i].mmass)
    # print(number_per_frame)
   overall_mass_array = np.append(overall_mass_array,frame_mass_array) # Append the mass array with the new frame values

#print(overall_mass_array) # Overall mass array for all of the events recorded in the data
#print(vertices[1].mmass)
#print(file["vertex"][3].mmass)

#np.sort(overall_mass_array)
print(overall_mass_array)

bin_width = 2
bin_edges = np.arange(min(overall_mass_array), max(overall_mass_array) + bin_width, bin_width)
# 
plt.hist(overall_mass_array, bins=bin_edges, edgecolor = 'black')
plt.xlabel('Mass (MeV/C^2)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Event Mass Frequency', fontsize=16)

plt.show()
#print(vertices.show)
#print(file.keys())
#print(file.classnames())
#print(dir(vertices[0]))

#print(len(vertices))

#print(vertices.show)

### Code attempting to plot the number of mass events per frame by number of frames## 