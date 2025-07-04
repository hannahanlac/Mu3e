# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:24:19 2024

@author: nicky
"""

import uproot 
import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD:NCodes/OldCodes/events_per_frame.py
file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_96_32652_execution_1_run_num_789062_sort.root")
=======
file = uproot.open("C:/Users/m4joh/OneDrive/University/Mu3e/Raw Simulation Data/signal1_0_1944629_execution_1_run_num_407942_vertex.root")
>>>>>>> f2b7602 (Raw Data Folder Name Change):N Codes/events_per_frame.py
vertices = file["vertex"].arrays()

#print(dir(vertices[0]))

print(len(vertices))

print('Test')

### Code attempting to plot the number of mass events per frame by number of frames## 

frames= np.arange(0, len(vertices)) # Total number of frames array
number_list = [] # List to count through and ammend the number of mass events

# for i in range(3800, 4500):
#     print(len(vertices[i].mmass))

for i in range(len(vertices)):           # Iterate through frames and count mass events, and add
    number_per_frame = len(vertices[i].mmass)
   # print(number_per_frame)
    number_list.append(number_per_frame) 
number_array = np.array(number_list) # Final array of number of events 

print(np.max(number_array))

# for i in range(0, 100):
#     print(len(vertices[i].mmass))
#     print(vertices[i].mmass)

# for i in range(20):
#     print(vertices[i].mmass)

print(np.sum(number_array))

# Create the bar chart
plt.bar(frames, number_array)
plt.ylim(0, 25)
plt.xlim()
# Add labels and title
plt.xlabel('Frame')
plt.ylabel('Number of events')
plt.title('Number of events for each frame')

# Show the chart
plt.show()



