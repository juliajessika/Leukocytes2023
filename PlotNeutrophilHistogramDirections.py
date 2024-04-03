# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:13:31 2024

@author: julia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:12:11 2021

@author: Julia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_scatter_density


"""This is a piece of code which allows for plotting a neutrophil direction histogram"""

#path to sorted trajectory files here. The piece of code would process all the subfolders
mypath = r'C:\Users\julia\OneDrive\Desktop\SignLab\Chemotaxis\Directions\dir_vWF'
from os import listdir
from os.path import isfile, join
from os import walk, path
import os
import re
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

files_all = []
for path, subdirs, files in walk(mypath):
    for name in files:
        files_all.append(os.path.join(path, name))
print(files_all)  
pixsize = 0.46      
phis = []
phis = list(phis)
        
cordx = []
cordy = []           
import math    
from matplotlib.colors import LinearSegmentedColormap

#the directory must contain a file with the names of all experiments with diverging flow directions (eg for us some students sometimes flipped the flow chamer)
with open(mypath + '/list_of_flipped_flow_patients.txt', 'r') as file:
    # Read the first line
    first_line = file.readline().strip()

extensionsToCheck = first_line.split(',')

#extensionsToCheck = ['julia','SDS_51976','Papin', 'sere

n_unique = {}

for iters in next(walk(mypath))[1]:
    n_unique[iters] = []
for files in files_all:
  if any(ext in files for ext in extensionsToCheck):
      flow = 1
  else:
      flow = -1
     
  if ('field' in files)and('xls' in files):
    #try:   
     
    data = pd.read_excel(files, sheet_name = 'Coords')
    data_vels = pd.read_excel(files, sheet_name = 'Velocity')
    len_datasheet = []
    for col in data.columns: 
        if re.search('Coords', col):
            
            len_datasheet.append(int(re.sub("[^0-9]", "", col)))
    phis_thisfile = []    
    if (len_datasheet):
        phis_thisfile = []
        len_max = np.max(len_datasheet)  
        for enum in range (1,len_max+1):
            namex = 'Coords X №'+str(enum)
            namey = 'Coords Y №'+str(enum)
            len_1 = (data[namex].size)
            phis_current = []
            #plt.plot([data[namex]-data[namex][1],(data[namey]-data[namey][1])])  
            if (np.mean(data_vels['Velocity №'+ str(enum)])<0.5)and(np.mean(data_vels['Velocity №'+ str(enum)])>0.03 ):
                  
                for m in range (2,len_1): 
                    
                    vel_x = -(data[namex][m-1]-(data[namex])[m])*flow
                    vel_y = -(data[namey][m-1]-(data[namey])[m])
                    if ((not(np.isnan(vel_x))) and(not(np.isnan(vel_y)))and (not((np.abs(vel_x)<0.001)and(np.abs(vel_y)<0.001)))):
                    
                        angle = np.arctan2(vel_y, vel_x) 
                    
                    #angle = (np.rad2deg(angle)) 
                        phis_current.append(angle)
                    if not ((np.isnan(data[namex][m-1]-data[namex][0])) or (np.isnan((data[namey][m-1]-data[namey][0])  ))):
                        cordx.append((data[namex][m-1]-data[namex][0])*flow)
                        cordy.append(data[namey][m-1]-data[namey][0])    
                #plt.plot(cordx,cordy) 
                
        
                #fig.colorbar(density, label='Number of points per pixel')    
                try:   
                        phis_current =  list(phis_current)
                        
                        if (len(phis_current)>2):
                            phis_current =  list(np.random.choice(phis_current,121))
                        
                            phis_thisfile.append(phis_current)
                        #else:
                            #phis_thisfile.append(phis_current)
                except:
                        phis_current =  list(phis_current)
        
  
            try: 
                   if (len(phis_thisfile)>2):
                       #choice_indices = np.random.choice(len(phis_thisfile), 10)
                       #phis_thisfile = [phis_thisfile[i] for i in choice_indices]
                       phis_thisfile = phis_thisfile
                     
                       for key in n_unique.keys():
                           if key in files:
                               for items in phis_thisfile:
                                   for subitems in items:
                             
                                       n_unique[key].append(subitems) 
                   
                       
            except Exception as e:
                   print(e)
               
print(n_unique)                
fig = plt.figure(figsize=(6, 6))
import matplotlib
import copy

data = np.arange(25).reshape((5,5))
my_cmap = copy.copy(matplotlib.cm.get_cmap('jet')) # copy the default cmap
from matplotlib import cm
jet = cm.get_cmap('jet')
my_cmap.set_bad(jet(0))
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')   
density = ax.hist2d(cordx, cordy, bins = [250,250], cmap = my_cmap, norm=matplotlib.colors.LogNorm())        
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10) 

def formatter2(x, pos):
    del pos
    return str(int(np.round(x*pixsize)))

ax.yaxis.set_major_formatter(formatter2)

ax.xaxis.set_major_formatter(formatter2)
ax.set_yticks([5/0.46,10/0.46,0,-10/0.46,-5/0.46])

ax.set_xticks([10/0.46,5/0.46,0,-10/0.46,-5/0.46])
plt.show()
plt.figure()         


for key in n_unique.keys():
    if (len(n_unique[key])>0):
        phis_choice = list(np.random.choice(n_unique[key],1625))
        for items in phis_choice:
            phis.append(items)

phis = [x for x in phis if (math.isnan(x) == False)] 
phis = list(phis)           
plt.locator_params(nbins=12) 
ax = plt.subplot(111, polar = True)

ax.tick_params(axis='both', which='major', labelsize = 0)  
ax.set_yticks([0.1,0.2])
ax.set_xticks([0,np.pi/2,np.pi,np.pi*3/2])
ax.set_ylim(0,0.3)
ax.grid(linewidth = 3, alpha = 1, color = 'black')
ax.spines[:].set_visible(False)
#plt.yticks([])
#circular_hist(ax,phis)   


plt.hist(phis, fill=False, color = 'blue', edgecolor = 'black',hatch='..',  histtype='step', bins = 15, density=True, alpha = 0.9, linewidth = 3) 

def formatter(x, pos):
    del pos
    return str(int(np.ceil(x*100)))+ '%'

ax.yaxis.set_major_formatter(formatter)

 