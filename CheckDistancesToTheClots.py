# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:52:16 2023

@author: Julia
"""
from typing import List
import os
import numpy as np
from scipy.ndimage import distance_transform_edt

import pandas as pd    


import xlrd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import random
import h5py


def init_new_traj(image: np.ndarray) -> List[np.ndarray]:
    size = np.shape(image)
    
    coords = [random.randrange(0, size[0]),random.randrange(0, size[1])]
    coords = [np.array(np.int32(np.floor(coords)))]
    return coords

def flatten_video(data: np.ndarray) -> np.ndarray:
    data_clots = np.zeros_like(data)
    
    data_clots[data==2]+=2
    data_clots[data_clots!=2]*=0
    data_clots = np.median(data_clots,axis = 0)
    return data_clots

def move_random_ellipse(coords, image: np.ndarray, iter_num = 0):
    iter_num+=1
   
    if (coords.any()==None):
        
        return init_new_traj(image)
    coords_0 = coords.copy()
    coords = np.array(coords) + np.array([random.randrange(-3, 3),random.randrange(-3, 3)])
    
    coords = coords
    
    #make_random_circle = cv2.ellipse(make_random_circle, coords, (5,5), random.randrange(0, 360), 0, 360, 255, -1)
    #circle = make_random_circle!=0
    #leuco_centers = image != 0
    #if not (np.logical_and(circle,leuco_centers)).any():
    if True:
        return(coords)
    elif (iter_num<10):
        move_random_ellipse(coords_0, image, iter_num)
    

def calculate_min_distance(image_path, cell_coordinates):
    #image = np.array(Image.open(image_path))
    
    file = h5py.File(image_path, 'r')
    
    # Read an image from the file (replace 'image_data' with the actual dataset name in your file)
    image_data = file['exported_data'][:,:,:,0]
    image = flatten_video(image_data)
    true_clot = np.count_nonzero(image==2)
    
    percent_clot = true_clot/(np.shape(image)[0]*np.shape(image)[1])
    image_raw = image.copy()
    image = np.where(image == 2, 0, 1) 
    if 2 in np.unique(image_raw):
        
        
        #print(image_path)
        clot_distances = distance_transform_edt(image)
        #print('cd',np.shape(clot_distances))
        min_distances = []
        min_distances_random = []
        
        coords_random = init_new_traj(image_raw)[0]
        for points in range(np.shape(cell_coordinates)[0]):
            
            coords_random = move_random_ellipse(coords_random,image_raw)[0]   
            
            coordinates = cell_coordinates[points]
            if True: 
                try:
                    min_distance_random = clot_distances[coords_random[0],coords_random[1]]
                
                    min_distances_random.append(min_distance_random)
                    
                except IndexError:
                    coords_random = init_new_traj(image_raw)[0]
                    min_distance_random = clot_distances[coords_random[0],coords_random[1]]
                
                    min_distances_random.append(min_distance_random)
                if coordinates.all():
                    
                    cell_distances = clot_distances[coordinates[1],coordinates[0]]
                    
                    min_distance = cell_distances
                    
                        
                    min_distances.append(min_distance)
          
        return min_distances, min_distances_random, percent_clot
    else:
        return None
    
def get_cell_coordinates_from_xls(file_path):
    wb = xlrd.open_workbook(file_path)
    sheet = wb.sheet_by_name('Coords')
    coordinates = []
    for i in range(sheet.ncols // 4):
        coordinates_current = []
        x_coords_col = sheet.col_values(i*4)[1:]
        y_coords_col = sheet.col_values(i*4 + 1)[1:]
        coords = [(x, y) for x, y in zip(x_coords_col, y_coords_col) if x or y]
        for q in coords:
           
            coordinates_current.append(np.int32(np.floor(q)))
        coordinates.append(coordinates_current)
    return coordinates
import re

def process_subfolder(folder_path):
    
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith("Simple Segmentation.h5")])
    xls_files = [re.sub('-clot_Simple Segmentation.h5','.xls', i) for i in image_files]
    xls_files = [re.sub('_Simple Segmentation.h5','.xls', i) for i in xls_files]
    min_distances  = []
    
    min_distances_random = []
    for image_file, xls_file in zip(image_files, xls_files):
       
        image_path = os.path.join(folder_path, image_file)
        xls_path = os.path.join(folder_path, xls_file)
        
        cell_coordinates = get_cell_coordinates_from_xls(xls_path)
        print(xls_file, 'Number of cells:', len(cell_coordinates))
        if len(cell_coordinates):
            for subcoordinates in cell_coordinates:
                q = calculate_min_distance(image_path, subcoordinates)
                if q:
                    if q[0]:
                        distances_currfile, distances_random, clot = q
                        
                        """    
                        for q in distances_currfile:
                            if q!=0:
                                min_distances.append(q)
                        for q in distances_random:
                            if q!=0:
                                min_distances_random.append(q)
                        """
                        if True:
                            min_distances_random.append(np.mean(distances_random))
                            min_distances.append(np.mean(distances_currfile))
    try: 
        pass               
        #min_distances_random =  list(np.random.choice(min_distances_random,45))  
        #min_distances =      list(np.random.choice(min_distances,45))      
    except:
        pass
        #print(min_distances)
        #print(f"Minimum distances from cells to clot in {image_file} at each time point are: {min_distances}")
    return np.array(min_distances), np.array(min_distances_random)


def filter_quantile(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def process_folder(folder_path_main):
    all_exp = []
    all_random = []
    for root, dirs, files in os.walk(folder_path_main):
        if dirs:
            for dir_name in dirs:
                #print(dir_name)
                q,p = process_subfolder(os.path.join(root, dir_name))
                
                q = filter_quantile(q,1)
                
                p = filter_quantile(p,1)
                q = np.array(q)
                p = np.array(p)
                for subitems in q:
                    
                    all_exp.append(subitems)
                for subitems in p:    
                    all_random.append(subitems)
        else:
             q,p = process_subfolder(folder_path_main)
             
             q = filter_quantile(q,1)
             
             p = filter_quantile(p,1)
             q = np.array(q)
             p = np.array(p)
             for subitems in q:
                 
                 all_exp.append(subitems)
             for subitems in p:    
                 all_random.append(subitems)
        return list(all_exp), list(all_random)    
            
# Set the path to the directory containing your images and data here
folder_path = os.path.join(os.getcwd(), 'health200s-1')

if (__name__ == "__main__"):
    plt.ylim([0,40])
    q,p = process_folder(folder_path)
    
    data1 = {'Experiment': q, 'Random':p}
    
    data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data1.items() ]))
    labels = [['Experiment', 'Random']]
    
    plt.figure(figsize=(10, 6))
    
    my_pal = {"Experiment": "r", "Random": (0.5,0.5,1,1), "virginica":"m"}
    

    sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15}) 
    fig, ax = plt.subplots(figsize=(6, 6))
    box_plot = sns.boxplot(data=data, ax=ax, width=0.5, fliersize=5, palette=my_pal, linewidth = 3)
    

    # Set the labels
    ax.set_ylabel('Distance, au')
    
    
    plt.show()
    pval =stats.mannwhitneyu(p,q)[1]
    
    
    data.to_excel(os.path.join(folder_path, 'results.xls'), index=False)
    print(pval)
    # Add in individual points with swarmplot
    #sns.swarmplot(data=data, color=".25")
