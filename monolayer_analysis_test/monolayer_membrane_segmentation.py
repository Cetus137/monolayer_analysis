import matplotlib.pyplot as plt
import skimage
from skimage import exposure
from skimage.filters import meijering, hessian, unsharp_mask, gaussian, threshold_otsu
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, shift, maximum_filter , convolve
from skimage.morphology import binary_closing ,disk
import tifffile as tiff
import numpy as np
import networkx as nx
import trackpy as tp
import pandas as pd


### A function to take in a monolayer image and output a skeletonized image
def image_processing(raw_data):
    dims = np.shape(raw_data)
    data = gaussian(raw_data , sigma=0)
    data = data/np.max(data)
    data = exposure.equalize_adapthist(data)
    sharp_data = unsharp_mask(data, radius=1.5 , amount=1)
    ridges = hessian(data, sigmas =[1,2] ,black_ridges=True)
    sharp_ridges = hessian(sharp_data, sigmas =[1,2,3] , black_ridges=True)
    thresh = threshold_otsu(ridges)
    binary_mask = ridges > thresh
    labelled_image = skimage.measure.label(binary_mask)
    cleaned_image = skimage.morphology.remove_small_objects(labelled_image, min_size =200)
    connected_cleaned_image = skimage.morphology.binary_closing(cleaned_image,disk(2))
    skeleton_mask = skimage.morphology.skeletonize(connected_cleaned_image).astype(int)
    '''
    plt.imshow(raw_data)
    plt.show()
    plt.imshow(ridges)
    plt.show()
    '''
    return skeleton_mask , connected_cleaned_image , ridges , sharp_ridges

def node_finder(skeleton_image):
    shifts = [[0,1], [0,-1], [1,0] , [-1, 0] , [1,1] , [1,-1], [-1,1] , [-1,-1]]
    sum = np.zeros_like(skeleton_image)
    for s in shifts:
        print(s)
        shifted_image = shift(skeleton_image,s,mode = 'grid-wrap')
        #plt.imshow(shifted_image)
        #plt.show()
        sum +=shifted_image
    maximum_partners = np.max(sum)

    return sum
# Function to find all nodes within a threshold distance
def find_close_nodes(node_coords, max_distance):
    groups = []
    visited = set()  # To keep track of visited nodes
    
    for node in node_coords:
        if node not in visited:
            # Start a new group with the current node
            group = [node]
            visited.add(node)
            
            # Check all other nodes
            for other_node in node_coords:
                if other_node != node and other_node not in visited:
                    # Calculate the Euclidean distance
                    distance = np.linalg.norm(np.array(node) - np.array(other_node))
                    if distance <= max_distance:
                        group.append(other_node)
                        visited.add(other_node)
            
            # Add the group to the list of groups
            groups.append(group)
    averaged_nodes = []
    for group in groups:
        averaged_node = tuple( np.round((np.mean(np.array(group), axis=0)) ,0))
        averaged_nodes.append((averaged_node))

    return groups , averaged_nodes

def get_branch_points(skeleton_mask):
    image = skeleton_mask.copy()
    #3x3 kernel to collect the region around each pixel
    kernel  = np.ones((11,11) , dtype=int)
    #the junctions will have at least 6 pixels in the region
    #of a 3x3 square centred on the junction pixel.
    neigh_sum = convolve(image, kernel, mode='constant', cval=0.0)
    junctions = np.argwhere(neigh_sum > 15)
    return junctions , neigh_sum



filepath = r'./monolayer_analysis_test/experimental_data/20241101_water_MDCKLiveMonolayer_ZO1_mS_ZO2_NG/Processed_data/20241101_MDCKmonolayer_ZO1_mS_ZO2_NG_singlechannel.tif'  #C1-20241101_MDCKmonolayer_ZO1_mS_ZO2_NG_merged_singlecell.tif'
stack_data = tiff.imread(filepath)
skeleton_stack = np.zeros_like(stack_data)
#stack_data = stack_data[25:30,:,:]
nodes_dict = {}
for i in range(np.shape(stack_data)[0]):
    raw_data = stack_data[i,:,:]
    skeleton_mask , connected_cleaned_image , ridges, sharp_ridges = image_processing(raw_data)
    skeleton_stack[i,:,:] = skeleton_mask
    junctions = get_branch_points(skeleton_mask)
    print(junctions[0])
    plt.imshow(skeleton_mask)
    plt.scatter(junctions[0][:, 1], junctions[0][:, 0], alpha=0.5, color='red')
    plt.show()

    print(junctions)


