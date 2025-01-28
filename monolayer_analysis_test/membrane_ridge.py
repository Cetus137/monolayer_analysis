import matplotlib.pyplot as plt
import skimage
from skimage import exposure
from skimage.filters import meijering, hessian, unsharp_mask, gaussian, threshold_otsu
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, shift, maximum_filter
from skimage.morphology import binary_closing ,disk
import tifffile as tiff
import numpy as np
import networkx as nx
import trackpy as tp
import pandas as pd

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
    cleaned_image = skimage.morphology.remove_small_objects(labelled_image, min_size =100)
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
    # 1. Create a graph from the skeleton
    graph = nx.Graph()
    for y, x in np.transpose(np.nonzero((skeleton_mask))):  # Extract skeleton pixel coordinates
        graph.add_node((y, x))  # Add each pixel as a node

        # Connect neighboring pixels
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = ((y + dy) % skeleton_mask.shape[0], (x + dx) % skeleton_mask.shape[1]) 
            if skeleton_mask[neighbor]:  # Check if the neighbor is part of the skeleton
                graph.add_edge((y, x), neighbor)

    branch_points = [node for node in graph.nodes if graph.degree[node] > 2]
    endpoints = [node for node in graph.nodes if graph.degree[node] == 1]
    return graph , branch_points, endpoints

def track_nodes(nodes, max_distance):
    """
    Track nodes across frames based on their proximity between consecutive frames.
    
    Args:
        nodes (dict): Dictionary where keys are frame numbers and values are lists of (x, y) coordinates.
        max_distance (float): Maximum distance a node can move between frames.
        
    Returns:
        list: A list of tracks, where each track is a list of coordinates for that node across frames.
    """
    tracks = []  # List to store tracks
    
    # Sort frames in ascending order
    frames = sorted(nodes.keys())
    
    # Initialize tracks with the nodes in the first frame
    for node in nodes[frames[0]]:
        tracks.append([node])
    
    # Iterate through consecutive frames
    for i in range(len(frames) - 1):
        current_frame = frames[i]
        next_frame = frames[i + 1]
        
        current_nodes = nodes[current_frame]
        next_nodes = nodes[next_frame]
        
        # Keep track of which nodes in the next frame have been matched
        matched_next = set()
        
        # Match nodes from the current frame to the next frame
        for track in tracks:
            last_node = track[-1]
            if last_node is None:
                continue
            
            # Find the closest node in the next frame within the max_distance
            min_distance = float('inf')
            best_match = None
            for j, next_node in enumerate(next_nodes):
                if j in matched_next:
                    continue  # Skip already matched nodes
                
                distance = np.linalg.norm(np.array(last_node) - np.array(next_node))
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    best_match = j
            
            # Append the matched node to the track, or None if no match found
            if best_match is not None:
                track.append(next_nodes[best_match])
                matched_next.add(best_match)
            else:
                track.append(None)
        
        # Handle unmatched nodes in the next frame (start new tracks)
        #for j, next_node in enumerate(next_nodes):
        #    if j not in matched_next:
        #        tracks.append([None] * (i + 1) + [next_node])
        
    tracks_dict = {}
    print(len(tracks))

    for i in range(len(tracks)):
        track_list= tracks[i]
        #print(track_list)
        track_x = np.array([])
        track_y = np.array([])
        for j in range(len(track_list)):
            if track_list[j] is not None:
                #print(track_list[j][0])
                track_x = np.append(track_x , np.array([track_list[j][0][0]]), axis = 0) 
                track_y = np.append(track_y , np.array([track_list[j][0][1]]), axis = 0) 
            else:
                track_x = np.append(track_x , np.array([np.nan]), axis = 0) 
                track_y = np.append(track_y , np.array([np.nan]), axis = 0) 

        track_coords = np.zeros( (2 , len(track_x) ) )
        track_coords[0] = track_x
        track_coords[1] = track_y
        tracks_dict[i] = track_coords

    return tracks_dict

def create_node_image(nodes_dict , stack_data):
    node_image = np.zeros_like(stack_data)
    for i in range(len(nodes_dict)):
        frame_nodes = nodes_dict[i]
        for j in range(len(frame_nodes)):
            node_image[i,int(frame_nodes[j][0]),int(frame_nodes[j][1])] += 1
    return node_image

#def get_edge_lengths(nodes, skeleton_mask)

filepath = r'./experimental_data/20241101_water_MDCKLiveMonolayer_ZO1_mS_ZO2_NG/Processed_data/singlemembrane_zo1.tif'  #C1-20241101_MDCKmonolayer_ZO1_mS_ZO2_NG_merged_singlecell.tif'
stack_data = tiff.imread(filepath)
skeleton_stack = np.zeros_like(stack_data)
#stack_data = stack_data[25:30,:,:]
nodes_dict = {}
for i in range(np.shape(stack_data)[0]):
    raw_data = stack_data[i,:,:]
    skeleton_mask , connected_cleaned_image , ridges, sharp_ridges = image_processing(raw_data)
    skeleton_stack[i,:,:] = skeleton_mask
    graph, branch_points , endpoints = get_branch_points(skeleton_mask)
    # Merge nodes within 3 pixel distance
    merged_graph = graph.copy()  # Create a copy to modify
    node_coords = branch_points
    node_groups, node_merged = find_close_nodes(node_coords, 3)

    '''
    plt.imshow(skeleton_mask)
    plt.scatter([x[1] for x in branch_points], [x[0] for x in branch_points], c='r', s=10)
    plt.show()
    plt.imshow(raw_data[:,:])
    for node in node_merged:
        plt.scatter(node[1], node[0], c='r', s=10)
    plt.show()
    '''
    

    #length = nx.shortest_path_length(graph , source = node_merged[0] , target = node_merged[1])
    nodes_dict[i] = branch_points #node_merged

node_image = create_node_image(nodes_dict , stack_data)
#location_df = tp.batch(255 * node_image[1,:,:],11,minmass=2, invert = False)
location_df = pd.DataFrame()
for i in range(len(nodes_dict)):
    frame_df = tp.locate(255 * node_image[i,:,:], 5 , minmass=20.0, invert = False)
    frame_df['frame'] = i
    location_df = pd.concat([location_df , frame_df])
    print(location_df)
#pred = tp.predict.NearestVelocityPredict()
#t = pred.link_df(location_df, 5, memory = 5)
t = tp.link_df(location_df, 5, memory = 2) #, predictor = tp.predict.NearestVelocityPredict())

length = np.ones(np.shape(stack_data)[0]) * np.nan
inter_node_distance = np.ones(np.shape(stack_data)[0]) * np.nan
for i in range(np.shape(stack_data)[0]):
        graph , ignore1, ignore2  = get_branch_points(skeleton_stack[i,:,:])
        frame_df = t[t['frame'] == i]
        print(frame_df)
        
        if 0 in frame_df['particle'].values and 2 in frame_df['particle'].values:
            source_df = frame_df[frame_df['particle'] == 0]
            target_df = frame_df[frame_df['particle'] == 2]
            
            if i % 15 == 0:
                fig , ax = plt.subplots(1,2)
                ax[0].imshow(stack_data[i,:,:])
                ax[0].plot(source_df['x'], source_df['y'] , 'x')
                ax[0].plot(target_df['x'], target_df['y'] , 'x')
                ax[1].imshow(skeleton_stack[i,:,:])
                ax[1].plot(source_df['x'], source_df['y'] , 'x')
                ax[1].plot(target_df['x'], target_df['y'] , 'x')
                plt.show()
            
            
        
            
            try:
                length[i] = nx.shortest_path_length(graph , source = (int(np.round(source_df['y'] , 0)), int(np.round(source_df['x'], 0))) , target = (int(target_df['y']), int(target_df['x'])))
                inter_node_distance[i] = np.linalg.norm(np.array([source_df['y'], source_df['x']]) - np.array([target_df['y'], target_df['x']]))
            except (nx.NetworkXNoPath , nx.NodeNotFound):
                length[i] = np.nan
                inter_node_distance[i] = np.nan
        
        '''
        fig , ax = plt.subplots(1,2)
        ax[0].imshow(stack_data[i,:,:])
        ax[1].imshow(skeleton_stack[i,:,:])
        for j in frame_df['particle'].values:
            source_df = frame_df[frame_df['particle'] == j]
            ax[0].plot(source_df['x'], source_df['y'] , 'x')
            ax[1].plot(source_df['x'], source_df['y'] , 'x')
        plt.show()
        '''
            

    
plt.plot(length, linewidth = 2, label = 'membrane length')
plt.plot(inter_node_distance, linewidth = 2, label = 'inter-node distance')
plt.legend()
plt.xlabel('Frame (2 minutes)')
plt.ylabel('membrane length estimate')
plt.show()


'''
for i in range(len(nodes_dict)):
    plt.imshow(stack_data[i,:,:])
    frame_df = t[t['frame'] == i]
    for j in range(len(frame_df)):
        plt.scatter(frame_df['x'], frame_df['y'], c = 'r')
    plt.show()
'''








