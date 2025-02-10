from ilastik_MembraneNetwork2segmentation import get_branch_points
from ilastik_MembraneNetworkOperations import sum_surrounding_points , set_pixels_within_distance_to_zero , set_pixels_within_distance
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.morphology import label , skeletonize

def membrane_intensity(membrane_seg):
    skeleton_seg = skeletonize(membrane_seg)
    raw_coords = np.argwhere(membrane_seg)
    raw_coords = np.array(raw_coords)
    # Extract the coordinates of the skeleton pixels
    skeleton_coords = np.argwhere(skeleton_seg)
    skeleton_coords = np.array(skeleton_coords)
    for i in range(len(skeleton_coords)):
        coord = skeleton_coords[i]
        sum = sum_surrounding_points(skeleton_seg, coord, 1)
        if sum == 2:
            edge_coord = coord
    def find_closest_coordinates(coord, coords_list):
        distances = np.sqrt((coords_list[:, 0] - coord[0])**2 + (coords_list[:, 1] - coord[1])**2)
        for i in range(len(coords_list)):
            if coords_list[i,0] == coord[0] and coords_list[i,1] == coord[1]:
                distances[i] = np.inf
        closest_indices = np.argmin(distances)
        closest_coords = tuple(coords_list[closest_indices])
        return [closest_coords]
    
    smoothed_coords = average_line_points(skeleton_coords, window_size=3)

    processed_coords = []
    sorted_coords = []

    #find the neighbour of the edge_coord
    current_coord = tuple(edge_coord)
    for c in range(len(skeleton_coords)):
        if c== len(skeleton_coords)-1:
            sorted_coords.append(tuple( [np.min(skeleton_coords[:,0], axis=0) , np.min(skeleton_coords[:,1])] ))
            break
        else: 
            sorted_coords.append(current_coord)
            processed_coords.append(tuple(current_coord))
            neighbors = (find_closest_coordinates(current_coord, skeleton_coords))
            neighbors = [n for n in neighbors if n not in processed_coords]
            if not neighbors:
                print('brokem')
                break
            for i in range(len(skeleton_coords)):
                if skeleton_coords[i,0] == current_coord[0] and skeleton_coords[i,1] == current_coord[1]:

                    skeleton_coords[i,0] = 1e5
                    skeleton_coords[i,1] = 1e5
            current_coord = neighbors[0]

    sorted_coords = np.array(sorted_coords)
    plt.imshow(skeleton_seg * raw_data_normal)
    plt.plot(sorted_coords[:,1] , sorted_coords[:,0])
    plt.plot(raw_coords[:,1] , raw_coords[:,0])
    plt.show()
    return sorted_coords



if __name__ == "__main__":
    # Example usage
    raw_filepath = './monolayer_analysis_test/experimental_data/20241101_water_MDCKLiveMonolayer_ZO1_mS_ZO2_NG/Processed_data/20241101_MDCKmonolayer_ZO1_mS_Magenta_frame1.tif'
    seg_ilastik_path = './monolayer_analysis_test/ilastik_output/20241101_MDCKmonolayer_ZO1_mS_Magenta_frame1_Multicut Segmentation_edges.npy'
    mem_seg_image = np.load(seg_ilastik_path)
    raw_data = tiff.imread(raw_filepath)
    raw_data_normal = (raw_data - np.min(raw_data) )/ np.max(raw_data)
    centres , junction_image , labeled_membranes , seg_image= get_branch_points(mem_seg_image, 2)
    for i in np.unique(seg_image):
        label = i+40
        mem_seg_image = seg_image.copy()
        mem_seg_image[mem_seg_image == label] = label
        mem_seg_image[mem_seg_image != label] = 0
        membrane_intensity(mem_seg_image)
