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

    return 



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
        print(label)
        mem_seg_image = seg_image.copy()
        mem_seg_image[mem_seg_image == label] = label
        
        mem_seg_image[mem_seg_image != label] = 0
        membrane_intensity(mem_seg_image)
