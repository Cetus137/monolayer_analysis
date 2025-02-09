import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import convolve
from skimage.filters import threshold_otsu
import tifffile as tiff
from skimage.morphology import skeletonize
from skimage.morphology import binary_closing ,disk , thin
def load_ilastik_segmentation(file_path):
    """
    Load segmentation output from ilastik software saved as a numpy file.

    Parameters:
    file_path (str): Path to the numpy file.

    Returns:
    np.ndarray: Segmentation data.
    """
    try:
        segmentation_data = np.load(file_path)
        return segmentation_data
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
def prune_skeleton(skeleton):
    # Define a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Iterate until no more changes
    # Count neighbors
    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
    
    # Find pixels with only one neighbor
    to_remove = (skeleton == 1) & (neighbor_count ==1)
    
    # Remove the pixels
    skeleton[to_remove] = 0
    
    return skeleton
# Example usage
file_path = './ilastik_output/Processed_data.npy'
segmentation_data = load_ilastik_segmentation(file_path)
print(segmentation_data)
print(segmentation_data.shape)
if segmentation_data is not None:
    print("Segmentation data loaded successfully.")
else:
    print("Failed to load segmentation data.")


filepath = r'./monolayer_analysis_test/experimental_data/20241101_water_MDCKLiveMonolayer_ZO1_mS_ZO2_NG/Processed_data/20241101_MDCKmonolayer_ZO1_mS_ZO2_NG_singlechannel.tif'  #C1-20241101_MDCKmonolayer_ZO1_mS_ZO2_NG_merged_singlecell.tif'
stack_data = tiff.imread(filepath)
for i in range(stack_data.shape[0]):
    cell_label = segmentation_data[-i,:,:,1]
    raw_data = stack_data[-i,:,:]
    # Assuming cell_label is a grayscale image
    threshold_value = threshold_otsu(cell_label)
    binary_image = cell_label > threshold_value
    labelled_image = skimage.measure.label(binary_image)
    cleaned_image = skimage.morphology.remove_small_objects(labelled_image, min_size =200)
    connected_cleaned_image = skimage.morphology.binary_closing(cleaned_image,disk(2))
    skeleton_mask = skimage.morphology.skeletonize(connected_cleaned_image).astype(int)
    pruned_skeleton = prune_skeleton(skeleton_mask) # Adjust max_iter as needed
    plt.imshow(connected_cleaned_image)
    plt.show()
    plt.imshow(pruned_skeleton, alpha=1 , cmap='Reds')
    plt.colorbar()
    plt.show()