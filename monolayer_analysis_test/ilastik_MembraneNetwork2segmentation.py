import tifffile as tiff
import numpy as np
from skimage.morphology import  label 
from skimage import measure
from scipy.ndimage import convolve , center_of_mass
import matplotlib.pyplot as plt
from ilastik_MembraneNetworkOperations import combine_labeled_images , set_pixels_within_distance_to_zero , set_pixels_within_distance

def get_branch_points(skeleton_mask, distance):
    image = skeleton_mask.copy()
    #3x3 kernel to collect the region around each pixel
    kernel  = np.ones((5,5) , dtype=int)
    #the junctions will have at least 6 pixels in the region
    #of a 3x3 square centred on the junction pixel.
    neigh_sum = convolve(image, kernel, mode='constant', cval=0.0)
    neigh_sum[neigh_sum < 14] = 0
    labeled_maxima = label(neigh_sum > 0)
     # Find the centers of the maxima
    centres = center_of_mass(image, labeled_maxima, range(1, labeled_maxima.max() + 1))
    centres = np.array(centres)
    junction_image = np.zeros_like(image)

    for centre in centres:
        centre_image = image.copy()
        # Create a grid of indices (y, x) for the image shape
        rows, cols = centre_image.shape
        y, x = np.indices((rows, cols))  # Grid of indices for rows and columns
        
        cy, cx = centre  # Extract the coordinate to search around
        # Calculate the distance to the given coordinate
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        
        # Set all pixels that are further than the distance to 0, and have a value of 1
        centre_image[(dist > distance) & (centre_image == 1)] = 0
        junction_image += centre_image 

    junction_image = label(junction_image > 0)

    #first remove the junctions
    image_minus_junctions = set_pixels_within_distance_to_zero(image, centres,distance)
    # Label each membrane
    labeled_membranes = label(image_minus_junctions, connectivity=2)

    seg_image = combine_labeled_images(junction_image, labeled_membranes)


    return centres , junction_image , labeled_membranes , seg_image


def main():
    file_path = './monolayer_analysis_test/ilastik_output/20241101_MDCKmonolayer_ZO1_mS_Magenta_frame1_Multicut Segmentation_edges.npy'
    image = np.load(file_path)
    centres , junction_image , labeled_membranes , seg_image= get_branch_points(image, 2)
    plt.imshow(seg_image)
    plt.scatter(centres[:, 1], centres[:, 0], c='k', s=10)
    #plt.scatter(junctions[:, 1], junctions[:, 0], c='r', s=10)
    plt.colorbar()
    plt.show()
    # Save the results)
    
    print("Intersections and labeled membranes have been saved.")

if __name__ == "__main__":
    main()