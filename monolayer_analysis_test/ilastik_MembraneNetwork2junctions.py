import tifffile as tiff
import numpy as np
from skimage.morphology import skeletonize , label , local_maxima
from skimage import measure
from scipy.ndimage import distance_transform_edt , convolve , center_of_mass
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
def detect_intersections_and_label_membranes(image):
    # Load the binary image
    
    # Detect intersection points using morphological operations
    skeleton = skeletonize(image > 0)
    intersections = label(local_maxima(distance_transform_edt(skeleton)))
    
    # Label each membrane
    labeled_membranes = measure.label(image, connectivity=2)
    
    return intersections, labeled_membranes
def get_branch_points(skeleton_mask):
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
    junctions = peak_local_max(neigh_sum, min_distance=4)
    return centres ,junctions , neigh_sum

def main():
    file_path = './monolayer_analysis_test/ilastik_output/20241101_MDCKmonolayer_ZO1_mS_Magenta_frame1_Multicut Segmentation_edges.npy'
    image = np.load(file_path)
    centres , junctions , neigh_sum = get_branch_points(image)
    intersections, labeled_membranes = detect_intersections_and_label_membranes(image)
    print(junctions)
    plt.imshow(image)
    plt.scatter(centres[:, 1], centres[:, 0], c='k', s=10)
    plt.scatter(junctions[:, 1], junctions[:, 0], c='r', s=10)
    plt.colorbar()
    plt.show()
    # Save the results

    
    print("Intersections and labeled membranes have been saved.")

if __name__ == "__main__":
    main()