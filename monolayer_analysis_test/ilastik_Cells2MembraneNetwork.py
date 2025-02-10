import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def make_2d_edges(segmentation):
    """ Make 2d edges from 2d segmentation
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    return ((gx ** 2 + gy ** 2) > 0)


if __name__ == "__main__":
    # Example usage
    filepath = './monolayer_analysis_test/ilastik_output/20241101_MDCKmonolayer_ZO1_mS_Magenta_frame1_Multicut Segmentation.npy'
    segmentation = np.load(filepath)
    segmentation = segmentation[...,0]
    edges = make_2d_edges(segmentation)
    print(edges)
    np.save('./monolayer_analysis_test/ilastik_output/20241101_MDCKmonolayer_ZO1_mS_Magenta_frame1_Multicut Segmentation_edges.npy', edges.astype(np.uint8))