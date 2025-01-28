import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot , utils , io
from scipy.ndimage import label, sum as ndi_sum
import skimage as ski
from skimage.measure import perimeter , approximate_polygon
import pandas as pd

masks_dict = np.load('monolayer_ZO1_seg.npy', allow_pickle=True).item()
masks = masks_dict['masks']
dims = np.shape(masks)
outlines = utils.outlines_list(masks)
outlines_no_boundaries = []
masks_no_boundaries = masks.copy()
cell = 1

for outline in outlines:
    if np.all((outline >= (4)) & (outline <= (dims[1]-4))):
        outlines_no_boundaries.append(outline)
        outline_polygon = approximate_polygon(outline,5)
        #plt.plot(outline[:,0] , outline[:,1])
        #plt.plot(outline_polygon[:,0] , outline_polygon[:,1])
    else:
        masks_no_boundaries[masks_no_boundaries == cell] = 0
    cell +=1
            #print(outline)
masks_dict['masks_no_boundaries'] = masks_no_boundaries
masks_dict['outlines_no_boundaries'] = outlines_no_boundaries

binary_masks_no_boundaries = masks_no_boundaries.copy()
binary_masks_no_boundaries[binary_masks_no_boundaries >= 1] = 1

regions = ski.measure.regionprops(masks_no_boundaries)
cell_prop_df = pd.DataFrame()
cell_no = np.linspace(1 , len(outlines_no_boundaries) , len(outlines_no_boundaries))
areas = []
perimeters = []
for region in regions:
    area = region.area
    perimeter = region.perimeter
    areas.append(area)
    perimeters.append(perimeter)

cell_prop_df['cell_no'] = cell_no
cell_prop_df['areas'] = areas
cell_prop_df['perimeters'] = perimeters

plt.hist(cell_prop_df['areas'], bins = 25)
plt.show()
plt.hist(cell_prop_df['areas']/cell_prop_df['perimeters'], bins = 25)
plt.show()
