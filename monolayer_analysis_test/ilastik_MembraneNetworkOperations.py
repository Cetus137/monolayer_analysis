import numpy as np

def sum_surrounding_points(image, coordinate, radius=1):
    """
    Sum the values of the points surrounding a given coordinate in a 2D grid.
    
    Parameters:
    - image: 2D numpy array representing the grid or image.
    - coordinate: tuple (y, x), the coordinate to sum the surrounding points around.
    - radius: the size of the neighborhood to sum (default is 1 for a 3x3 neighborhood).
    
    Returns:
    - sum_value: The sum of the values in the surrounding neighborhood.
    """
    y, x = coordinate
    rows, cols = image.shape
    
    # Define the neighborhood boundaries
    y_min = max(0, y - radius)
    y_max = min(rows, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(cols, x + radius + 1)
    
    # Extract the neighborhood and sum the values
    neighborhood = image[y_min:y_max, x_min:x_max]
    sum_value = np.sum(neighborhood)
    
    return sum_value
def combine_labeled_images(img1, img2):
    # Make a copy of img1 to keep the first image unchanged
    combined_img = np.copy(img1)
    
    # Add an offset to the labels of img2 to ensure they don't overlap with img1
    max_label_img1 = np.max(img1)
    
    # Set the labels in img2 to start after the max label of img1
    combined_img[img2 > 0] = img2[img2 > 0] + max_label_img1
    
    return combined_img

def set_pixels_within_distance_to_zero(image, coordinates, distance):
    # Create a grid of coordinates that matches the shape of the image
    rows, cols = image.shape
    y, x = np.indices((rows, cols))  # Create index grid for rows and columns
    
    # Loop over each coordinate and set nearby pixels to 0
    for (cy, cx) in coordinates:
        # Calculate the distance from this point to all pixels in the image
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        
        # Set all pixels within the given distance to 0
        image[dist <= distance] = 0

    return image

def set_pixels_within_distance(image, coordinate, distance):
    # Create a grid of indices (y, x) for the image shape
    rows, cols = image.shape
    y, x = np.indices((rows, cols))  # Grid of indices for rows and columns
    
    cy, cx = coordinate  # Extract the coordinate to search around
    # Calculate the distance to the given coordinate
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    # Set all pixels that are further than the distance to 0, and have a value of 1
    image[(dist > distance) & (image == 1)] = 0
    
    return image