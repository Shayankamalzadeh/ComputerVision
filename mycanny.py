import numpy as np
import cv2

def canny_edge_detection(image, sigma, kernel_size, low_threshold, high_threshold):
    # Apply Gaussian smoothing to the image
   
    smoothed = cv2.GaussianBlur(image, kernel_size, sigma,cv2.BORDER_REFLECT)
     # Calculate the gradient magnitude and direction
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    # Convert the gradient direction to degrees
    grad_dir_deg = np.rad2deg(grad_dir)

    # Ensure that the gradient direction is within [0, 180)
    grad_dir_deg[grad_dir_deg < 0] += 180

    # Initialize the edge map
   
 
    edge_map = non_maximum_suppression(grad_mag, grad_dir)
    
    # Apply hysteresis thresholding to the edge map
    thresholded = hysteresis_thresholding(edge_map,low_threshold, high_threshold)

    return thresholded

def non_maximum_suppression(grad_mag, grad_dir):
    # Round the gradient direction to the nearest of 0, 45, 90, or 135 degrees
    grad_dir_deg = np.rad2deg(grad_dir) % 180
    grad_dir_rounded = np.zeros_like(grad_dir_deg, dtype=np.int32)
    grad_dir_rounded[(grad_dir_deg < 22.5) | (grad_dir_deg >= 157.5)] = 0
    grad_dir_rounded[(22.5 <= grad_dir_deg) & (grad_dir_deg < 67.5)] = 45
    grad_dir_rounded[(67.5 <= grad_dir_deg) & (grad_dir_deg < 112.5)] = 90
    grad_dir_rounded[(112.5 <= grad_dir_deg) & (grad_dir_deg < 157.5)] = 135

    # Apply non-maximum suppression to the gradient magnitude
    edge_map = np.zeros_like(grad_mag,dtype="uint8")
    for i in range(1, grad_mag.shape[0] - 1):
        for j in range(1, grad_mag.shape[1] - 1):
            angle = grad_dir_rounded[i, j]
            if angle == 0:
                if np.all([grad_mag[i, j] > grad_mag[i, j-1], grad_mag[i, j] > grad_mag[i, j+1]]):
                    edge_map[i, j] = 255
            elif angle == 45:
                if np.all([grad_mag[i, j] > grad_mag[i-1, j-1], grad_mag[i, j] > grad_mag[i+1, j+1]]):
                    edge_map[i, j] = 255
            elif angle == 90:
                if np.all([grad_mag[i, j] > grad_mag[i-1, j], grad_mag[i, j] > grad_mag[i+1, j]]):
                    edge_map[i, j] = 255
            elif angle == 135:
                if np.all([grad_mag[i, j] > grad_mag[i-1, j+1], grad_mag[i, j] > grad_mag[i+1, j-1]]):
                    edge_map[i, j] = 255

    return edge_map

def hysteresis_thresholding(edge_map, low_thresh, high_thresh):
    # Set pixel values to either strong, weak, or non-edges based on thresholds
    strong_edge_value = 255
    weak_edge_value = 50
    non_edge_value = 0
    
    strong_edges = (edge_map >= high_thresh)
    strong_edges_indices = np.argwhere(strong_edges)
    weak_edges = ((edge_map >= low_thresh) & (edge_map < high_thresh))
    weak_edges_indices = np.argwhere(weak_edges)
    non_edges = (edge_map < low_thresh)
    non_edges_indices = np.argwhere(non_edges)
    
    edge_map_hyst = np.zeros_like(edge_map)
    
    # Set strong edges and connect weak edges to strong edges if they are adjacent
    for i, j in strong_edges_indices:
        edge_map_hyst[i, j] = strong_edge_value
        
        # Check 8-connected neighbors for weak edges
        if (i-1 >= 0 and j-1 >= 0 and weak_edges[i-1, j-1]):
            edge_map_hyst[i-1, j-1] = weak_edge_value
        if (i-1 >= 0 and weak_edges[i-1, j]):
            edge_map_hyst[i-1, j] = weak_edge_value
        if (i-1 >= 0 and j+1 < edge_map.shape[1] and weak_edges[i-1, j+1]):
            edge_map_hyst[i-1, j+1] = weak_edge_value
        if (j-1 >= 0 and weak_edges[i, j-1]):
            edge_map_hyst[i, j-1] = weak_edge_value
        if (j+1 < edge_map.shape[1] and weak_edges[i, j+1]):
            edge_map_hyst[i, j+1] = weak_edge_value
        if (i+1 < edge_map.shape[0] and j-1 >= 0 and weak_edges[i+1, j-1]):
            edge_map_hyst[i+1, j-1] = weak_edge_value
        if (i+1 < edge_map.shape[0] and weak_edges[i+1, j]):
            edge_map_hyst[i+1, j] = weak_edge_value
        if (i+1 < edge_map.shape[0] and j+1 < edge_map.shape[1] and weak_edges[i+1, j+1]):
            edge_map_hyst[i+1, j+1] = weak_edge_value
    
    # Set remaining weak edges to non-edges
    for i, j in weak_edges_indices:
        if edge_map_hyst[i, j] != weak_edge_value:
            edge_map_hyst[i, j] = non_edge_value
    
    # Set non-edges to 0
    for i, j in non_edges_indices:
        edge_map_hyst[i, j] = non_edge_value
    
    return edge_map_hyst


def canny(img):
      # Define the Gaussian filter parameters
    kernel_size = (5, 5)
    sigma = 1.4

# Apply the Gaussian filter
    smooth_img = cv2.GaussianBlur(img, kernel_size, sigma)

# Calculate the gradient magnitude and direction
    grad_x = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

# Convert the gradient direction to degrees
    grad_dir_deg = np.rad2deg(grad_dir)

# Ensure that the gradient direction is within [0, 180)
    grad_dir_deg[grad_dir_deg < 0] += 180

# Initialize the edge map
    edge_map = np.zeros_like(grad_mag, dtype=np.uint8)

# Apply non-maximum suppression to the gradient magnitude
    for i in range(1, grad_mag.shape[0] - 1):
        for j in range(1, grad_mag.shape[1] - 1):
            angle = grad_dir_deg[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                if (grad_mag[i, j] > grad_mag[i, j-1]) and (grad_mag[i, j] > grad_mag[i, j+1]):
                    edge_map[i, j] = 255
            elif (22.5 <= angle < 67.5):
                if (grad_mag[i, j] > grad_mag[i-1, j-1]) and (grad_mag[i, j] > grad_mag[i+1, j+1]):
                    edge_map[i, j] = 255
            elif (67.5 <= angle < 112.5):
             if (grad_mag[i, j] > grad_mag[i-1, j]) and (grad_mag[i, j] > grad_mag[i+1, j]):
                edge_map[i, j] = 255
            elif (112.5 <= angle < 157.5):
                if (grad_mag[i, j] > grad_mag[i-1, j+1]) and (grad_mag[i, j] > grad_mag[i+1, j-1]):
                    edge_map[i, j] = 255

# Save the output image
   
    return edge_map
    


