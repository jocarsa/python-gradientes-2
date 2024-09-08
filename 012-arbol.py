import numpy as np
import cv2
from scipy.spatial import KDTree

# Create a blank image of 1280x720 with 3 color channels (RGB)
image = np.zeros((720, 1280, 3), dtype=np.uint8)

# List to store the positions and colors of the drawn pixels
pixel_positions = []
pixel_colors = []

# Draw 100 random pixels with random colors
for _ in range(100):
    # Random position
    x = np.random.randint(0, 1280)
    y = np.random.randint(0, 720)
    
    # Random color
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    
    # Draw the pixel
    image[y, x] = color
    
    # Store the position and color
    pixel_positions.append((x, y))
    pixel_colors.append(color)

# Convert lists to arrays for faster processing
pixel_positions = np.array(pixel_positions)
pixel_colors = np.array(pixel_colors)

# Build KD-Tree for efficient nearest neighbor search
tree = KDTree(pixel_positions)

# Define the step size and the Gaussian sigma
step_size = 5
sigma = 200  # Standard deviation for Gaussian weighting

# Loop through the grid with a step of 5 pixels
for new_x in range(0, 1280, step_size):
    for new_y in range(0, 720, step_size):
        
        # Query the KD-Tree for neighbors within the radius
        distances, indices = tree.query((new_x, new_y), k=10, distance_upper_bound=500)
        
        # Filter out infinite distances (no neighbors found within radius)
        valid = distances < np.inf
        distances = distances[valid]
        indices = indices[valid]

        if len(distances) == 0:
            continue  # Skip if no neighbors within the radius

        # Calculate Gaussian weights
        weights = np.exp(-distances**2 / (2 * sigma**2))

        # Calculate the blended color
        blended_color = np.sum(weights[:, np.newaxis] * pixel_colors[indices], axis=0)
        blended_color /= np.sum(weights)

        # Convert the blended color to a tuple of integers
        blended_color = tuple(map(int, blended_color))

        # Draw a rectangle with the blended color
        top_left = (new_x, new_y)
        bottom_right = (new_x + step_size, new_y + step_size)
        cv2.rectangle(image, top_left, bottom_right, blended_color, -1)  # -1 fills the rectangle

# Display the image using OpenCV
cv2.imshow('Random Pixels with Gaussian Blended Colors (KD-Tree)', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
