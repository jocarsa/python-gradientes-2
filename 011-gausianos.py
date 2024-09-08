import numpy as np
import cv2

# Create a blank image of 1280x720 with 3 color channels (RGB)
image = np.zeros((720, 1280, 3), dtype=np.uint8)

# List to store the positions and colors of the drawn pixels
pixel_positions = []
pixel_colors = []

# Draw 100 random pixels with random colors
for _ in range(50):
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

# Define the step size and the Gaussian sigma
step_size = 5
radius = 500
sigma = 200  # Standard deviation for Gaussian weighting

# Loop through the grid with a step of 5 pixels
for new_x in range(0, 1280, step_size):
    for new_y in range(0, 720, step_size):
        
        # Store distances and corresponding colors
        weights = []
        blended_color = np.zeros(3)

        # Radial scan of the nearest pixels
        for idx, (x, y) in enumerate(pixel_positions):
            distance = np.sqrt((x - new_x) ** 2 + (y - new_y) ** 2)
            if distance <= radius:  # Only consider pixels within the defined radius
                weight = np.exp(-distance**2 / (2 * sigma**2))
                weights.append(weight)
                blended_color += weight * np.array(pixel_colors[idx])

        if len(weights) == 0:
            continue  # Skip if no neighbors within the radius

        # Normalize the blended color by the sum of the weights
        total_weight = sum(weights)
        if total_weight > 0:
            blended_color /= total_weight

        # Convert the blended color to a tuple of integers
        blended_color = tuple(map(int, blended_color))

        # Draw a rectangle with the blended color
        top_left = (new_x, new_y)
        bottom_right = (new_x + step_size, new_y + step_size)
        cv2.rectangle(image, top_left, bottom_right, blended_color, -1)  # -1 fills the rectangle

# Display the image using OpenCV
cv2.imshow('Random Pixels with Gaussian Blended Colors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
