import numpy as np
import cv2

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

# Define the step size
step_size = 20

# Loop through the grid with a step of 200 pixels
for new_x in range(0, 1280, step_size):
    for new_y in range(0, 720, step_size):
        
        # Store distances
        distances = []

        # Radial scan of 500 pixels to find the 3 nearest pixels
        for idx, (x, y) in enumerate(pixel_positions):
            distance = np.sqrt((x - new_x) ** 2 + (y - new_y) ** 2)
            if distance <= 500:  # Only consider pixels within a radius of 500 pixels
                distances.append((distance, pixel_colors[idx], (x, y)))

        if len(distances) < 3:
            continue  # Skip if there are not enough neighbors

        # Sort by distance
        distances.sort()

        # Select the 3 nearest pixels
        nearest_pixels = distances[:3]

        # Calculate weights based on distance (inverse of distance)
        weights = []
        for dist, color, position in nearest_pixels:
            if dist == 0:  # Avoid division by zero
                weight = 1.0
            else:
                weight = 1 / dist
            weights.append(weight)

        # Normalize the weights so they sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Calculate the blended color
        blended_color = np.zeros(3)
        for i, (_, color, _) in enumerate(nearest_pixels):
            blended_color += normalized_weights[i] * np.array(color)

        # Set the new pixel to the blended color
        image[new_y, new_x] = blended_color.astype(np.uint8)

        # Draw lines to the 3 nearest pixels
        for _, _, (x, y) in nearest_pixels:
            cv2.line(image, (new_x, new_y), (x, y), (0, 255, 0), 1)  # Draw green lines

# Display the image using OpenCV
cv2.imshow('Random Pixels with Blended Colors and Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
