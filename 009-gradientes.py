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
step_size = 5
radius = 500

# Loop through the grid with a step of 5 pixels
for new_x in range(0, 1280, step_size):
    for new_y in range(0, 720, step_size):
        
        # Store distances
        distances = []

        # Radial scan of the nearest pixels
        for idx, (x, y) in enumerate(pixel_positions):
            distance = np.sqrt((x - new_x) ** 2 + (y - new_y) ** 2)
            if distance <= radius:  # Only consider pixels within the defined radius
                distances.append((distance, pixel_colors[idx]))

        if len(distances) == 0:
            continue  # Skip if no neighbors within the radius

        # Sort by distance
        distances.sort()

        # Select the nearest neighbors (e.g., 10)
        nearest_pixels = distances[:10]

        # Calculate weights based on inverse distance
        weights = []
        for dist, color in nearest_pixels:
            if dist == 0:  # Handle case where distance is zero
                weight = 1.0
            else:
                weight = 1 / (dist ** 2)  # Inverse square weighting for smoother gradient
            weights.append(weight)

        # Normalize the weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Calculate the blended color
        blended_color = np.zeros(3)
        for i, (_, color) in enumerate(nearest_pixels):
            blended_color += normalized_weights[i] * np.array(color)

        # Convert the blended color to a tuple of integers
        blended_color = tuple(map(int, blended_color))

        # Draw a rectangle with the blended color
        top_left = (new_x, new_y)
        bottom_right = (new_x + step_size, new_y + step_size)
        cv2.rectangle(image, top_left, bottom_right, blended_color, -1)  # -1 fills the rectangle

# Display the image using OpenCV
cv2.imshow('Random Pixels with Smoothly Blended Colors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
