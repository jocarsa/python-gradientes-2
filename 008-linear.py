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

# Loop through the grid with a step of 5 pixels
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

        # Calculate weights based on distance (linear weighting)
        total_distance = sum(dist[0] for dist in nearest_pixels)
        if total_distance == 0:
            continue  # Avoid division by zero

        weights = [(dist[0] / total_distance) for dist in nearest_pixels]

        # Calculate the blended color
        blended_color = np.zeros(3)
        for i, (_, color, _) in enumerate(nearest_pixels):
            blended_color += weights[i] * np.array(color)

        # Convert the blended color to a tuple of integers
        blended_color = tuple(map(int, blended_color))

        # Draw a rectangle with the blended color
        top_left = (new_x, new_y)
        bottom_right = (new_x + step_size, new_y + step_size)
        cv2.rectangle(image, top_left, bottom_right, blended_color, -1)  # -1 fills the rectangle

# Display the image using OpenCV
cv2.imshow('Random Pixels with Linearly Blended Colors in Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
