import numpy as np
import cv2

# Create a blank image of 1280x720 with 3 color channels (RGB)
image = np.zeros((720, 1280, 3), dtype=np.uint8)

# List to store the positions of the drawn pixels
pixel_positions = []

# Draw 100 random pixels with random colors
for _ in range(100):
    # Random position
    x = np.random.randint(0, 1280)
    y = np.random.randint(0, 720)
    
    # Random color
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    
    # Draw the pixel
    image[y, x] = color
    
    # Store the position
    pixel_positions.append((x, y))

# Draw an additional random pixel
new_x = np.random.randint(0, 1280)
new_y = np.random.randint(0, 720)
new_color = (255, 255, 255)  # White color for the new pixel

# Draw the new pixel
image[new_y, new_x] = new_color

# Store distances
distances = []

# Radial scan of 100 pixels to find the 3 nearest pixels
for x, y in pixel_positions:
    distance = np.sqrt((x - new_x) ** 2 + (y - new_y) ** 2)
    if distance <= 100:  # Only consider pixels within a radius of 100 pixels
        distances.append((distance, x, y))

# Sort by distance
distances.sort()

# Select the 3 nearest pixels
nearest_pixels = distances[:3]

# Highlight the 3 nearest pixels
for _, x, y in nearest_pixels:
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw red circles around the nearest pixels

# Display the image using OpenCV
cv2.imshow('Random Pixels with Nearest Neighbors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
