import os
import numpy as np
import cv2
import time
from scipy.spatial import KDTree

# Parameters
width, height = 1920, 1080
num_particles = 100
duration = 60  # seconds
fps = 60
total_frames = duration * fps
transition_duration = 5  # seconds
transition_frames = transition_duration * fps

# Create the 'render' directory if it doesn't exist
output_folder = 'render'
os.makedirs(output_folder, exist_ok=True)

# Path to save the video
output_path = os.path.join(output_folder, 'particles_bounce_gradient.mp4')

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize particles with random positions, directions, speeds, and colors
particles = np.zeros((num_particles, 8))  # columns: x, y, dx, dy, speed, r, g, b
particles[:, 0] = np.random.randint(0, width, size=num_particles)  # x positions
particles[:, 1] = np.random.randint(0, height, size=num_particles)  # y positions
angles = np.random.uniform(0, 2 * np.pi, size=num_particles)  # random angles
particles[:, 2] = np.cos(angles)  # dx (direction x)
particles[:, 3] = np.sin(angles)  # dy (direction y)
particles[:, 4] = np.random.uniform(1, 5, size=num_particles)  # speeds
particles[:, 5:8] = np.random.randint(0, 256, size=(num_particles, 3))  # initial colors

# Next color for transition
next_colors = np.random.randint(0, 256, size=(num_particles, 3))

# Start the timer
start_time = time.time()

# Create the video frames
for frame_idx in range(total_frames):
    # Create a black image
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the blend factor for the color transition
    blend_factor = (frame_idx % transition_frames) / transition_frames

    # Interpolate the particle colors
    current_colors = (1 - blend_factor) * particles[:, 5:8] + blend_factor * next_colors

    # If the transition is complete, assign the next colors
    if frame_idx % transition_frames == 0 and frame_idx != 0:
        particles[:, 5:8] = next_colors
        next_colors = np.random.randint(0, 256, size=(num_particles, 3))

    # Update particle positions
    particles[:, 0] += particles[:, 4] * particles[:, 2]  # x += speed * dx
    particles[:, 1] += particles[:, 4] * particles[:, 3]  # y += speed * dy

    # Bounce off the walls
    for i in range(num_particles):
        if particles[i, 0] <= 0 or particles[i, 0] >= width:
            particles[i, 2] *= -1  # Invert x direction
        if particles[i, 1] <= 0 or particles[i, 1] >= height:
            particles[i, 3] *= -1  # Invert y direction

    # Build KD-Tree for efficient nearest neighbor search
    tree = KDTree(particles[:, 0:2])

    # Apply gradient computation with KD-Tree
    step_size = 5
    radius = 500
    sigma = 200  # Standard deviation for Gaussian weighting

    # Loop through the grid with a step of 5 pixels
    for new_x in range(0, width, step_size):
        for new_y in range(0, height, step_size):

            # Query the KD-Tree for neighbors within the radius
            distances, indices = tree.query((new_x, new_y), k=10, distance_upper_bound=radius)

            # Filter out infinite distances (no neighbors found within radius)
            valid = distances < np.inf
            distances = distances[valid]
            indices = indices[valid]

            if len(distances) == 0:
                continue  # Skip if no neighbors within the radius

            # Calculate Gaussian weights
            weights = np.exp(-distances**2 / (2 * sigma**2))

            # Calculate the blended color
            blended_color = np.sum(weights[:, np.newaxis] * current_colors[indices], axis=0)
            blended_color /= np.sum(weights)

            # Convert the blended color to a tuple of integers
            blended_color = tuple(map(int, blended_color))

            # Draw a rectangle with the blended color
            top_left = (new_x, new_y)
            bottom_right = (new_x + step_size, new_y + step_size)
            cv2.rectangle(frame, top_left, bottom_right, blended_color, -1)  # -1 fills the rectangle

    # Write the frame to the video file
    out.write(frame)

    # Show statistics every 60 frames (1 second)
    if frame_idx % fps == 0 and frame_idx != 0:
        elapsed_time = time.time() - start_time
        percentage_completed = (frame_idx / total_frames) * 100
        time_remaining = (elapsed_time / frame_idx) * (total_frames - frame_idx)
        estimated_finish_time = time.strftime('%H:%M:%S', time.localtime(time.time() + time_remaining))

        print(f"Time Passed: {elapsed_time:.2f} seconds")
        print(f"Time Remaining: {time_remaining:.2f} seconds")
        print(f"Estimated Time of Finish: {estimated_finish_time}")
        print(f"Percentage of Completion: {percentage_completed:.2f}%")

# Release the video writer
out.release()

print("Video creation complete.")
