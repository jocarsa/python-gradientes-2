import os
import numpy as np
import cv2
import time

# Parameters
width, height = 1920, 1080
num_particles = 100
duration = 60  # seconds
fps = 60
total_frames = duration * fps

# Create the 'render' directory if it doesn't exist
output_folder = 'render'
os.makedirs(output_folder, exist_ok=True)

# Path to save the video
output_path = os.path.join(output_folder, 'particles_bounce.mp4')

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize particles with random positions, directions, and speeds
particles = np.zeros((num_particles, 5))  # columns: x, y, dx, dy, speed
particles[:, 0] = np.random.randint(0, width, size=num_particles)  # x positions
particles[:, 1] = np.random.randint(0, height, size=num_particles)  # y positions
angles = np.random.uniform(0, 2 * np.pi, size=num_particles)  # random angles
particles[:, 2] = np.cos(angles)  # dx (direction x)
particles[:, 3] = np.sin(angles)  # dy (direction y)
particles[:, 4] = np.random.uniform(1, 5, size=num_particles)  # speeds

# Start the timer
start_time = time.time()

# Create the video frames
for frame_idx in range(total_frames):
    # Create a black image
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Update particle positions
    particles[:, 0] += particles[:, 4] * particles[:, 2]  # x += speed * dx
    particles[:, 1] += particles[:, 4] * particles[:, 3]  # y += speed * dy

    # Bounce off the walls
    for i in range(num_particles):
        if particles[i, 0] <= 0 or particles[i, 0] >= width:
            particles[i, 2] *= -1  # Invert x direction
        if particles[i, 1] <= 0 or particles[i, 1] >= height:
            particles[i, 3] *= -1  # Invert y direction

    # Draw the particles
    for i in range(num_particles):
        x = int(particles[i, 0])
        y = int(particles[i, 1])
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)  # Draw a white circle for each particle

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
