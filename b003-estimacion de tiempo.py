import numpy as np
import cv2
import random
import os
import time
from datetime import timedelta, datetime

# Define the width and height of the image
width = 3840
height = 2160

num_points = 6

fps = 30
minutos = 60
segundosporminuto = 60
num_frames = fps * segundosporminuto * minutos

# Create the render folder if it does not exist
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Generate random points, their colors, directions, and speeds
points = np.array([(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_points)], dtype=float)
colors = np.array([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)])
directions = np.array([(random.choice([-1, 1]), random.choice([-1, 1])) for _ in range(num_points)])
speeds = np.array([random.uniform(1, 5) for _ in range(num_points)])

def update_points(points, directions, speeds):
    for i in range(num_points):
        # Update the position
        points[i] += directions[i] * speeds[i]
        
        # Bounce off the walls
        if points[i, 0] < 0 or points[i, 0] >= width:
            directions[i, 0] *= -1
        if points[i, 1] < 0 or points[i, 1] >= height:
            directions[i, 1] *= -1

    return points

def generate_frame():
    # Update points positions
    update_points(points, directions, speeds)
    
    # Create mesh grid for all pixel coordinates
    xv, yv = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    # Calculate distances from each pixel to each point
    distances = np.sqrt((xv[..., None] - points[:, 0]) ** 2 + (yv[..., None] - points[:, 1]) ** 2)

    # Calculate weights using a sharper inverse distance function
    weights = 1 / (distances ** 3 + 1e-10)  # Cubic inverse distance for sharper transitions

    # Normalize weights
    weights /= weights.sum(axis=2)[..., None]

    # Calculate interpolated colors
    interpolated_colors = np.dot(weights, colors)

    # Convert to integers
    interpolated_colors = interpolated_colors.astype(np.uint8)

    return interpolated_colors

# Get the current epoch time for unique filenames
epoch_time = int(time.time())

# Define the codec and create VideoWriter object
mp4_path = os.path.join(render_folder, f"gradient_points_animation_{epoch_time}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

# Record the start time
start_time = time.time()

# Generate each frame and write it to the video file
for frame_num in range(num_frames):
    frame = generate_frame()
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

    # Every second (30 frames), print progress
    if frame_num % fps == 0:
        elapsed_time = time.time() - start_time
        frames_done = frame_num + 1
        percentage_done = frames_done / num_frames * 100
        
        # Estimate time remaining
        frames_remaining = num_frames - frames_done
        estimated_time_remaining = frames_remaining * (elapsed_time / frames_done)
        estimated_finish_time = datetime.now() + timedelta(seconds=estimated_time_remaining)
        
        print(f"Progress: {percentage_done:.2f}% done.")
        print(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        print(f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}")
        print(f"Estimated completion time: {estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 40)

# Release the video writer
out.release()

print(f"Animation saved successfully as {mp4_path}.")
