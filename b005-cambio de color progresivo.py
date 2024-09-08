import numpy as np
import cv2
import random
import os
import time
from datetime import timedelta, datetime

# Define the width and height of the image
width = 1920
height = 1080

num_points = 6

fps = 60
minutos = 1
segundosporminuto = 60
num_frames = fps * segundosporminuto * minutos

# Create the render folder if it does not exist
render_folder = "render"
os.makedirs(render_folder, exist_ok=True)

# Generate random points, their initial colors, directions, and speeds
points = np.array([(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_points)], dtype=np.float32)
current_colors = np.array([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)], dtype=np.float32)
target_colors = np.array([(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)], dtype=np.float32)
directions = np.array([(random.choice([-1, 1]), random.choice([-1, 1])) for _ in range(num_points)], dtype=np.float32)
speeds = np.array([random.uniform(1, 5) for _ in range(num_points)], dtype=np.float32)

# Downsample factor
downsample_factor = 4
downsampled_width = width // downsample_factor
downsampled_height = height // downsample_factor

def update_points(points, directions, speeds):
    points += directions * speeds[:, None]
    
    # Bounce off the walls
    directions[points[:, 0] < 0, 0] *= -1
    directions[points[:, 0] >= width, 0] *= -1
    directions[points[:, 1] < 0, 1] *= -1
    directions[points[:, 1] >= height, 1] *= -1

    return points

def update_colors(current_colors, target_colors, progress):
    # Interpolate between current colors and target colors
    interpolated_colors = (1 - progress) * current_colors + progress * target_colors
    return interpolated_colors

def generate_frame(frame_num):
    # Update points positions
    update_points(points, directions, speeds)
    
    # Calculate progress within the current 5-second interval (0 to 1)
    interval_progress = (frame_num % (5 * fps)) / (5 * fps)
    
    # If the interval is complete, move to the next target color set
    if interval_progress == 0 and frame_num > 0:
        current_colors[:] = target_colors  # Set current to target
        target_colors[:] = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_points)]
    
    # Update colors with continuous transition
    colors = update_colors(current_colors, target_colors, interval_progress)
    
    # Create mesh grid for all pixel coordinates (downsampled)
    xv, yv = np.meshgrid(np.arange(downsampled_width), np.arange(downsampled_height), indexing='xy')

    # Scale mesh grid to original size
    xv = xv * downsample_factor
    yv = yv * downsample_factor

    # Calculate distances from each pixel to each point
    distances = np.sqrt((xv[..., None] - points[:, 0]) ** 2 + (yv[..., None] - points[:, 1]) ** 2)

    # Calculate weights using a sharper inverse distance function
    weights = 1 / (distances ** 3 + 1e-10)  # Cubic inverse distance for sharper transitions

    # Normalize weights
    weights /= weights.sum(axis=2)[..., None]

    # Calculate interpolated colors
    interpolated_colors = np.dot(weights, colors)

    # Convert to integers and upsample to the original size
    interpolated_colors = cv2.resize(interpolated_colors.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

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
    frame = generate_frame(frame_num)
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
