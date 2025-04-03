import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Correct the file path to handle the backslash issue
img_path = r"D:\cd\python\shree.png"  # Correct path

# Load the image
img = mpimg.imread(img_path)

# Check if the image loaded correctly
if img is None:
    print("Image not found or failed to load!")
else:
    print("Image loaded successfully!")

# Get the dimensions of the image
rows, cols, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)

# Generate 2D coordinates (x, y)
x, y = np.meshgrid(range(cols), range(rows))

# Flatten the coordinates
x_flat = x.flatten()
y_flat = y.flatten()

# Normalize the image values for color mapping
img_normalized = img / 255.0

# Flatten the color data (convert to RGB if grayscale)
if channels == 3:
    # If RGB image, flatten it directly (will give one color per pixel)
    color_flat = img_normalized.reshape(-1, 3)
else:
    # If grayscale, make it an RGB-like structure
    color_flat = np.repeat(img_normalized.flatten()[:, np.newaxis], 3, axis=1)

# Ensure that the number of color values matches the number of coordinates
assert len(x_flat) == len(color_flat), "Mismatch between number of coordinates and colors"

# Create a 2D scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x_flat, y_flat, c=color_flat, s=1)

# Hide the axes for better visualization
plt.axis('off')

# Show the plot
plt.show(block=True)
