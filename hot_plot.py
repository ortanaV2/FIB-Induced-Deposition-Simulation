import os, time
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

image_path = "result.png"
if not os.path.exists(image_path):
    print(f"{image_path} not found.")
    exit()

def safe_load_image(path, retries=5, delay=0.1):
    for _ in range(retries):
        try:
            with Image.open(path) as img:
                return np.array(img.convert("RGB"))
        except (OSError, UnidentifiedImageError):
            time.sleep(delay)
    raise RuntimeError(f"Failed to load {path} after {retries} attempts.")

last_modified = os.path.getmtime(image_path)
img_array = safe_load_image(image_path)

fig, ax = plt.subplots()
im = ax.imshow(img_array)
ax.axis('off')
plt.tight_layout()
plt.show(block=False)

print("Live-updater started. Monitoring changes to 'result.png'...")

try:
    while True:
        plt.pause(0.5)
        current_modified = os.path.getmtime(image_path)
        if current_modified != last_modified:
            last_modified = current_modified
            try:
                img_array = safe_load_image(image_path)
                im.set_data(img_array)
                fig.canvas.draw_idle()
                print("Image updated.")
            except RuntimeError as e:
                print("Error loading image:", e)
except KeyboardInterrupt:
    print("\nTerminated by user.")
