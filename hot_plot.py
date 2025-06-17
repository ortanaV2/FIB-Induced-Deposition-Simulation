import os
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt

image_path = "result.png"

if not os.path.exists(image_path):
    print(f"{image_path} nicht gefunden.")
    exit()

last_modified = os.path.getmtime(image_path)

def safe_load_image(path, retries=5, delay=0.1):
    for _ in range(retries):
        try:
            with Image.open(path) as img:
                return np.array(img.convert("RGB"))
        except (OSError, UnidentifiedImageError):
            time.sleep(delay)
    raise RuntimeError(f"Konnte {path} nach {retries} Versuchen nicht laden (möglicherweise noch im Schreibvorgang).")

img_array = safe_load_image(image_path)

fig, ax = plt.subplots()
im = ax.imshow(img_array)
ax.axis('off')
plt.tight_layout()
plt.show(block=False)

print("Live-Updater gestartet. Änderungen an 'result.png' werden überwacht...")

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
                print("Bild aktualisiert.")
            except RuntimeError as e:
                print("Fehler beim Laden des Bildes:", e)
except KeyboardInterrupt:
    print("\nBeendet durch Benutzer.")
