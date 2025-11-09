import time
import cv2
import numpy as np
from scipy.ndimage import convolve

# 3D edge detector kernel
kernel3d = np.array([
    [[-1, -2, -1],
     [-2, -4, -2],
     [-1, -2, -1]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]]
], dtype=np.float32)

# Open input and output
input_path = "sample.mp4"
output_path = "output.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Read first 3 frames
frames = []
for _ in range(3):
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f.astype(np.float32) / 255.0)

start_time = time.time()

while True:
    if len(frames) < 3:
        break

    # Apply 3D convolution on 3 stacked frames
    stacked = np.stack(frames, axis=0)  # shape (3, h, w, 3)
    out_channels = []
    for c in range(3):  # RGB
        out = convolve(stacked[:, :, :, c], kernel3d, mode='nearest')
        out_channels.append(out[1])  # middle frame result

    out_frame = np.stack(out_channels, axis=2)
    out_frame = np.clip(out_frame * 255, 0, 255).astype(np.uint8)
    writer.write(out_frame)

    # Slide window
    ret, next_frame = cap.read()
    if not ret:
        break
    frames.pop(0)
    frames.append(next_frame.astype(np.float32) / 255.0)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.6f}s")

cap.release()
writer.release()
print("✅ Done! Saved to", output_path)
# python3.13.exe conv.py