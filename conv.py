import time
import cv2
import ctypes
import numpy as np
import numpy.ctypeslib as npct
from scipy.ndimage import convolve

FFI : bool = True

def convy(frames : list, k3d) -> bool:
    stacked = np.stack(frames, axis=0)  # shape (3, h, w, 3)
    out_channels = []
    for c in range(3):  # RGB
        out = convolve(stacked[:, :, :, c], k3d, mode='nearest')
        out_channels.append(out[1])  # middle frame result

    out_frame = np.stack(out_channels, axis=2)
    out_frame = np.clip(out_frame * 255, 0, 255).astype(np.uint8)
    writer.write(out_frame)

    # Slide window
    ret, next_frame = cap.read()
    if not ret:
        return False
    frames.pop(0)
    frames.append(next_frame.astype(np.float32) / 255.0)
    return True

def convffi(conv3d_func, frames : list, k3d_flat) -> bool:
    stacked = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.float32)
    out_frame = np.zeros_like(frames[1])
    height, width = stacked.shape[1:3]
    
    conv3d_func(
        npct.as_ctypes(stacked),
        k3d_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        npct.as_ctypes(out_frame),
        width,
        height,
        1
    )

    final_frame = out_frame.astype(np.uint8)
    writer.write(final_frame)

    ret, next_frame = cap.read()
    if not ret:
        return False
    
    frames.pop(0)
    frames.append(next_frame.astype(np.float32))
    return True

def main():
    global cap, writer

    lib = ctypes.CDLL("./build/libcconv3d.so")
    conv3d_func = lib['conv3d']
    conv3d_func.argtypes = [
        ctypes.c_voidp,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_voidp,
        ctypes.c_int,
        ctypes.c_int]
    conv3d_func.restype = None

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

    input_path = "./input_videos/sample.mp4"
    output_path = "./output_videos/output_py.mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at {input_path}. Please check the path.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frames = []
    for _ in range(3):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f.astype(np.float32))
        # frames.append(f.astype(np.float32) / 255.0)

    start_time = time.time()
    print(f'FFI : {FFI}')
    print(f"Processing video (W:{w}, H:{h}, FPS:{fps})...")

    kernel3d_flat = kernel3d.flatten()
    while True:
        if len(frames) < 3:
            break

        if FFI:
            if not convffi(conv3d_func, frames, kernel3d_flat):
                break
        else:
            if not convy(frames, kernel3d):
                break

    print("All frames processed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.6f}s")

    cap.release()
    writer.release()
    print(f"✅ Done! Saved to {output_path}")


if __name__ == "__main__":
    main()