import cv2
import os
import numpy as np

INPUT_DIR = "dataset/raw_videos/safe"

augmentations = [
    "blur", "dark", "bw", "flip", "noise", "zoom", "rot10", "lowlight",
    "blur_bw", "blur_dark", "bright", "bright_noise", "flip_blur",
    "flip_dark", "flip_noise", "noise_dark", "rot10_blur",
    "rot10_dark", "zoom_flip"
]

def add_noise(frame):
    noise = np.random.normal(0, 45, frame.shape).astype(np.int16)
    noisy = frame.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def darken(frame):
    return cv2.convertScaleAbs(frame, alpha=0.35, beta=-15)

def brighten(frame):
    return cv2.convertScaleAbs(frame, alpha=1.45, beta=45)

def blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def flip(frame):
    return cv2.flip(frame, 1)

def rotate(frame):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)
    return cv2.warpAffine(frame, M, (w, h))

def zoom(frame):
    h, w = frame.shape[:2]
    crop = frame[70:h-70, 70:w-70]
    return cv2.resize(crop, (w, h))

def lowlight(frame):
    return cv2.convertScaleAbs(frame, alpha=0.22, beta=-30)

def apply_aug(frame, aug):
    if aug == "blur": return blur(frame)
    if aug == "dark": return darken(frame)
    if aug == "bw": return grayscale(frame)
    if aug == "flip": return flip(frame)
    if aug == "noise": return add_noise(frame)
    if aug == "zoom": return zoom(frame)
    if aug == "rot10": return rotate(frame)
    if aug == "lowlight": return lowlight(frame)
    if aug == "blur_bw": return grayscale(blur(frame))
    if aug == "blur_dark": return darken(blur(frame))
    if aug == "bright": return brighten(frame)
    if aug == "bright_noise": return add_noise(brighten(frame))
    if aug == "flip_blur": return blur(flip(frame))
    if aug == "flip_dark": return darken(flip(frame))
    if aug == "flip_noise": return add_noise(flip(frame))
    if aug == "noise_dark": return darken(add_noise(frame))
    if aug == "rot10_blur": return blur(rotate(frame))
    if aug == "rot10_dark": return darken(rotate(frame))
    if aug == "zoom_flip": return flip(zoom(frame))
    return frame

videos = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".mov") and not any(x in f for x in augmentations)
]

for video in videos:
    video_path = os.path.join(INPUT_DIR, video)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    base_name = video.replace(".mov", "")

    for aug in augmentations:
        output_path = os.path.join(INPUT_DIR, f"{base_name}_{aug}.mov")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(apply_aug(frame, aug))

        out.release()

    cap.release()

print("Safe strong augmentation complete.")
