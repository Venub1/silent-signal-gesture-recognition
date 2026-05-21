import cv2
import os
import numpy as np

INPUT_DIR = "dataset/raw_videos/proceed"

augmentations = [
    "blur",
    "dark",
    "bw",
    "flip",
    "noise",
    "zoom",
    "rot10",
    "lowlight",
    "blur_bw",
    "blur_dark",
    "bright",
    "bright_noise",
    "flip_blur",
    "flip_dark",
    "flip_noise",
    "noise_dark",
    "rot10_blur",
    "rot10_dark",
    "zoom_flip"
]

def add_noise(frame):
    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
    return cv2.add(frame, noise)

def darken(frame):
    return cv2.convertScaleAbs(frame, alpha=0.5, beta=0)

def brighten(frame):
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

def blur(frame):
    return cv2.GaussianBlur(frame, (9, 9), 0)

def grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def flip(frame):
    return cv2.flip(frame, 1)

def rotate(frame):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1)
    return cv2.warpAffine(frame, M, (w, h))

def zoom(frame):
    h, w = frame.shape[:2]
    crop = frame[40:h-40, 40:w-40]
    return cv2.resize(crop, (w, h))

def lowlight(frame):
    return cv2.convertScaleAbs(frame, alpha=0.3, beta=-20)

def apply_aug(frame, aug):
    if aug == "blur":
        return blur(frame)
    elif aug == "dark":
        return darken(frame)
    elif aug == "bw":
        return grayscale(frame)
    elif aug == "flip":
        return flip(frame)
    elif aug == "noise":
        return add_noise(frame)
    elif aug == "zoom":
        return zoom(frame)
    elif aug == "rot10":
        return rotate(frame)
    elif aug == "lowlight":
        return lowlight(frame)
    elif aug == "blur_bw":
        return grayscale(blur(frame))
    elif aug == "blur_dark":
        return darken(blur(frame))
    elif aug == "bright":
        return brighten(frame)
    elif aug == "bright_noise":
        return add_noise(brighten(frame))
    elif aug == "flip_blur":
        return blur(flip(frame))
    elif aug == "flip_dark":
        return darken(flip(frame))
    elif aug == "flip_noise":
        return add_noise(flip(frame))
    elif aug == "noise_dark":
        return darken(add_noise(frame))
    elif aug == "rot10_blur":
        return blur(rotate(frame))
    elif aug == "rot10_dark":
        return darken(rotate(frame))
    elif aug == "zoom_flip":
        return flip(zoom(frame))
    return frame

videos = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mov") and "_" in f and "blur" not in f]

for video in videos:
    video_path = os.path.join(INPUT_DIR, video)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    base_name = video.replace(".mov", "")

    for aug in augmentations:
        output_name = f"{base_name}_{aug}.mov"
        output_path = os.path.join(INPUT_DIR, output_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            aug_frame = apply_aug(frame, aug)
            out.write(aug_frame)

        out.release()

    cap.release()

print("Proceed augmentation complete.")
