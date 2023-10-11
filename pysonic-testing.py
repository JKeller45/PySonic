import tomllib as tl
from Audio_Processing import render
from Classes import Progress_Spoof, Main_Spoof
import cv2
import numpy as np

if __name__ == "__main__":
    with open("testing/testing-config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]

    render(config, Progress_Spoof(), Main_Spoof())

    print("Done rendering, now validating...")
    video = cv2.VideoCapture("testing/Avicii - Levels.webm.wav_Audio.mp4")
    comparison = cv2.VideoCapture("testing/pysonic-test-comparison.mp4")
    ret1, frame1 = video.read()
    ret2, frame2 = comparison.read()
    if not ret1 or not ret2:
        raise Exception("Video files not found!")
    while ret1 and ret2:
        mse = np.sum(np.square(np.subtract(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).flatten(), cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).flatten()))) / frame1.size
        if mse > .03:
            print(mse)
        assert mse < 0.05, "Video files do not match!"
    video.release()
    comparison.release()
    print("Validation complete!")