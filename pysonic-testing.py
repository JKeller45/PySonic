import tomllib as tl
from Audio_Processing import render
from Classes import Progress_Spoof, Main_Spoof
import cv2
import numpy as np
import os

def compare_videos(test, comparison):
    ret1, frame1 = test.read()
    ret2, frame2 = comparison.read()
    if not ret1 or not ret2:
        raise Exception("Video files not found!")
    while ret1 and ret2:
        mse = np.sum(np.square(np.subtract(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).flatten(), cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).flatten()))) / frame1.size
        if mse > 10:
            print(mse)
        assert mse < 32, "Video files do not match!"
        ret1, frame1 = test.read()
        ret2, frame2 = comparison.read()
        if not ret1 and ret2 or ret1 and not ret2:
            raise Exception("Video files not the same length!")
    test.release()
    comparison.release()

def test_config(test_number, output_path):
    print(f"Beginning test {test_number} validation...")
    with open(f"testing/testing-config-{test_number}.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]
    render(config, Progress_Spoof(), Main_Spoof())
    video = cv2.VideoCapture(output_path)
    comparison = cv2.VideoCapture(f"testing/pysonic-test-comparison-{test_number}.mp4")
    compare_videos(video, comparison)
    print(f"Test {test_number} validation complete!")

if __name__ == "__main__":
    test_config(1, "testing/test_1_audio.webm.wav_Audio.mp4")
    test_config(2, "testing/test_2_audio.wav_Audio.mp4")
    
    print("Validation complete!")

    os.remove("testing/test_1_audio.webm.wav_Audio.mp4")
    os.remove("testing/test_2_audio.wav_Audio.mp4")