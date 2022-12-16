import scipy.io.wavfile as wavfile
import numpy as np
import cv2
from multiprocessing import Pool
import tomllib as tl
from moviepy.editor import VideoFileClip, AudioFileClip
import Functions as F
from copy import deepcopy


if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)

    FILE = config["settings"]["FILE"]
    color = config["settings"]["color"]
    background = config["settings"]["background"]
    width = config["settings"]["width"]
    connected = config["settings"]["connected_bars"]
    separation = config["settings"]["separation"]
    frame_rate = config["settings"]["frame_rate"]

    background = cv2.imread(background)
    background = cv2.resize(background, (1920,1080), interpolation=cv2.INTER_AREA)

    if connected:
        separation = 0
    else:
        separation = 4

    fs_rate, signal = wavfile.read(FILE)
    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate
    print ("Timestep between samples Ts", Ts)

    num_frames = int((1/frame_rate)/Ts)
    curr_step = num_frames
    prev_step = 0
    
    num_bars = 1920 // (width + separation)
    if num_bars >= 1920:
        num_bars = 1919
    ffts = []

    length_in_frames = 30
    length_in_seconds = 1/frame_rate * length_in_frames

    result = cv2.VideoWriter(f'{FILE}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (1920,1080))

    for _ in range(3):
        args = []
        for n in range(int(length_in_frames/3)):
            args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
            curr_step += num_frames
            prev_step += num_frames
        with Pool(processes=10) as pool:
            ffts = pool.map(F.calc_fft, args)

        args = []
        for n in ffts:
            args.append((deepcopy(background), num_bars, F.bins(n[0], n[1], np.ones(num_bars), num_bars, width), color, width, separation))
        with Pool(processes=10) as pool:
            output = pool.map(F.draw_bars, args)
        args = []

        for f in output:
            result.write(f)
        output = []
        ffts = []
        
    result.release()
    video = VideoFileClip(f"{FILE}.mp4")
    audio = AudioFileClip(FILE)
    audio = audio.subclip(0, length_in_seconds)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f"{FILE}_Audio.mp4")