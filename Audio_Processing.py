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

    config = config["settings"]

    background = cv2.imread(config["background"])
    background = cv2.resize(background, config["size"], interpolation=cv2.INTER_AREA)

    if config["connected_bars"]:
        config["background"] = 0

    fs_rate, signal = wavfile.read(config["FILE"])
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

    num_frames = int((1/config["frame_rate"])/Ts)
    curr_step = num_frames
    prev_step = 0
    
    if config["horizontal_bars"]:
        num_bars = config["size"][1] // (config["width"] + config["separation"])
    else:
        num_bars = config["size"][0] // (config["width"] + config["separation"])
    if num_bars >= config["size"][0]:
        num_bars = config["size"][0] - 1
    ffts = []

    length_in_seconds = 178
    length_in_frames = config["frame_rate"] * length_in_seconds

    result = cv2.VideoWriter(f'{config["FILE"]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"], config["size"])

    for _ in range(20):
        args = []
        for n in range(int(length_in_frames/20)):
            args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
            curr_step += num_frames
            prev_step += num_frames
        with Pool(processes=10) as pool:
            ffts = pool.map(F.calc_fft, args)

        args = []
        for n in ffts:
            args.append((deepcopy(background), num_bars, F.bins(n[0], n[1], np.ones(num_bars), num_bars, config["width"]), config))
        with Pool(processes=10) as pool:
            output = pool.map(F.draw_bars, args)
        args = []

        for f in output:
            result.write(f)
        output = []
        ffts = []
        
    FILE = config["FILE"]
    result.release()
    video = VideoFileClip(f"{FILE}.mp4")
    audio = AudioFileClip(FILE)
    audio = audio.subclip(0, length_in_seconds)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f"{FILE}_Audio.mp4")