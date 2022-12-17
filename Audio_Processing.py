import scipy.io.wavfile as wavfile
import numpy as np
import cv2
from multiprocessing import Pool
import tomllib as tl
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import Functions as F
from copy import deepcopy

def render(config):
    background = cv2.imread(config["background"])
    background = cv2.resize(background, config["size"], interpolation=cv2.INTER_AREA)

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

    length_in_seconds = config["length"]
    length_in_frames = config["frame_rate"] * length_in_seconds

    if config["interpolation"]:
        result = cv2.VideoWriter(f'{config["FILE"]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"] * 2, config["size"])
    else:
        result = cv2.VideoWriter(f'{config["FILE"]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"], config["size"])
    FILE = config["FILE"]

    for _ in range(5):
        args = []
        for n in range(int(length_in_frames/5)):
            args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
            curr_step += num_frames
            prev_step += num_frames
        with Pool(processes=10) as pool:
            ffts = pool.map(F.calc_fft, args)

        args = []
        heights = []
        for n in ffts:
            heights.append(F.bins(n[0], n[1], np.ones(num_bars), num_bars, config["width"]))
        for c,n in enumerate(ffts):
            args.append((deepcopy(background), num_bars, heights[c], config))
        with Pool(processes=10) as pool:
            output = pool.map(F.draw_bars, args)
        args = []

        if config["interpolation"]:
            midpoint_heights = []
            for c,h in enumerate(heights):
                if c == 0:
                    pass
                midpoint_heights.append((h + heights[c - 1]) // 2)
            for c,n in enumerate(ffts):
                args.append((deepcopy(background), num_bars, midpoint_heights[c], config))
            with Pool(processes=10) as pool:
                midpoints = pool.map(F.draw_bars, args)
        args = []

        for i,f in enumerate(output):
            result.write(f)
            if config["interpolation"]:
                result.write(midpoints[i])
        output = []
        midpoints = []
        ffts = []
        
    result.release()
    video = VideoFileClip(f"{FILE}.mp4")
    audio = AudioFileClip(FILE)
    audio = audio.subclip(0, length_in_seconds)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f"{FILE}_Audio.mp4")

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]
    render(config)