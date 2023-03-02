import scipy.io.wavfile as wavfile
import numpy as np
import cv2
from multiprocessing import Pool
import tomllib as tl
from moviepy.editor import VideoFileClip, AudioFileClip
import Functions as F
import Classes
import logging
import os

def create_gpu_group(l, n, config):
    for i in range(0, len(l), n):
        yield (l[i:i + n], config)

def gpu_task(groups):
    args, config = groups
    output = []
    if config["solar"]:
        for arg in args:
            output.append(F.draw_circle(arg))
    elif config["wave"]:
        for arg in args:
            output.append(F.draw_wave(arg))
    else:
        for arg in args:
            output.append(F.draw_bars(arg))
    return output
        
def render(config, progress, main):
    """
    The main render function

    Parameters
    ----------
    config (dict): the dictionary with all render settings
    progress (ttk.Progressbar): the progress bar in the UI. Used for updating
    main (tk.mainwindow): the main window. Used for refreshing the app
    """
    logging.basicConfig(filename='log.log', level=logging.WARNING)
    if config["AISS"]:
        config["size"] = (config["size"][0] // 2, config["size"][1] // 2)
        config["width"] = max(config["width"] // 2, 1)
        config["separation"] //= 2

    if type(config["background"]) == str:
        background = cv2.imread(config["background"])
    else:
        background = config["background"]
    background = cv2.resize(background, config["size"], interpolation=cv2.INTER_AREA)
    config["background"] = background

    if config["FILE"][-4:] != ".wav":
        if config["FILE"][-4:] == ".mp3":
            raise IOError("MP3 File Support Is In The Works") # Temp
            sound = AudioFileClip(config["FILE"])
        else:
            raise IOError("File Type Not Supproted")
        fs_rate, signal = sound.fps, sound.to_soundarray()
        print(signal)
    else:
        fs_rate, signal = wavfile.read(config["FILE"])
    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate

    num_frames = int((1/config["frame_rate"])/Ts)
    curr_step = num_frames
    prev_step = 0

    if config["wave"]:
        config["separation"] = 0
        config["width"] = 1
    if config["position"] == "Left" or config["position"] == "Right":
        num_bars = config["size"][1] // (config["width"] + config["separation"])
    else:
        num_bars = config["size"][0] // (config["width"] + config["separation"])
    if num_bars >= config["size"][0]:
        num_bars = config["size"][0] - 1
    if config["solar"]:
        num_bars = 360
    ffts = []

    length_in_seconds = config["length"]
    length_in_frames = config["frame_rate"] * length_in_seconds

    if config["AISS"]:
        result = cv2.VideoWriter(f'{config["FILE"]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"], (config["size"][0] * 2, config["size"][1] * 2))
    else:
        result = cv2.VideoWriter(f'{config["FILE"]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"], config["size"])
    FILE = config["FILE"]

    if config["use_gpu"]:
        loops = length_in_frames // 300 + 1
    else:
        loops = length_in_frames // 300 + 8
    for _ in range(loops):
        args = []
        for n in range(int(length_in_frames / loops)):
            if curr_step >= len(signal):
                break
            args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
            curr_step += num_frames
            prev_step += num_frames
        with Pool(processes=os.cpu_count() // 2) as pool:
            ffts = pool.map(F.calc_fft, args)
        args = []
        
        heights = []
        output = []
        for n in ffts:
            heights.append(F.bins(n[0], n[1], np.ones(num_bars), num_bars, config))
        for c,n in enumerate(ffts):
                args.append((num_bars, heights[c], config))
        if not config["use_gpu"]:
            with Pool(processes=os.cpu_count() // 2) as pool:
                if config["solar"]:
                    output = pool.map(F.draw_circle, args)
                elif config["wave"]:
                    output = pool.map(F.draw_wave, args)
                else:
                    output = pool.map(F.draw_bars, args)
        else:
            groups = list(create_gpu_group(args, len(args) // (os.cpu_count() // 2), config))
            with Pool(processes=os.cpu_count() // 2) as pool:
                output = pool.map(gpu_task, groups)
        groups = []
        args = []

        flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
        output = flatten_list(output)

        for _,f in enumerate(output):
            result.write(f)
        output = []
        ffts = []

        progress.step(100 // loops)
        main.update()

    result.release()
    try:
        video = VideoFileClip(f"{FILE}.mp4")
        audio = AudioFileClip(FILE)
        audio = audio.subclip(0, length_in_seconds)
        final_clip = video.set_audio(audio)
        final_clip.write_videofile(f"{FILE}_Audio.mp4", logger=None)
    except Exception as e:
        logging.error("MoviePy Error, check your FFMPEG distribution", exc_info=True)
    
    progress.step(100)
    main.update()
    return

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]

    from time import perf_counter
    start = perf_counter()
    render(config, Classes.Progress_Spoof(), Classes.Main_Spoof())
    middle = perf_counter()
    config["use_gpu"] = False
    render(config, Classes.Progress_Spoof(), Classes.Main_Spoof())
    end = perf_counter()

    print(f"GPU: {middle-start}")
    print(f"CPU: {end-middle}")