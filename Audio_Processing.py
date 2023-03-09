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
from PIL import Image as im
from itertools import cycle

def calc_heights_async(fft, background, num_bars, config):
    heights = F.bins(fft[0], fft[1], np.ones(num_bars), num_bars, config)
    if config["solar"]:
        return F.draw_circle(background, num_bars, heights, config)
    elif config["wave"]:
        return F.draw_wave(background, num_bars, heights, config)
    else:
        return F.draw_bars(background, num_bars, heights, config)

def render(config, progress, main, ret_val):
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

    if config["background"][-4:] in (".mp4",".avi",".mov",".MOV"):
        vid = cv2.VideoCapture(config["background"])
        backgrounds = []
        success, image = vid.read()
        count = 1
        vid_length = length_in_frames
        if config["circular_looped_video"]:
            vid_length //= 2
        while success and count <= vid_length:
            backgrounds.append(image)
            success, image = vid.read()
            count += 1
        for c,v in enumerate(backgrounds):
            backgrounds[c] = cv2.resize(v, config["size"], interpolation=cv2.INTER_AREA)
        backgrounds = backgrounds + backgrounds[::-1]
        backgrounds = cycle(backgrounds)
    elif type(config["background"]) == str:
        background = cv2.imread(config["background"])
        background = cv2.resize(background, config["size"], interpolation=cv2.INTER_AREA)
    else:
        background = config["background"]
        background = cv2.resize(background, config["size"], interpolation=cv2.INTER_AREA)

    path, file_name = os.path.split(config["FILE"])

    if config["AISS"]:
        result = cv2.VideoWriter(f'{config["output"]}{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"], (config["size"][0] * 2, config["size"][1] * 2))
    else:
        result = cv2.VideoWriter(f'{config["output"]}{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["frame_rate"], config["size"])

    args = []
    outputs = []
    for _ in range(length_in_frames):
        if curr_step >= len(signal):
            break
        args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
        curr_step += num_frames
        prev_step += num_frames
    cpus = os.cpu_count()
    with Pool(processes=cpus // 3 * 2) as FramePool:
        with Pool(processes=cpus // 3) as FFTPool:
            ffts = FFTPool.imap(F.calc_fft, args, chunksize=3)
            for c,fft in enumerate(ffts):
                if backgrounds:
                    background = next(backgrounds)
                outputs.append(FramePool.apply_async(calc_heights_async, (fft, background, num_bars, config)))
                fft = None
            for c,frame in enumerate(outputs):
                img = frame.get()
                if config["memory_compression"]:
                    result.write(np.array(im.open(img)))
                    img.flush()
                else:
                    result.write(img)
                outputs[c] = None
                progress.step(100 / length_in_frames)
                main.update()
    result.release()

    try:
        video = VideoFileClip(f'{config["output"]}{file_name}.mp4')
        audio = AudioFileClip(config["FILE"])
        audio = audio.subclip(0, length_in_seconds)
        final_clip = video.set_audio(audio)
        final_clip.write_videofile(config["FILE"] + "_Audio.mp4", logger=None)
    except Exception as e:
        logging.error("MoviePy Error, check your FFMPEG distro", exc_info=True)
    
    progress.step(100)
    main.update()
    ret_val.append("Done!")
    return

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]

    from time import perf_counter
    start = perf_counter()
    render(config, Classes.Progress_Spoof(), Classes.Main_Spoof(), [])
    middle = perf_counter()

    print(f"CPU: {middle-start}")