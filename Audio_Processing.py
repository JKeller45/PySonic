import scipy.io.wavfile as wavfile
import numpy as np
import cv2
from multiprocessing import Pool
import tomllib as tl
from moviepy.editor import VideoFileClip, AudioFileClip
from copy import deepcopy
import Functions as F
import Classes
        
def render(config, progress, main):
    """
    The main render function

    Parameters
    ----------
    config (dict): the dictionary with all render settings
    progress (ttk.Progressbar): the progress bar in the UI. Used for updating
    main (tk.mainwindow): the main window. Used for refreshing the app
    """
    if config["AISS"]:
        config["size"] = (config["size"][0] // 2, config["size"][1] // 2)
        config["width"] = max(config["width"] // 2, 1)
        config["separation"] //= 2

    if type(config["background"]) == str:
        background = cv2.imread(config["background"])
    else:
        background = config["background"]
    background = cv2.resize(background, config["size"], interpolation=cv2.INTER_AREA)

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
    if config["horizontal_bars"]:
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

    loops = length_in_frames // 300 + 8
    for _ in range(loops):
        args = []
        for n in range(int(length_in_frames / loops)):
            if curr_step >= len(signal):
                break
            args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
            curr_step += num_frames
            prev_step += num_frames
        with Pool(processes=10) as pool:
            ffts = pool.map(F.calc_fft, args)
        args = []
        
        heights = []
        for n in ffts:
            heights.append(F.bins(n[0], n[1], np.ones(num_bars), num_bars, config))
        for c,n in enumerate(ffts):
                args.append((deepcopy(background), num_bars, heights[c], config))
        with Pool(processes=10) as pool:
            if config["solar"]:
                output = pool.map(F.draw_circle, args)
            elif config["wave"]:
                output = pool.map(F.draw_wave, args)
            else:
                output = pool.map(F.draw_bars, args)
        args = []

        for _,f in enumerate(output):
            result.write(f)
        output = []
        ffts = []

        progress.step(100 // loops)
        main.update()
        
    result.release()
    video = VideoFileClip(f"{FILE}.mp4")
    audio = AudioFileClip(FILE)
    audio = audio.subclip(0, length_in_seconds)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f"{FILE}_Audio.mp4")
    
    progress.step(100)
    main.update()
    return

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]
    render(config, Classes.Progress_Spoof(), Classes.Main_Spoof())