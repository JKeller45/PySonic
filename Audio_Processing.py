import scipy.io.wavfile as wavfile
import numpy as np
import cv2
from multiprocessing import Pool
import tomllib as tl
from moviepy.editor import VideoFileClip, AudioFileClip
import Functions as F
from Classes import Settings, Main_Spoof, Progress_Spoof
import logging
import os
from PIL import Image as im
from itertools import cycle
import numpy.typing as npt

def calc_heights_async(fft: tuple[npt.ArrayLike, npt.ArrayLike], background: npt.ArrayLike, num_bars: int, settings: Settings) -> npt.ArrayLike:
    heights = F.bins(fft[0], fft[1], np.ones(num_bars), num_bars, settings)
    if settings.solar:
        return F.draw_circle(background, num_bars, heights, settings)
    elif settings.wave:
        return F.draw_wave(background, num_bars, heights, settings)
    else:
        return F.draw_bars(background, num_bars, heights, settings)

def render(config: dict, progress, main, ret_val: list):
    """
    The main render function

    Parameters
    ----------
    config (dict): the dictionary with all render settings
    progress (ttk.Progressbar): the progress bar in the UI. Used for updating
    main (tk.mainwindow): the main window. Used for refreshing the app
    """

    settings = Settings(audio_file=config["FILE"], output=config["output"], length=config["length"], size=config["size"], \
                            color=config["color"], background=config["background"],frame_rate=config["frame_rate"], width=config["width"],\
                            separation=config["separation"], position=config["position"], SSAA=config["SSAA"], AISS=config["AISS"], \
                            solar=config["solar"], wave=config["wave"], use_gpu=config["use_gpu"], memory_compression=config["memory_compression"], \
                            circular_looped_video=config["circular_looped_video"])

    progress.step(5)
    logging.basicConfig(filename='log.log', level=logging.WARNING)
    if settings.AISS:
        settings.size = (settings.size[0] // 2, settings.size[1] // 2)
        settings.width = max(settings.width // 2, 1)
        settings.separation //= 2

    if settings.audio_file[-4:] != ".wav":
        if settings.audio_file[-4:] == ".mp3":
            raise IOError("MP3 File Support Is In The Works") # Temp
            sound = AudioFileClip(settings.audio_file)
        else:
            raise IOError("File Type Not Supproted")
        fs_rate, signal = sound.fps, sound.to_soundarray()
        print(signal)
    else:
        fs_rate, signal = wavfile.read(settings.audio_file)
    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate

    num_frames = int((1/settings.frame_rate)/Ts)
    curr_step = num_frames
    prev_step = 0

    if settings.wave:
        settings.separation = 0
        settings.width = 1
    if settings.position == "Left" or settings.position == "Right":
        num_bars = settings.size[1] // (settings.width + settings.separation)
    else:
        num_bars = settings.size[0] // (settings.width + settings.separation)
    if num_bars >= settings.size[0]:
        num_bars = settings.size[0] - 1
    if settings.solar:
        num_bars = 360
    ffts = []

    length_in_seconds = settings.length
    length_in_frames = int(settings.frame_rate * length_in_seconds)

    if settings.background[-4:] in (".mp4",".avi",".mov",".MOV"):
        vid = cv2.VideoCapture(settings.background)
        backgrounds = []
        success, image = vid.read()
        count = 1
        vid_length = length_in_frames
        if settings.circular_looped_video:
            vid_length //= 2
        while success and count <= vid_length:
            backgrounds.append(cv2.resize(image, settings.size, interpolation=cv2.INTER_AREA))
            success, image = vid.read()
            count += 1
        backgrounds = backgrounds + backgrounds[::-1]
        backgrounds = cycle(backgrounds)
    elif type(settings.background) == str:
        background = cv2.imread(settings.background)
        background = cv2.resize(background, settings.size, interpolation=cv2.INTER_AREA)
    else:
        background = settings.background
        background = cv2.resize(background, settings.size, interpolation=cv2.INTER_AREA)

    path, file_name = os.path.split(settings.audio_file)

    if settings.AISS:
        result = cv2.VideoWriter(f'{settings.output}{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), settings.frame_rate, (settings.size[0] * 2, settings.size[1] * 2))
    else:
        result = cv2.VideoWriter(f'{settings.output}{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), settings.frame_rate, settings.size)

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
                outputs.append(FramePool.apply_async(calc_heights_async, (fft, background, num_bars, settings)))
                fft = None
            for c,frame in enumerate(outputs):
                img = frame.get()
                if settings.memory_compression:
                    result.write(np.array(im.open(img)))
                    img.flush()
                else:
                    result.write(img)
                outputs[c] = None
                progress.step(90 / length_in_frames)
                main.update()
    result.release()

    try:
        video = VideoFileClip(f'{settings.output}{file_name}.mp4')
        audio = AudioFileClip(settings.audio_file)
        audio = audio.subclip(0, length_in_seconds)
        final_clip = video.set_audio(audio)
        final_clip.write_videofile(settings.audio_file + "_Audio.mp4", logger=None)
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
    render(config, Progress_Spoof(), Main_Spoof(), [])
    middle = perf_counter()

    print(f"CPU: {middle-start}")