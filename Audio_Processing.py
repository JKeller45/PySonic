import scipy.io.wavfile as wavfile
import numpy as np
import cv2
from multiprocessing import Pool
import tomllib as tl
import Functions as F
from Classes import Settings, EffectSettings, Main_Spoof, Progress_Spoof
import logging
import os
from PIL import Image as im
from itertools import cycle
import numpy.typing as npt
import subprocess

def calc_heights_async(args) -> tuple[npt.ArrayLike, int, int]:
    num_bars, settings = args[4:6]
    freq, amp = F.calc_fft(args[0:4])
    heights = F.bins(freq, amp, np.ones(num_bars), num_bars, settings)
    return (heights, np.mean(amp[0:12]) / 2_000_000_000, np.mean(amp[0:2]) / 1_000_000_000)
    
def pick_react(background: npt.ArrayLike, num_bars: int, heights: npt.ArrayLike, avg_heights: tuple[float, float], settings: Settings) -> npt.ArrayLike:
    if settings.solar:
        return F.draw_circle(background, num_bars, heights, avg_heights, settings)
    elif settings.wave:
        return F.draw_wave(background, num_bars, heights, avg_heights, settings)
    else:
        return F.draw_bars(background, num_bars, heights, avg_heights, settings)

def impose_height_diff(last, curr):
    diff = np.subtract(curr, last)
    for c,v in enumerate(diff):
        if v > 50:
            curr[c] = last[c] + 40
        elif v < -50:
            curr[c] = last[c] - 40
    return curr

def render(config: dict, progress, main, pools: list, ret_val: list):
    """
    The main render function

    Parameters
    ----------
    config (dict): the dictionary with all render settings
    progress (ttk.Progressbar): the progress bar in the UI. Used for updating
    main (tk.mainwindow): the main window. Used for refreshing the app
    """

    settings = Settings(audio_file=config["FILE"], output=config["output"], length=config["length"], size=config["size"],
                            color=config["color"], background=config["background"],frame_rate=config["frame_rate"], width=config["width"],
                            separation=config["separation"], position=config["position"], SSAA=config["SSAA"], AISS=config["AISS"],
                            solar=config["solar"], wave=config["wave"], use_gpu=config["use_gpu"], memory_compression=config["memory_compression"],
                            circular_looped_video=config["circular_looped_video"], snowfall=config["snowfall"], zoom=config["zoom"],
                            effect_settings=None)
    
    settings.effect_settings = EffectSettings(seed=int(np.random.rand() * 1000000))

    progress.step(5)
    logging.basicConfig(filename='log.log', level=logging.WARNING)
    if settings.AISS:
        settings.size = (settings.size[0] // 2, settings.size[1] // 2)
        settings.width = max(settings.width // 2, 1)
        settings.separation //= 2

    if settings.audio_file[-4:] != ".wav":
        convert_args = ["ffmpeg","-y", "-i", settings.audio_file, "-acodec", "pcm_s32le", "-ar", "44100", f"{settings.audio_file}.wav"]
        if subprocess.run(convert_args).returncode == 0:
            settings.audio_file = f"{settings.audio_file}.wav"
        else:
            raise IOError("FFMPEG Error: try a different file or file type")
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
    
    length_in_seconds = secs if settings.length == 0 else min([settings.length, secs])
    settings.length = min([settings.length, secs])
    length_in_frames = int(settings.frame_rate * length_in_seconds)
    backgrounds = None

    if settings.background[-4:] in (".mp4",".avi",".mov",".MOV"):
        vid = cv2.VideoCapture(settings.background)
        backgrounds = []
        success, image = vid.read()
        fps = vid.get(cv2.CAP_PROP_FPS)
        count = 1
        vid_length = length_in_frames
        if settings.circular_looped_video:
            vid_length //= 2
        while success and count <= vid_length:
            if settings.frame_rate < fps:
                backgrounds.append(cv2.resize(image, settings.size, interpolation=cv2.INTER_AREA))
                for _ in range(round(fps / settings.frame_rate)):
                    success, image = vid.read()
            elif settings.frame_rate >= fps:
                for _ in range(round(settings.frame_rate / fps)):
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
        args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step], num_bars, settings))
        curr_step += num_frames
        prev_step += num_frames
    cpus = os.cpu_count()
    with Pool(processes=cpus // 3 * 2) as FramePool:
        with Pool(processes=cpus // 3) as FFTPool:
            snowfall_heights = np.zeros(length_in_frames)
            zoom_heights = np.zeros(length_in_frames)
            held_heights = np.zeros((length_in_frames, num_bars))
            pools.append(FramePool)
            pools.append(FFTPool)
            heights = FFTPool.imap(calc_heights_async, args, chunksize=3)
            for c, height in enumerate(heights):
                snowfall = height[1]
                zoom = height[2]
                height = height[0]

                if backgrounds:
                    background = next(backgrounds)

                if settings.snowfall:
                    if snowfall_heights.size >= 1:
                        snowfall_heights[c] = (snowfall + snowfall_heights[c - 1]) // 2 + 30
                    else:
                        snowfall_heights[c] = snowfall + 30

                if settings.zoom:
                    if zoom_heights.size >= 2:
                        zoom_heights[c] = (zoom + zoom_heights[c - 1] + zoom_heights[c - 2]) // 3
                    else:
                        zoom_heights[c] = zoom

                if c > 1:
                    height = impose_height_diff(held_heights[c - 1], height)
                held_heights[c] = height

                outputs.append(FramePool.apply_async(pick_react, (background, num_bars, height, (np.sum(snowfall_heights), zoom_heights[c]), settings)))
            del heights
            for c,frame in enumerate(outputs):
                img = frame.get()
                if settings.memory_compression:
                    frame = np.array(im.open(img))
                    img.flush()
                if settings.AISS or settings.SSAA:
                    frame = F.upsampling(frame, settings)
                result.write(frame)
                outputs[c] = None
                progress.step(90 / length_in_frames)
                main.update()
    result.release()

    try:
        combine_cmds = ["ffmpeg", "-y", "-i", f'{settings.output}{file_name}.mp4', '-i', settings.audio_file, '-map', '0', '-map', '1:a', '-c:v', 'copy', '-shortest', f"{settings.audio_file}_Audio.mp4"]
        if subprocess.run(combine_cmds).returncode == 0:
            os.remove(f'{settings.output}{file_name}.mp4')
            progress.step(100)
            main.update()
            ret_val.append("Done!")
            return
    except Exception:
        logging.error("FFMPEG Error, check your FFMPEG distro", exc_info=True)

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]

    from time import perf_counter
    start = perf_counter()
    render(config, Progress_Spoof(), Main_Spoof(), [], [])
    middle = perf_counter()

    print(f"CPU: {middle-start}")