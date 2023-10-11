import scipy.io.wavfile as wavfile
from scipy import signal
import numpy as np
import cv2
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
import tomllib as tl
import Functions as F
from Classes import *
import logging
import os
from itertools import cycle
import numpy.typing as npt
import subprocess
import warnings
from sys import platform
    
def pick_react(args) -> npt.ArrayLike:
    background, num_bars, heights, avg_heights, settings = args
    if settings.wave:
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

def render(config: dict, progress, main):
    """
    The main render function

    Parameters
    ----------
    config (dict): the dictionary with all render settings
    progress (ttk.Progressbar): the progress bar in the UI. Used for updating
    main (tk.mainwindow): the main window. Used for refreshing the app
    """
    # logging.basicConfig(filename='log.log', level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
    # logging.log(logging.INFO, "Starting Render...")
    settings = Settings(audio_file=config["FILE"], output=config["output"], length=config["length"], size=np.array(config["size"]),
                            color=tuple(config["color"]), background=config["background"],frame_rate=config["frame_rate"],
                            separation=config["separation"], position=config["position"], AISS=config["AISS"], 
                            width=config["width"], wave=config["wave"], circular_looped_video=config["circular_looped_video"], 
                            snowfall=config["snowfall"], zoom=config["zoom"], snow_seed=int(np.random.rand() * 1000000))
    
    settings.color = settings.color[::-1]

    progress.value = .01
    main.update()
    warnings.simplefilter("ignore", np.ComplexWarning)

    if settings.AISS:
        settings.size = (settings.size[0] // 2, settings.size[1] // 2)
        settings.width = max(settings.width // 2, 1)
        settings.separation //= 2

    # logging.log(logging.INFO, "Loading Audio...")
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    if settings.audio_file[-4:] != ".wav":
        convert_args = ["ffmpeg","-y", "-i", settings.audio_file, "-acodec", "pcm_s32le", "-ar", "44100", f"{settings.audio_file}.wav", ]
        try:
            if subprocess.run(convert_args, startupinfo=si).returncode == 0:
                settings.audio_file = f"{settings.audio_file}.wav"
            else:
                # logging.error("ffmpeg failure", exc_info=True)
                pass
        except Exception as e:
            raise e
            pass

    fs_rate, audio = wavfile.read(settings.audio_file)
    #print ("Frequency sampling", fs_rate)
    l_audio = len(audio.shape)
    if l_audio == 2:
        audio = audio.sum(axis=1) / 2
    N = audio.shape[0]
    secs = N / float(fs_rate)
    #print ("secs", secs)

    if settings.wave:
        settings.separation = 0
        settings.width = 1
    if settings.position == "Left" or settings.position == "Right":
        num_bars = settings.size[1] // (settings.width + settings.separation)
    else:
        num_bars = settings.size[0] // (settings.width + settings.separation)
    if num_bars >= settings.size[0]:
        num_bars = settings.size[0] - 1
    
    length_in_seconds = secs if settings.length <= 0 else min([settings.length, secs])
    settings.length = min([settings.length, secs])
    length_in_frames = int(settings.frame_rate * length_in_seconds)
    backgrounds = None
    background = None

    # logging.log(logging.INFO, "Loading Background...")
    if settings.background[-4:] in (".mp4",".avi",".mov",".MOV"):
        vid = cv2.VideoCapture(settings.background)
        backgrounds = []
        background = []
        success, image = vid.read()
        fps = vid.get(cv2.CAP_PROP_FPS)
        count = 1
        vid_length = 30 * 5
        while success and count <= vid_length:
            if settings.frame_rate < fps:
                background.append(cv2.resize(image, settings.size, interpolation=cv2.INTER_CUBIC))
                for _ in range(round(fps / settings.frame_rate)):
                    success, image = vid.read()
            elif settings.frame_rate >= fps:
                for _ in range(round(settings.frame_rate / fps)):
                    background.append(cv2.resize(image, settings.size, interpolation=cv2.INTER_CUBIC))
                success, image = vid.read()
            backgrounds.append(count - 1)
            count += 1
        if settings.circular_looped_video:
            background = background + background[::-1]
            backgrounds = backgrounds + backgrounds[::-1]
        background = np.array(background, dtype=np.uint8)
        backgrounds = cycle(backgrounds)
    elif type(settings.background) == str:
        background = cv2.imread(settings.background)
        background = cv2.resize(background, settings.size, interpolation=cv2.INTER_CUBIC)
    else:
        background = settings.background
        background = cv2.resize(background, settings.size, interpolation=cv2.INTER_CUBIC)

    path, file_name = os.path.split(settings.audio_file)

    if settings.AISS:
        result = cv2.VideoWriter(f'{settings.output}{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), settings.frame_rate, (settings.size[0] * 2, settings.size[1] * 2))
    else:
        result = cv2.VideoWriter(f'{settings.output}{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), settings.frame_rate, settings.size)

    # logging.log(logging.INFO, "Processing Audio...")

    audio = audio[:int(fs_rate * length_in_seconds)]
    hop_scale = 2
    frame_size = (fs_rate * length_in_seconds * hop_scale) // (length_in_frames - 1 + hop_scale)
    hop_size = int(frame_size / hop_scale)
    freqs, times, amps = signal.stft(audio, fs_rate, nperseg=frame_size, noverlap=hop_size, return_onesided=True)
    amps = amps.T[:length_in_frames].astype(np.int64)
    freqs = np.log10(freqs[1:])
    heights = []
    args = []

    # logging.log(logging.INFO, "Calculating Heights...")

    for amp in amps:
        heights = [0] * num_bars
        heights = F.bins(freqs, amp, heights, num_bars, settings)
        if type(backgrounds) == cycle:
            args.append((Frame_Information(True, "", 0, next(backgrounds)), num_bars, heights, heights.mean(axis=0), settings))
        else:
            args.append((Frame_Information(False, "", 0, 0), num_bars, heights, heights.mean(axis=0), settings))

    # logging.log(logging.INFO, "Calculating Average Heights...")

    heights = signal.savgol_filter(heights, 30, 15, axis=0)
    max_height = max(max(arg[2]) for arg in args)
    average_heights = [arg[3] / 20000000 + 1 for arg in args]
    average_lows = [np.mean(arg[2][0:4 * num_bars // 100]) * (settings.size[1] // 5) // max_height for arg in args]
    average_lows = signal.savgol_filter(average_lows, 17, 7)
    average_heights = np.cumsum(average_heights)

    # logging.log(logging.INFO, "Rendering...")
    with SharedMemoryManager() as smm:
        frame_bg = smm.SharedMemory(size=background.nbytes)
        shared_background = np.ndarray(background.shape, dtype=np.uint8, buffer=frame_bg.buf)
        shared_background[:] = background[:]
        del background
        progress_counter = 0
        args = [(Frame_Information(arg[0].video, frame_bg.name, shared_background.size, arg[0].frame_number), arg[1], arg[2] * (settings.size[1] // 5) // max_height, 
                    (average_heights[index], average_lows[index]), arg[4]) for index,arg in enumerate(args)]
        with Pool(processes=4, maxtasksperchild=50) as pool:
            outputs = pool.imap(pick_react, args)
            sr = None
            if settings.AISS:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                try:
                    path = F.find_by_relative_path(r"assets/ESPCN_x2.pb")
                except Exception as e:
                    # logging.error(e, exc_info=True)
                    pass
                sr.readModel(path)
                sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                sr.setModel("espcn", 2)
            for frame in outputs:
                if settings.AISS:
                    frame = sr.upsample(frame)
                result.write(frame)
                progress_counter += 1
                if progress_counter / length_in_frames >= .01:
                    progress_counter = 0
                    progress.value += .01
                    main.update()
                del frame
            del sr
    result.release()

    progress.value = .99
    main.update()

    # logging.log(logging.INFO, "Combining Audio...")
    print(f'{settings.output}{file_name}.mp4', settings.audio_file, f"{settings.output}{file_name}_Audio.mp4")
    combine_cmds = ["ffmpeg","-y", "-i", f'{settings.output}{file_name}.mp4', '-i', settings.audio_file, '-map', '0', '-map', '1:a', '-c:v', 'copy', '-shortest', f"{settings.output}{file_name}_Audio.mp4"]

    try:
        if subprocess.run(combine_cmds, startupinfo=si).returncode == 0:
            os.remove(f'{settings.output}{file_name}.mp4')
            os.remove(settings.audio_file)
        else:
            # logging.error(e, exc_info=True)
            pass
    except Exception as e:
        raise e
        pass

    # logging.log(logging.INFO, "Done")
    progress.value = 1
    main.update()

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)
        config = config["settings"]

    from time import perf_counter
    start = perf_counter()
    render(config, Progress_Spoof(), Main_Spoof())
    middle = perf_counter()

    print(f"CPU: {middle-start}")