import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftp
import numpy as np
from matplotlib import pyplot as plt
import cv2
from math import log10
from multiprocessing import Pool
import tomllib as tl
from moviepy.editor import VideoFileClip, AudioFileClip

FILE = "C:/Users/james/Desktop/All-The-Kids-Are_depressed_remix.wav"
color = [255, 255, 255]

def plot_fft(freqs_side, FFT_side):
    plt.subplot(311)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.xscale("log")
    plt.xlim((0,25000))
    plt.show()

def calc_fft(args):
    start, stop, step, signal = args
    t = np.arange(start, stop, step)
    FFT = abs(fftp.fft(signal))
    FFT_side = FFT[range(len(FFT)//2)]
    freqs = fftp.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(len(FFT)//2)]
    return freqs_side, FFT_side

def draw_rect(output_image, xcoord, ycoord, width, height, color):
    for y in range(int(height)):
        for x in range(width):
            output_image[ycoord - y][xcoord + x] = color

def draw_bars(args):
    num_bars, heights, color, width, separation = args
    output_image = np.zeros((1080,1920,3), np.uint8)
    xcoord = 0
    for i in range(num_bars):
        draw_rect(output_image, xcoord, 1079, width, heights[i] + 1, color)
        xcoord += (width + separation)
    return output_image

def bins(freq, amp, heights, num_bars):
    for c,v in enumerate(freq):
        if v == 0:
            continue
        freq[c] = log10(v)
    group_size = freq[-1] // num_bars
    bins = np.linspace(log10(20),log10(25000),num_bars)
    for c,v in enumerate(bins):
        if c == 0:
            continue
        for i,f in enumerate(freq):
            if f <= bins[c - 1]:
                continue
            if f > v:
                break
            heights[c] += amp[i]
    heights = heights / 1_000_000_000
    if max(heights) > 300:
        heights = heights / (max(heights) / 300)
    return heights

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tl.load(f)

    width = config["settings"]["width"]
    connected = config["settings"]["connected_bars"]
    separation = config["settings"]["separation"]
    frame_rate = config["settings"]["frame_rate"]

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

    length_in_frames = 3000
    length_in_seconds = 1/frame_rate * length_in_frames

    args = []
    for n in range(length_in_frames):
        args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
        curr_step += num_frames
        prev_step += num_frames
    with Pool(processes=10) as pool:
        ffts = pool.map(calc_fft, args)

    args = []
    for n in ffts:
        args.append((num_bars, bins(n[0], n[1], np.ones(num_bars), num_bars), color, width, separation))
    with Pool(processes=10) as pool:
        output = pool.map(draw_bars, args)

    result = cv2.VideoWriter(f'{FILE}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (1920,1080))
    for f in output:
        result.write(f)
    result.release()

    video = VideoFileClip(f"{FILE}.mp4")
    audio = AudioFileClip(FILE)
    audio = audio.subclip(0, length_in_seconds)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(f"{FILE}_Audio.mp4")