import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftp
import numpy as np
from matplotlib import pyplot as plt
import cv2
from math import log10
from multiprocessing import Pool

FILE = "C:/Users/james/Desktop/Above-a-Sea-of-Fog.wav"
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
    num_bars, heights, color = args
    output_image = np.zeros((1080,1920,3), dtype=np.ushort)
    xcoord = 0
    for i in range(num_bars):
        draw_rect(output_image, xcoord, 1079, 6, heights[i] + 1, color)
        xcoord += 10
    return output_image

def bins(freq, amp, heights, num_bars):
    group_size = len(freq) // num_bars
    for c,v in enumerate(freq):
        if v <= 0:
            continue
        freq[c] = log10(v)
    pos = 0
    for c,v in enumerate(freq):
        if c >= len(amp) or pos >= len(heights):
            break
        heights[pos] += amp[c]
        if c >= (pos + 1) * group_size:
            pos += 1
    return heights / 1_000_000_000

if __name__ == "__main__":
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

    num_frames = int((1/30)/Ts)
    curr_step = num_frames
    prev_step = 0
    num_bars = 1920 // 10
    ffts = []

    args = []
    for n in range(600):
        args.append((prev_step, curr_step, Ts, signal[prev_step:curr_step]))
        curr_step += num_frames
        prev_step += num_frames
    with Pool() as pool:
        ffts = pool.map(calc_fft, args)

    args = []
    for n in ffts:
        args.append((num_bars, bins(n[0], n[1], np.ones(num_bars), num_bars), color))
    with Pool() as pool:
        output = pool.map(draw_bars, args)

    result = cv2.VideoWriter(f'{FILE}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))
    for f in output:
        result.write(f)