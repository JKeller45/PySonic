import scipy.fftpack as fftp
import numpy as np
from matplotlib import pyplot as plt
from math import log10, sin, radians

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
    backgroud, num_bars, heights, color, width, separation = args
    xcoord = 0
    for i in range(num_bars):
        draw_rect(backgroud, xcoord, 1079, width, heights[i] + 1, color)
        xcoord += (width + separation)
    return backgroud

def bins(freq, amp, heights, num_bars):
    for c,v in enumerate(freq):
        if v == 0:
            continue
        freq[c] = log10(v)
    bins = np.linspace(log10(20),log10(25000),num_bars)
    for c,_ in enumerate(bins):
        if c == 0:
            continue
        for i,f in enumerate(freq):
            if f <= bins[c - 1]:
                continue
            if f > bins[c]:
                break
            add_height(heights, c, amp[i], 90, "middle")
    heights = heights / 1_000_000_000
    if max(heights) > 300:
        heights = heights / (max(heights) / 300)
    return heights

def add_height(heights, group, amp, angle, side):
    if angle <= 0 or group < 0 or group >= len(heights):
        return
    heights[group] += amp * sin(radians(angle))
    if side == "left" or side == "middle":
        add_height(heights, group - 1, amp, angle - 15, "left")
    if side == "right" or side == "middle":
        add_height(heights, group + 1, amp, angle - 15, "right")