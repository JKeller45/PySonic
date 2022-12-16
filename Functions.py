import scipy.fftpack as fftp
import numpy as np
from matplotlib import pyplot as plt
from math import log10, sin, radians
import cv2
from PIL import Image as im

def plot_fft(freqs_side, FFT_side):
    plt.subplot(311)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.xscale("log")
    plt.xlim((0,25000))
    plt.show()

def alpha_composite(foreground, background):
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return background

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
            output_image[ycoord - y][xcoord + x] = [color[0], color[1], color[2], 255]
    return output_image

def draw_bars(args):
    backgroud, num_bars, heights, color, width, separation = args
    transparent = np.zeros((len(backgroud), len(backgroud[0]), 4))
    xcoord = 0
    for i in range(num_bars):
        draw_rect(transparent, xcoord, 1079, width, heights[i] + 1, color)
        xcoord += (width + separation)
    #transparent = cv2.GaussianBlur(transparent, (3,3), cv2.BORDER_DEFAULT)
    return cv2.cvtColor(alpha_composite(transparent, cv2.cvtColor(backgroud, cv2.COLOR_BGR2BGRA)), cv2.COLOR_BGRA2BGR)

def bins(freq, amp, heights, num_bars, width):
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
            add_height(heights, c, amp[i], 90, "middle", width)
    heights = heights / 1_000_000_000
    if max(heights) > 300:
        heights = heights / (max(heights) / 300)
    return heights

def add_height(heights, group, amp, angle, side, width):
    if angle <= 0 or group < 0 or group >= len(heights):
        return
    heights[group] += amp * sin(radians(angle))
    if side == "left" or side == "middle":
        add_height(heights, group - 1, amp, angle - 2 * width * log10(group), "left", width)
    if side == "right" or side == "middle":
        add_height(heights, group + 1, amp, angle - 2 * width * log10(group), "right", width)