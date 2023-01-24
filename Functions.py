import scipy.fftpack as fftp
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
from PIL import Image as im

def plot_fft(freqs_side, FFT_side):
    """
    Plots the FFT with matplotlib

    Parameters
    ----------
    freqs_side (np.ndarray): the descrete frequency steps
    FFT_side (np.ndarray): the amplitudes at each frequency step
    """
    plt.subplot(311)
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.xscale("log")
    plt.xlim((0,25000))
    plt.show()

def alpha_composite(foreground, background):
    """
    Implementation of alpha composition. Used for overlaying RGBA images for partial/fully transparent overlays.

    Parameters
    ----------
    foreground (np.ndarray): the image in the foreground
    background (np.ndarray): the image being being composited on top of (the background)

    Returns
    -------
    background (np.ndarray): the fully composited image
    """
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return background

def calc_fft(args):
    """
    Calculates FFT for a given range. This method is designed to be used in a multithreaded way.

    Parameters
    ----------
    start (int): the starting index to calculate from
    stop (int): the ending index to calculate
    step (int): the step size between samples
    signal (np.ndarray): the audio signal to be processed

    Returns
    -------
    freqs_side (np.ndarray): the descrete frequency steps
    FFT_side (np.ndarray): the amplitudes at each frequency step
    """
    start, stop, step, signal = args
    t = np.arange(start, stop, step)
    FFT = abs(fftp.fft(signal))
    FFT_side = FFT[range(len(FFT) // 2)]
    freqs = fftp.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(len(FFT)//2)]
    return freqs_side, FFT_side

def draw_rect(output_image, xcoord, ycoord, config, height):
    """
    Draws the bar of a given size for a given frame.

    Parameters
    ----------
    output_image (np.ndarray): the background image being drawn over
    xcoord (int): the x-coord being drawn at
    ycoord (int): the y-coordinate being drawn at
    config (dict): the config dictionary that contains the render settings
    height (int): the height/length of each rectangle being drawn

    Returns
    -------
    output_image (np.ndarray): the image with the rectangle drawn over it
    """
    xcoord = int(xcoord)
    ycoord = int(ycoord)
    height = int(height)

    if config["inverted_bars"] and config["horizontal_bars"]:
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord - height, ycoord + config["width"]), config["color"], -1)
    elif not config["inverted_bars"] and config["horizontal_bars"]:
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord + height, ycoord + config["width"]), config["color"], -1)
    elif config["inverted_bars"] and not config["horizontal_bars"]:
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord + config["width"], ycoord + height), config["color"], -1)
    else:
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord + config["width"], ycoord - height), config["color"], -1)

    return output_image

def draw_bars(args):
    """
    Draws the bars for a given frame. This method is designed to be used in a multithreaded way.

    Parameters
    ----------
    background (np.ndarray): the background image being drawn over. Should be a deepcopy.
    num_bars (int): the number of bars being drawn on the frame
    heights (list): heights for each bar being drawn
    config (dict): the config dictionary that contains the render settings

    Returns
    -------
    background (np.ndarray): the final frame with all bars drawn over its
    """
    background, num_bars, heights, config = args
    #transparent = np.zeros((len(backgroud), len(backgroud[0]), 4))
    offset = 0
    for i in range(num_bars):
        if config["inverted_bars"] and config["horizontal_bars"]:
            draw_rect(background, config["size"][0] - 1, offset, config, heights[i] + 1)
        if config["inverted_bars"] and not config["horizontal_bars"]:
            draw_rect(background, offset, 0, config, heights[i] + 1)
        if not config["inverted_bars"] and config["horizontal_bars"]:
            draw_rect(background, 0, offset, config, heights[i] + 1)
        if not config["inverted_bars"] and not config["horizontal_bars"]:
            draw_rect(background, offset, config["size"][1] - 1, config, heights[i] + 1)
        offset += (config["width"] + config["separation"])

    if config["SSAA"] or config["AISS"]:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "ESPCN_x2.pb"
        sr.readModel(path)
        sr.setModel("espcn", 2)
        result = sr.upsample(background)
        background = result
        if config["SSAA"]:
            background = np.array(im.fromarray(background).resize((len(background[0]) // 2, len(background) // 2), resample=im.ANTIALIAS))
        #cv2.cvtColor(alpha_composite(transparent, cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)), cv2.COLOR_BGRA2BGR)
    return background

def bins(freq, amp, heights, num_bars, config):
    """
    Computes the heights of each bar from the FFT

    Parameters
    ----------
    freq (np.ndarray): the frequency bins for the FFT
    amp (np.ndarray): the total amplitudes at each frequency bin
    heights (list): the list where heights will be added
    num_bars (int): the number of bars in each frame
    config (dict): the config dictionary that contains the render settings

    Returns
    -------
    heights (list): the final heights of each bar in the frame
    """
    for c,v in enumerate(freq):
        if v == 0:
            continue
        freq[c] = math.log10(v)
    bins = np.linspace(math.log10(50),math.log10(20000),num_bars)
    for c,_ in enumerate(bins):
        if c == 0:
            continue
        for i,f in enumerate(freq):
            if f <= bins[c - 1]:
                continue
            if f > bins[c]:
                break
            if config["circle"]:
                add_height(heights, c, amp[i], 90, "middle", config["width"], lambda angle: math.sin(math.radians(angle)))
            else:
                add_height(heights, c, amp[i], 90, "middle", config["width"], lambda angle: math.sin(math.radians(angle)))
    heights = heights / 1_000_000_000
    heights = heights * (config["frame_rate"] // 30)
    heights = heights * (config["size"][1] / 1080)
    if max(heights) > 300:
        heights = heights / (max(heights) / 300)
    return heights

def add_height(heights, group, amp, angle, side, width, damping):
    """
    Uses a sinoidal decay to add height to adjacent bars to give a more natural, non blocky, look to each frame.

    Parameters
    ----------
    heights (list): the heights of each bar
    group (int): the index of heights that is being added to
    amp (int): the amplitude of the height being decayed
    angle (int): the angle (in degrees) at the current point in the decay (range: 90 to 0)
    side (str): whether its on the right or left side of the initial point. Used for recursion.
    width (int): the width of each bar. Used for scaling the decay.
    """
    if angle <= 0 or group < 0 or group >= len(heights):
        return
    heights[group] += amp * damping(angle)
    if side == "left" or side == "middle":
        add_height(heights, group - 1, amp, angle - width * math.log10(group + 1), "left", width, damping)
    if side == "right" or side == "middle":
        add_height(heights, group + 1, amp, angle - width * math.log10(group + 1), "right", width, damping)

def get_coords(x, y, angle, length):
    x_length = length * math.cos(angle)
    y_length = length * math.sin(angle)
    end_x = int(x + x_length)
    end_y = int(y + y_length)
    return end_x, end_y

def draw_ray(output_image, x, y, height, angle, config):
    output_image = cv2.line(output_image, get_coords(x,y, math.radians(angle / 3), 80), get_coords(x, y, math.radians(angle / 3), height), config["color"], 4)
    return output_image

def draw_circle(args):
    background, num_bars, heights, config = args
    background = cv2.circle(background, (config["size"][0] // 2, config["size"][1] // 2), 80, config["color"], -1)
    for angle in range(num_bars):
        background = draw_ray(background, config["size"][0] // 2, config["size"][1] // 2, heights[angle], angle, config)
    return background