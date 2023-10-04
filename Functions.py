import pyfftw.interfaces.scipy_fft as fft
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
from PIL import Image as im
import sys, os
from io import BytesIO
from Classes import Settings, Frame_Information
import numpy.typing as npt
from numba import njit
from multiprocessing import shared_memory
from attrs import define, validators, field

def find_by_relative_path(relative_path: str) -> str:
    """
    Finds the location of the DNN models when compiled

    Parameters
    ----------
    relative_path (str): the path in the file system before compilation

    Returns
    -------
    (str): the adjusted path after compilation
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def plot_fft(freqs_side: npt.ArrayLike, FFT_side: npt.ArrayLike):
    """
    Plots the FFT with matplotlib

    Parameters
    ----------
    freqs_side (np.ndarray): the descrete frequency steps
    FFT_side (np.ndarray): the amplitudes at each frequency step
    """
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.xscale("log")
    plt.xlim((0,25000))
    plt.show()

def calc_fft(args: tuple[int, int, int, npt.ArrayLike]):
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
    FFT = abs(fft.fft(signal))
    FFT_side = FFT[:len(FFT) // 2]
    freqs = fft.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[:len(FFT) // 2]
    return freqs_side, FFT_side

def draw_rect(output_image: npt.ArrayLike, xcoord: int, ycoord: int, settings: Settings, height: int) -> npt.ArrayLike:
    """
    Draws the bar of a given size for a given frame.

    Parameters
    ----------
    output_image (np.ndarray): the background image being drawn over
    xcoord (int): the x-coord being drawn at
    ycoord (int): the y-coordinate being drawn at
    settings (Settings): the settings dataclass that contains the render settings
    height (int): the height/length of each rectangle being drawn

    Returns
    -------
    output_image (np.ndarray): the image with the rectangle drawn over it
    """

    if settings.position == "Right":
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord - height, ycoord + settings.width), settings.color, -1)
    elif settings.position == "Left":
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord + height, ycoord + settings.width), settings.color, -1)
    elif settings.position == "Top":
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord + settings.width, ycoord + height), settings.color, -1)
    elif settings.position == "Bottom":
        output_image = cv2.rectangle(output_image, (xcoord, ycoord), (xcoord + settings.width, ycoord - height), settings.color, -1)
    return output_image

def draw_bars(background: Frame_Information, num_bars: int, heights: npt.ArrayLike, avg_heights: float, settings: Settings) -> npt.ArrayLike:
    """
    Draws the bars for a given frame. This method is designed to be used in a multithreaded way.

    Parameters
    ----------
    background (np.ndarray): the background image being drawn over. Should be a deepcopy.
    num_bars (int): the number of bars being drawn on the frame
    heights (list): heights for each bar being drawn
    settings (Settings): the settings dataclass that contains the render settings

    Returns
    -------
    background (np.ndarray): the final frame with all bars drawn over its
    """
    existing_shm = shared_memory.SharedMemory(name=str(background.shared_name))
    shared_bg = np.ndarray(background.shared_memory_size, dtype=np.uint8, buffer=existing_shm.buf)
    if background.video:
        shared_bg = shared_bg.reshape((shared_bg.shape[0] // (settings.size[1] * settings.size[0] * 3), settings.size[1], settings.size[0], 3))
    else:
        shared_bg = shared_bg.reshape((settings.size[1], settings.size[0], 3))
    new_bg = np.zeros((settings.size[1], settings.size[0], 3), dtype=np.uint8)
    if background.video:
        new_bg[:] = shared_bg[background.frame_number][:]
    else:
        new_bg[:] = shared_bg[:]
    background = new_bg

    offset = 0
    if settings.zoom:
        background = zoom_effect(background, avg_heights[1], settings)

    for i in range(num_bars):
        if settings.position == "Right":
            draw_rect(background, settings.size[0] - 1, offset, settings, heights[i] + 1)
        elif settings.position == "Top":
            draw_rect(background, offset, 0, settings, heights[i] + 1)
        elif settings.position == "Left":
            draw_rect(background, 0, offset, settings, heights[i] + 1)
        elif settings.position == "Bottom":
            draw_rect(background, offset, settings.size[1] - 1, settings, heights[i] + 1)
        offset += (settings.width + settings.separation)

    if settings.snowfall:
        snow_matrix = generate_snowfall_matrix(avg_heights[0], -45, settings)
        background = create_snowfall(background, snow_matrix, settings.color)
    return background

def bins(freq: npt.ArrayLike, amp: npt.ArrayLike, heights: npt.ArrayLike, num_bars: int, settings: Settings) -> npt.ArrayLike:
    """
    Computes the heights of each bar from the FFT

    Parameters
    ----------
    freq (npt.ArrayLike): the frequency bins for the FFT
    amp (npt.ArrayLike): the total amplitudes at each frequency bin
    heights (npt.ArrayLike): the list where heights will be added
    num_bars (int): the number of bars in each frame
    settings (Settings): the settings dataclass that contains the render settings

    Returns
    -------
    heights (npt.ArrayLike): the final heights of each bar in the frame
    """
    bins = np.linspace(math.log10(60), math.log10(22000), num_bars)
    for c,_ in enumerate(bins):
        if c == 0:
            continue
        for i, f in enumerate(freq):
            if f <= bins[c - 1]:
                continue
            if f > bins[c]:
                break
            add_height(heights, c, amp[i], 90, "middle", settings.width, lambda amp, angle: amp * math.sin(math.radians(angle)), settings)
    heights = np.asarray(heights) / 200_000
    heights = heights * (settings.frame_rate // 30)
    heights = heights * (settings.size[1] / 1080)
    return np.int64(heights)

def add_height(heights: npt.ArrayLike, group: int, amp: float, angle: int, side: str, width: int, damping: callable, settings: Settings):
    """
    Uses a decay function to add height to adjacent bars to give a more natural, non blocky, look to each frame.

    Parameters
    ----------
    heights (list): the heights of each bar
    group (int): the index of heights that is being added to
    amp (int): the amplitude of the height being decayed
    angle (int): the angle (in degrees) at the current point in the decay (range: 90 to 0)
    side (str): whether its on the right or left side of the initial point. Used for recursion.
    width (int): the width of each bar. Used for scaling the decay.
    damping (function): the damping function to be used
    settings (Settings): the settings dataclass that contains the render settings
    """
    if angle <= 0 or group < 0 or group >= len(heights) or amp <= 0:
        return
    heights[group] += amp
    ang = angle - width * math.log10(group + 1)
    if side == "left" or side == "middle":
        add_height(heights, group - 1, damping(amp, angle), ang, "left", width, damping, settings)
    if side == "right" or side == "middle":
        add_height(heights, group + 1, damping(amp, angle), ang, "right", width, damping, settings)

def draw_wave(background: Frame_Information, num_bars: int, heights: npt.ArrayLike, avg_heights: float, settings: Settings) -> npt.ArrayLike:
    existing_shm = shared_memory.SharedMemory(name=str(background.shared_name))
    shared_bg = np.ndarray(background.shared_memory_size, dtype=np.uint8, buffer=existing_shm.buf)
    if background.video:
        shared_bg = shared_bg.reshape((shared_bg.shape[0] // (settings.size[1] * settings.size[0] * 3), settings.size[1], settings.size[0], 3))
    else:
        shared_bg = shared_bg.reshape((settings.size[1], settings.size[0], 3))
    new_bg = np.zeros((settings.size[1], settings.size[0], 3), dtype=np.uint8)
    if background.video:
        new_bg[:] = shared_bg[background.frame_number][:]
    else:
        new_bg[:] = shared_bg[:]
    background = new_bg

    offset = 0
    if settings.position == "Right":
        last_coord = (int((settings.size[0] - 1) - (heights[1] + 1)), int(offset))
    elif settings.position == "Top":
        last_coord = (int(offset + (heights[1] + 1)), 0)
    elif settings.position == "Left":
        last_coord = (0, int(offset + (heights[1] + 1)))
    elif settings.position == "Bottom":
        last_coord = (offset, int((settings.size[1] - 1) - (heights[1] + 1)))

    for i in range(num_bars):
        if settings.position == "Right":
            last_coord = draw_wave_segment(background, settings.size[0] - 1, offset, settings, heights[i] + 1, last_coord)
        elif settings.position == "Top":
            last_coord = draw_wave_segment(background, offset, 0, settings, heights[i] + 1, last_coord)
        elif settings.position == "Left":
            last_coord = draw_wave_segment(background, 0, offset, settings, heights[i] + 1, last_coord)
        elif settings.position == "Bottom":
            last_coord = draw_wave_segment(background, offset, settings.size[1] - 1, settings, heights[i] + 1, last_coord)
        offset += (settings.width + settings.separation)

    if settings.snowfall:
        snow_matrix = generate_snowfall_matrix(avg_heights[0], -45, settings)
        background = create_snowfall(background, snow_matrix, settings.color)

    return background

def draw_wave_segment(output_image: npt.ArrayLike, xcoord: int, ycoord: int, settings: Settings, height: int, last_coord: tuple[int, int]) -> tuple[int, int]:
    if settings.position == "Right":
        output_image = cv2.line(output_image, last_coord, (xcoord - height, ycoord + settings.width), settings.color, 2)
        last_coord = (xcoord - height, ycoord + settings.width)
    elif settings.position == "Left":
        output_image = cv2.line(output_image, last_coord, (xcoord + height, ycoord + settings.width), settings.color, 2)
        last_coord = (xcoord + height, ycoord + settings.width)
    elif settings.position == "Top":
        output_image = cv2.line(output_image, last_coord, (xcoord + settings.width, ycoord + height), settings.color, 2)
        last_coord = (xcoord + settings.width, ycoord + height)
    elif settings.position == "Bottom":
        output_image = cv2.line(output_image, last_coord, (xcoord + settings.width, ycoord - height), settings.color, 2)
        last_coord = (xcoord + settings.width, ycoord - height)
    return last_coord

def upsampling(frame: npt.ArrayLike, sr, settings: Settings) -> npt.ArrayLike:
    sr.upsample(frame)
    if settings.SSAA:
        frame = np.array(im.fromarray(frame).resize((len(frame[0]) // 2, len(frame) // 2), resample=im.ANTIALIAS))
    return frame

def compress(img: npt.ArrayLike) -> BytesIO:
    buffer = BytesIO()
    img = im.fromarray(img)
    img.save(buffer, "JPEG", quality=95)
    return buffer

def generate_snowfall_matrix(avg_heights: int, angle: int, settings: Settings) -> npt.ArrayLike:
    np.random.seed(settings.effect_settings.seed)
    matrix = np.random.choice(settings.size[0] * settings.size[1] // 150, size=settings.size)
    x_shift = int(math.cos(math.radians(angle)) * avg_heights / 40)
    y_shift = int(math.sin(math.radians(angle)) * avg_heights / 40)
    matrix = np.roll(matrix, x_shift, 1)
    matrix = np.roll(matrix, y_shift, 0)
    return np.argwhere(matrix == 1)

@njit
def create_snowfall(img: npt.ArrayLike, snow_matrix: npt.ArrayLike, color: tuple[int, int, int]) -> npt.ArrayLike:
    for x in snow_matrix:
        img = circle(img, x, 3, color)
    return img

@njit
def zoom_effect(img: npt.ArrayLike, zoom_height):
    zoom_amt = 1 + zoom_height / 300 * .15
    img = zoom(img, zoom_amt)
    return img

def zoom(img: npt.ArrayLike, zoom: float, coord=None) -> npt.ArrayLike:
    h, w, _ = [zoom * i for i in img.shape]
    if coord is None: 
        cx, cy = w/2, h/2
    else: 
        cx, cy = [zoom*c for c in coord]
    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    img = img[int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)), int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)), :]
    return img

def calc_difference(base: npt.ArrayLike, img: npt.ArrayLike) -> npt.ArrayLike:
    diff = np.subtract(img, base)
    kv = []
    index = np.argwhere(diff != 0)
    for i in index:
        kv.append((i, diff[i[0]][i[1]]))
    return np.asarray(kv)

def reconstruct(base: npt.ArrayLike, diff: npt.ArrayLike) -> npt.ArrayLike:
    frame = np.copy(base)
    for x in diff:
        frame[x[0][0]][x[0][1]] += x[1]
    return frame

@njit
def circle(img: npt.ArrayLike, coord: tuple[int, int], radius: int, color: tuple[int, int, int]) -> npt.ArrayLike:
    for x in range(coord[0] - radius, coord[0] + radius):
        for y in range(coord[1] - radius, coord[1] + radius):
            if (x - coord[0]) ** 2 + (y - coord[1]) ** 2 < radius ** 2 and x < len(img[0]) and y < len(img) and x >= 0 and y >= 0:
                img[y][x] = color
    return img