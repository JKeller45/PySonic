import numpy as np
import math
import cv2
import sys, os
from Classes import Settings, Frame_Information
import numpy.typing as npt
from numba import njit
from multiprocessing import shared_memory

def find_by_relative_path(relative_path: str) -> str:
    """
    Finds the location of file when compiled

    Parameters
    ----------
    relative_path (str): the path in the file system before compilation

    Returns
    -------
    (str): the adjusted path after compilation
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

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

def draw_bars(background: Frame_Information, num_bars: int, heights: npt.ArrayLike, cummulative_avg_heights: tuple[float, float], settings: Settings) -> npt.ArrayLike:
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
        background = zoom_effect(background, cummulative_avg_heights[1])

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
        snow_matrix = generate_snowfall_matrix(cummulative_avg_heights[0], -45, settings)
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
    bins = np.linspace(math.log10(60), math.log10(20000), num_bars)
    for c,_ in enumerate(bins):
        if c == 0:
            continue
        for i, f in enumerate(freq):
            if f <= bins[c - 1]:
                continue
            if f > bins[c]:
                break
            add_height(heights, c, amp[i], 90, "middle", settings.width, lambda amp, angle, group: amp * math.sin(math.radians(angle)), num_bars, settings)
    return np.int64(heights)

def add_height(heights: npt.ArrayLike, group: int, amp: float, angle: float, side: str, width: int, damping: callable, num_bars: int, settings: Settings):
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
    ang = angle - width * (2 ** ((6/num_bars) * (group - num_bars / 4)))
    if side == "left" or side == "middle":
        add_height(heights, group - 1, damping(amp, angle, group), ang, "left", width, damping, num_bars,settings)
    if side == "right" or side == "middle":
        add_height(heights, group + 1, damping(amp, angle, group), ang, "right", width, damping, num_bars,settings)

def draw_wave(background: Frame_Information, num_bars: int, heights: npt.ArrayLike, cummulative_avg_heights: tuple[float, float], settings: Settings) -> npt.ArrayLike:
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

    if settings.zoom:
        background = zoom_effect(background, cummulative_avg_heights[1])

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
        snow_matrix = generate_snowfall_matrix(cummulative_avg_heights[0], -45, settings)
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

def generate_snowfall_matrix(cummulative_avg_heights: int, angle: int, settings: Settings) -> npt.ArrayLike:
    np.random.seed(settings.snow_seed)
    matrix = np.random.choice(settings.size[0] * settings.size[1] // 150, size=settings.size)
    x_shift = int(math.cos(math.radians(angle)) * cummulative_avg_heights)
    y_shift = int(math.sin(math.radians(angle)) * cummulative_avg_heights)
    matrix = np.roll(matrix, x_shift, 1)
    matrix = np.roll(matrix, y_shift, 0)
    return np.argwhere(matrix == 1)

@njit
def create_snowfall(img: npt.ArrayLike, snow_matrix: npt.ArrayLike, color: tuple[int, int, int]) -> npt.ArrayLike:
    for x in snow_matrix:
        img = circle(img, x, 4, color)
    return img

def zoom_effect(img: npt.ArrayLike, zoom_height: float, coord: tuple[int, int] = None) -> npt.ArrayLike:
    zoom_amt = 1 + zoom_height / 300 * .15
    h, w, _ = [zoom_amt * i for i in img.shape]
    if coord is None: 
        cx, cy = w/2, h/2
    else: 
        cx, cy = [zoom_amt*c for c in coord]
    img = cv2.resize(img, (0, 0), fx=zoom_amt, fy=zoom_amt)
    img = img[int(round(cy - h/zoom_amt * .5)) : int(round(cy + h/zoom_amt * .5)), int(round(cx - w/zoom_amt * .5)) : int(round(cx + w/zoom_amt * .5)), :]
    return img 

@njit
def circle(img: npt.ArrayLike, coord: tuple[int, int], radius: int, color: tuple[int, int, int]) -> npt.ArrayLike:
    for x in range(coord[0] - radius, coord[0] + radius):
        for y in range(coord[1] - radius, coord[1] + radius):
            if (x - coord[0]) ** 2 + (y - coord[1]) ** 2 < radius ** 2 and x < len(img[0]) and y < len(img) and x >= 0 and y >= 0:
                img[y][x] = color
    return img

def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """
    Converts HSV color values to RGB color values.
    :param h: Hue value (0-360)
    :param s: Saturation value (0-1)
    :param v: Value (brightness) value (0-1)
    :return: Tuple of RGB color values (0-255)
    """
    if s == 0:
        return int(v * 255), int(v * 255), int(v * 255)
    else:
        h /= 60
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        if i == 0:
            return int(v * 255), int(t * 255), int(p * 255)
        elif i == 1:
            return int(q * 255), int(v * 255), int(p * 255)
        elif i == 2:
            return int(p * 255), int(v * 255), int(t * 255)
        elif i == 3:
            return int(p * 255), int(q * 255), int(v * 255)
        elif i == 4:
            return int(t * 255), int(p * 255), int(v * 255)
        else:
            return int(v * 255), int(p * 255), int(q * 255)
        
def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """
    Converts RGB color values to HSV color values.
    :param r: Red value (0-255)
    :param g: Green value (0-255)
    :param b: Blue value (0-255)
    :return: Tuple of HSV color values (0-360, 0-1, 0-1)
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(r, g, b), min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        hue = 0
    elif cmax == r:
        hue = ((g - b) / delta) % 6
    elif cmax == g:
        hue = (b - r) / delta + 2
    else:
        hue = (r - g) / delta + 4
    hue = round(hue * 60)
    if hue < 0:
        hue += 360
    saturation = 0 if cmax == 0 else delta / cmax
    value = cmax
    return hue, saturation, value
