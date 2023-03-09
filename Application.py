from Audio_Processing import render
import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import pygubu
from tkinter.colorchooser import askcolor
from PIL import ImageColor, Image
import numpy as np
import cv2
from multiprocessing import freeze_support
from threading import Thread

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "ui.ui"

config = {}

class Application:
    def __init__(self, master=None):

        callbacks = {
                'run': run,
                'set_aa': set_aa,
                'pick_color': pick_color,
                'set_ss': set_ss,
                'pick_bg': pick_bg,
                'compression': set_compression
        }
        
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        self.mainwindow = builder.get_object('PySonic', master)

        self.aa = False
        self.ss = False
        self.bg_color = None
        self.compress = False

        self.audio_path = builder.get_object('audio_path')
        self.vid_length = builder.get_object('vid_length')
        self.res = builder.get_object('res')
        self.color = builder.get_object('hex')
        self.bg_path = builder.get_object('bg_path')
        self.fps = builder.get_object('fps')
        self.width = builder.get_object('bar_width')
        self.sep = builder.get_object('bar_sep')
        self.ssaa = builder.get_object('ssaa')
        self.pos = builder.get_object("pos")
        self.progress = builder.get_object("progress")
        self.bar_type = builder.get_object("bar_type")
        self.output_path = builder.get_object("output_path")
        self.backend = builder.get_object("backend")

        builder.connect_callbacks(callbacks)

    def run(self):
        self.mainwindow.mainloop()

def set_aa():
    """
    Toggles the Anti-aliasing setting variable
    """
    app.aa = not app.aa

def set_compression():
    """
    Toggles the in-memory compression setting variable
    """
    app.compress = not app.compress

def set_ss():
    """
    Toggles the Super Sampling setting variable
    """
    app.ss = not app.ss

def pick_color():
    """
    Uses the Tkinter color chooser to select a color value for the bar and convert to RGB
    """
    global config
    colors = askcolor(title="Color Chooser")
    if colors[1] != None:
        config["color"] = list(ImageColor.getrgb(colors[1]))
        config["color"].reverse()

def pick_bg():
    """
    Uses the Tkinter color chooser to select a color value for the background and convert to RGB
    """
    global config
    colors = askcolor(title="Color Chooser")
    if colors[1] != None:
        app.bg_color = ImageColor.getrgb(colors[1])

def run():
    """
    Runs the application. Sets all settings in the config dictionary and calls the render function
    """
    global config
    config["FILE"] = app.audio_path.cget('path')
    config["length"] = int(app.vid_length.get())
    config["output"] = app.output_path.cget('path')

    size = app.res.get()

    if size == "720p":
        config["size"] = [1280, 720]
    elif size == "1080p":
        config["size"] = [1920, 1080]
    elif size == "1440p":
        config["size"] = [2560, 1440]
    else:
        config["size"] = [3840, 2160]
    
    if app.bg_color != None:
        config["background"] = cv2.cvtColor(np.array(Image.new(mode="RGB", size=(config["size"][0], config["size"][1]), color=app.bg_color)), cv2.COLOR_RGB2BGR)
    else:
        config["background"] = app.bg_path.cget('path')

    config["frame_rate"] = int(app.fps.get())
    config["width"] = int(app.width.get())
    config["separation"] = int(app.sep.get())
    config["SSAA"] = app.aa
    config["AISS"] = app.ss

    pos = app.pos.get()

    config["position"] = pos
    
    bar_type = app.bar_type.get()
    if bar_type == "Solar":
        config["solar"] = True
        config["wave"] = False
    elif bar_type == "Wave":
        config["wave"] = True
        config["solar"] = False
    else:
        config["wave"] = False
        config["solar"] = False

    if app.backend.get() == "CPU+GPU":
        config["use_gpu"] = True
    else:
        config["use_gpu"] = False
        
    config["memory_compression"] = app.compress

    ret_val = list()
    thread = Thread(target=render, args=(config, app.progress, app.mainwindow, ret_val), daemon=True)
    thread.start()
    while ret_val == []:
        app.mainwindow.update()
    thread.join()

if __name__ == '__main__':
    freeze_support()
    app = Application()
    app.run()