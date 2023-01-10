from Audio_Processing import render
import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import pygubu
from tkinter.colorchooser import askcolor
from PIL import ImageColor

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "ui.ui"

config = {}

class Application:
    def __init__(self, master=None):
        callbacks = {
                'run': run,
                'set_aa': set_aa,
                'pick_color': pick_color,
                'set_ss': set_ss
        }
        
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        self.mainwindow = builder.get_object('PySonic', master)

        self.aa = False
        self.ss = False

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

        builder.connect_callbacks(callbacks)

    def run(self):
        self.mainwindow.mainloop()

def set_aa():
    """
    Toggles the Anti-aliasing setting variable
    """
    app.aa = not app.aa

def set_ss():
    """
    Toggles the Super Sampling setting variable
    """
    app.ss = not app.ss

def pick_color():
    """
    Uses the Tkinter color chooser to select a color value and convert to RGB
    """
    global config
    colors = askcolor(title="Color Chooser")
    if colors != None:
        config["color"] = list(ImageColor.getrgb(colors[1]))

def run():
    """
    Runs the application. Sets all settings in the config dictionary and calls the render function
    """
    global config
    config["FILE"] = app.audio_path.cget('path')
    config["length"] = int(app.vid_length.get())
    size = app.res.get()

    if size == "720p":
        config["size"] = [1280, 720]
    elif size == "1080p":
        config["size"] = [1920, 1080]
    elif size == "1440p":
        config["size"] = [2560, 1440]
    else:
        config["size"] = [3840, 2160]
    
    config["background"] = app.bg_path.cget('path')
    config["frame_rate"] = int(app.fps.get())
    config["width"] = int(app.width.get())
    config["separation"] = int(app.sep.get())
    config["SSAA"] = app.aa
    config["AISS"] = app.ss
    pos = app.pos.get()
    if pos == "Top":
        config["horizontal_bars"] = False
        config["inverted_bars"] = True
    elif pos == "Bottom":
        config["horizontal_bars"] = False
        config["inverted_bars"] = False
    elif pos == "Left":
        config["horizontal_bars"] = True
        config["inverted_bars"] = False
    elif pos == "Right":
        config["horizontal_bars"] = True
        config["inverted_bars"] = True
    config["interpolation"] = False

    render(config)

if __name__ == '__main__':
    app = Application()
    app.run()