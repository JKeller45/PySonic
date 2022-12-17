from Audio_Processing import render
import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import pygubu

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "ui.ui"

class Application:
    def __init__(self, master=None):
        callbacks = {
                'run': run,
                'set_aa': set_aa
        }
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        self.mainwindow = builder.get_object('PySonic', master)

        self.aa = False

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
    app.aa = not app.aa

def run():
    config = {}
    config["FILE"] = app.audio_path.cget('path')
    config["length"] = int(app.vid_length.get())
    size = app.res.get()
    if size == "720p":
        config["size"] = [1280, 720]
    elif size == "1080p":
        config["size"] = [1920, 1080]
    else:
        config["size"] = [3840, 2160]
    hex = app.color.get().lstrip('#')
    config["color"] = list(int(hex[i:i+2], 16) for i in (0, 2, 4))
    config["background"] = app.bg_path.cget('path')
    config["frame_rate"] = int(app.fps.get())
    config["width"] = int(app.width.get())
    config["separation"] = int(app.sep.get())
    config["SSAA"] = app.aa
    print(config["SSAA"])
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

    render(config)

if __name__ == '__main__':
    app = Application()
    #app.run()