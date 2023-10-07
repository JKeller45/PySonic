import flet as ft
import cv2
import numpy as np
from PIL import ImageColor, Image
from Audio_Processing import render
from multiprocessing import freeze_support

def main(page: ft.Page):
    config = {}
    access_widgets = {}
    page.title = "PySonic"
    page.window_width = 800
    page.window_height = 500
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.add(ft.Text("Welcome to PySonic!", size=25, weight=ft.FontWeight.BOLD))
    page.add(ft.Text("Let's start rendering"))

    def continue_to_files(e):
        page.clean()
        config.clear()
        page.add(ft.Text("File Settings", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.Container(height=10))

        def on_audio_file_picker_result(result):
            if result.files is not None:
                config["FILE"] = result.files[0].path

        audio_file_picker = ft.FilePicker(on_result=on_audio_file_picker_result)
        page.overlay.append(audio_file_picker)

        def on_bg_file_picker_result(result):
            if result.files is not None:
                config["background"] = result.files[0].path

        background_picker = ft.FilePicker(on_result=on_bg_file_picker_result)
        page.overlay.append(background_picker)
        bg_color = ft.TextField(label="Input Hex Color", width=150, height=50)
        access_widgets["bg_color"] = bg_color

        def on_output_picker_result(result):
            if result.path is not None:
                config["output"] = result.path + "/"

        output_folder_picker = ft.FilePicker(on_result=on_output_picker_result)
        page.overlay.append(output_folder_picker)
        page.add(ft.Column([
            ft.ElevatedButton("Select Audio File", on_click=lambda _: audio_file_picker.pick_files(allow_multiple=False, file_type="CUSTOM", allowed_extensions=["mp3", "wav", "webm", "ogg", "aac", "flac", "aiff", "wma", "oga"])),
            ft.Row([
                ft.ElevatedButton("Select Background", on_click=lambda _: background_picker.pick_files(allow_multiple=False, file_type="CUSTOM", allowed_extensions = ["png", "jpg", "jpeg", "gif", "mp4", "mov", "wmv", "avi"])),
                ft.Text("OR", size=15),
                bg_color], alignment=ft.MainAxisAlignment.CENTER),
            ft.ElevatedButton("Select Output Folder", on_click=lambda _: output_folder_picker.get_directory_path())
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15))
        page.add(ft.Container(height=40))
        page.add(ft.ElevatedButton("Continue", on_click=continue_to_react))

    def continue_to_react(e):
        if config.get("FILE", None) == None or config.get("FILE", None) == "" or \
            config.get("output", None) == None or config.get("output", None) == "":
            return
        if config.get("background", None) == None or config.get("background", None) == "":
            if access_widgets["bg_color"].value == "":
                return
            config["background"] = access_widgets["bg_color"].value.strip(" #")[0:6]
        page.clean()
        page.add(ft.Text("React Settings", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.Container(height=10))
        
        react_type = ft.Dropdown(options=[ft.dropdown.Option("Bars"), ft.dropdown.Option("Waveform")], label="React Type", width=150, height=60)
        access_widgets["react_type"] = react_type

        page.add(react_type)
        page.add(ft.Container(height=15))
        page.add(ft.Row([ft.ElevatedButton("Back", on_click=continue_to_files), ft.ElevatedButton("Continue", on_click=continue_to_react_config)], alignment=ft.MainAxisAlignment.CENTER, spacing=20))

    def continue_to_react_config(e):
        page.clean()
        page.add(ft.Text("React Settings", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.Container(height=10))

        hex_color = ft.TextField(label="Bar Color (hex)", width=150, height=50)
        zoom_checkbox = ft.Checkbox(label="Zoom Effect", value=False, width=150, height=50)
        snowfall_checkbox = ft.Checkbox(label="Snowfall Effect", value=False, width=150, height=50)
        access_widgets["hex_color"] = hex_color
        access_widgets["zoom_checkbox"] = zoom_checkbox
        access_widgets["snowfall_checkbox"] = snowfall_checkbox

        if access_widgets["react_type"].value == "Bars":
            width = ft.TextField(label="Bar Width", width=150, height=60)
            separation = ft.TextField(label="Bar Separation", width=150, height=60)
            bar_pos = ft.Dropdown(options=[ft.dropdown.Option("Top"), ft.dropdown.Option("Bottom"), ft.dropdown.Option("Left"), ft.dropdown.Option("Right")], label="Bar Position", width=150, height=60)
            page.add(ft.Column([
                ft.Row([width, separation, bar_pos], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
                ft.Row([hex_color, zoom_checkbox, snowfall_checkbox], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20))
            access_widgets["width"] = width
            access_widgets["separation"] = separation
            access_widgets["bar_pos"] = bar_pos
        else:
            page.add(ft.Row([hex_color, zoom_checkbox, snowfall_checkbox], alignment=ft.MainAxisAlignment.CENTER, spacing=20))

        page.add(ft.Container(height=10))

        page.add(ft.Row([ft.ElevatedButton("Back", on_click=continue_to_react), ft.ElevatedButton("Continue", on_click=continue_to_video_settings)], alignment=ft.MainAxisAlignment.CENTER, spacing=20))

    def continue_to_video_settings(e):
        if access_widgets["react_type"].value == "Bars":
            config["width"] = int(access_widgets["width"].value)
            config["separation"] = int(access_widgets["separation"].value)
            config["position"] = access_widgets["bar_pos"].value
        else:
            config["width"] = 1
            config["separation"] = 0
            config["position"] = "Bottom"
        config["wave"] = access_widgets["react_type"].value == "Waveform"
        config["color"] = ImageColor.getrgb(f"#{access_widgets['hex_color'].value.strip(' #')[0:6]}")
        config["zoom"] = access_widgets["zoom_checkbox"].value
        config["snowfall"] = access_widgets["snowfall_checkbox"].value

        page.clean()

        page.add(ft.Text("Video Settings", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.Container(height=10))
        frame_rate = ft.TextField(label="Frame Rate", width=150, height=60)
        vid_length = ft.TextField(label="Video Length (seconds)", width=150, height=60)
        vid_res = ft.Dropdown(options=[ft.dropdown.Option("720p"), ft.dropdown.Option("1080p"), ft.dropdown.Option("1440p"), ft.dropdown.Option("4K")], label="Video Resolution", width=150, height=60)
        access_widgets["frame_rate"] = frame_rate
        access_widgets["vid_length"] = vid_length
        access_widgets["vid_res"] = vid_res

        circular_vid = ft.Checkbox(label="Circular Looped Video", value=False, width=150, height=50)
        AISS = ft.Checkbox(label="AI Supersampling", value=False, width=150, height=50)
        page.add(ft.Column([
            ft.Row([frame_rate, vid_length, vid_res], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            ft.Row([circular_vid, AISS], alignment=ft.MainAxisAlignment.CENTER, spacing=40)],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20))
        access_widgets["circular_vid"] = circular_vid
        access_widgets["AISS"] = AISS

        page.add(ft.Container(height=10))

        page.add(ft.Row([ft.ElevatedButton("Back", on_click=continue_to_react_config), ft.ElevatedButton("Render", on_click=continue_to_render)], alignment=ft.MainAxisAlignment.CENTER, spacing=20))

    def continue_to_render(e):
        config["frame_rate"] = int(access_widgets["frame_rate"].value)
        config["length"] = int(access_widgets["vid_length"].value)
        if access_widgets["vid_res"].value == "720p":
            config["size"] = [1280, 720]
        elif access_widgets["vid_res"].value == "1080p":
            config["size"] = [1920, 1080]
        elif access_widgets["vid_res"].value == "1440p":
            config["size"] = [2560, 1440]
        elif access_widgets["vid_res"].value == "4K":
            config["size"] = [3840, 2160]
        config["circular_looped_video"] = access_widgets["circular_vid"].value
        config["AISS"] = access_widgets["AISS"].value

        if len(config["background"]) == 6:
            config["background"] = cv2.cvtColor(np.array(Image.new(mode="RGB", size=(config["size"][0], config["size"][1]), color=ImageColor.getrgb(f"#{config['background']}"))), cv2.COLOR_RGB2BGR)

        page.clean()

        page.add(ft.Text("Render In Progress...", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.Container(height=10))
        progress_bar = ft.ProgressBar(width=500, height=25)
        progress_bar.value = 0
        page.add(progress_bar)

        render(config, progress_bar, page)

        page.clean()
        page.add(ft.Text("Render Complete!", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.ElevatedButton("Render Again", on_click=continue_to_files))

    page.add(ft.ElevatedButton("Continue", on_click=continue_to_files))

if __name__ == "__main__":
    freeze_support()
    ft.app(target=main)