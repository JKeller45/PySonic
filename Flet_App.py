import flet as ft
import cv2
import numpy as np
from PIL import ImageColor, Image
from Audio_Processing import render
from multiprocessing import freeze_support
from PIL import ImageColor
from Functions import hsv_to_rgb, rgb_to_hsv

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
        bg_color = ft.ElevatedButton("Select Background Color", on_click=color_picker)

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

    def color_picker(e):
        page.clean()
        picked_color = ft.canvas.Canvas([ft.canvas.Rect(0, 0, 320, 20, 0, ft.Paint(color="#ff0000"))])
        
        def change_hsv(e):
            rgb_tuple = hsv_to_rgb(hue.value, 1, 1)
            color = "#" + "".join([hex(int(rgb_tuple[0]))[2:].zfill(2), hex(int(rgb_tuple[1]))[2:].zfill(2), hex(int(rgb_tuple[2]))[2:].zfill(2)])
            primary.paint.gradient.colors[0] = ft.colors.with_opacity(1, color)
            primary.paint.gradient.colors[1] = ft.colors.with_opacity(0, color)

            picker.x = saturation.value / 100 * 255
            picker.y = (1 - brightness.value / 100) * 255

            rgb_tuple = hsv_to_rgb(hue.value, saturation.value / 100, brightness.value / 100)
            color = "#" + "".join([hex(int(rgb_tuple[0]))[2:].zfill(2), hex(int(rgb_tuple[1]))[2:].zfill(2), hex(int(rgb_tuple[2]))[2:].zfill(2)])
            picker.paint.gradient = ft.PaintRadialGradient((picker.x, picker.y), 5, colors=[ft.colors.with_opacity(1, color), ft.colors.with_opacity(1, "#000000")])
            picked_color._get_children()[0].paint.color = color
            hex_color.value = color

            page.update()

        def pan_start(e: ft.DragStartEvent):
            coords[0] = e.local_x
            coords[1] = e.local_y
            coords[0] = max(min(coords[0], 255), 0)
            coords[1] = max(min(coords[1], 255), 0)

            bright = (255 - coords[1]) / 255
            sat = coords[0] / 255
            rgb = hsv_to_rgb(hue.value, sat, bright)
            color = "#" + "".join([hex(int(rgb[0]))[2:].zfill(2), hex(int(rgb[1]))[2:].zfill(2), hex(int(rgb[2]))[2:].zfill(2)])
            saturation.value = sat * 100
            brightness.value = bright * 100

            picker.x = coords[0]
            picker.y = coords[1]
            picker.paint.gradient = ft.PaintRadialGradient((picker.x, picker.y), 5, colors=[ft.colors.with_opacity(1, color), ft.colors.with_opacity(1, "#000000")])
            picked_color._get_children()[0].paint.color = color
            hex_color.value = color
            page.update()

        def pan_update(e: ft.DragUpdateEvent):
            coords[0] = e.local_x
            coords[1] = e.local_y
            coords[0] = max(min(coords[0], 255), 0)
            coords[1] = max(min(coords[1], 255), 0)

            bright = (255 - coords[1]) / 255
            sat = coords[0] / 255
            rgb = hsv_to_rgb(hue.value, sat, bright)
            color = "#" + "".join([hex(int(rgb[0]))[2:].zfill(2), hex(int(rgb[1]))[2:].zfill(2), hex(int(rgb[2]))[2:].zfill(2)])
            saturation.value = sat * 100
            brightness.value = bright * 100

            picker.x = coords[0]
            picker.y = coords[1]
            picker.paint.gradient = ft.PaintRadialGradient((picker.x, picker.y), 5, colors=[ft.colors.with_opacity(1, color), ft.colors.with_opacity(1, "#000000")])
            picked_color._get_children()[0].paint.color = color
            hex_color.value = color
            page.update()

        def hex_change(e):
            text = e.control.value
            if len(text) == 0 or len(text) > 7:
                return
            if text[0] == "#" and len(text) == 7:
                red = text[1:3]
                green = text[3:5]
                blue = text[5:7]
                try:
                    red = int(red, 16)
                    green = int(green, 16)
                    blue = int(blue, 16)
                except ValueError:
                    return
                hsv = rgb_to_hsv(red, green, blue)
                hue.value = hsv[0]
                saturation.value = hsv[1] * 100
                brightness.value = hsv[2] * 100

                picker.x = saturation.value / 100 * 255
                picker.y = (1 - brightness.value / 100) * 255

                rgb_tuple = hsv_to_rgb(hue.value, saturation.value / 100, brightness.value / 100)
                color = "#" + "".join([hex(int(rgb_tuple[0]))[2:].zfill(2), hex(int(rgb_tuple[1]))[2:].zfill(2), hex(int(rgb_tuple[2]))[2:].zfill(2)])
                picker.paint.gradient = ft.PaintRadialGradient((picker.x, picker.y), 5, colors=[ft.colors.with_opacity(1, color), ft.colors.with_opacity(1, "#000000")])
                picked_color._get_children()[0].paint.color = color

                rgb_tuple = hsv_to_rgb(hue.value, 1, 1)
                color = "#" + "".join([hex(int(rgb_tuple[0]))[2:].zfill(2), hex(int(rgb_tuple[1]))[2:].zfill(2), hex(int(rgb_tuple[2]))[2:].zfill(2)])
                primary.paint.gradient.colors[0] = ft.colors.with_opacity(1, color)
                primary.paint.gradient.colors[1] = ft.colors.with_opacity(0, color)

                page.update()


        page.add(ft.Text("Color Selection", size=25, weight=ft.FontWeight.BOLD))
        page.add(ft.Container(height=10))
        hue = ft.Slider(label="Hue", min=0, max=359, width=255, height=25, value=0, on_change=change_hsv)
        saturation = ft.Slider(label="Saturation", min=0, max=100, width=255, value=100, height=25, on_change=change_hsv)
        brightness = ft.Slider(label="Brightness", min=0, max=100, width=255, value=100, height=25, on_change=change_hsv)
        bg = ft.canvas.Rect(0, 0, 255, 255, 0, ft.Paint(color="#000000"))
        primary = ft.canvas.Rect(0, 0, 255, 255, 0, ft.Paint(gradient=ft.PaintLinearGradient((0, 0), (0, 255), colors=[ft.colors.with_opacity(1, "#ff0000"), ft.colors.with_opacity(0, "#ff0000")])))
        white = ft.canvas.Rect(0, 0, 255, 255, 0, ft.Paint(gradient=ft.PaintLinearGradient((0, 0), (255, 0), colors=[ft.colors.with_opacity(1, "#ffffff"), ft.colors.with_opacity(0, "#ffffff")])))
        black = ft.canvas.Rect(0, 0, 255, 255, 0, ft.Paint(gradient=ft.PaintLinearGradient((0, 0), (0, 255), colors=[ft.colors.with_opacity(0, "#000000"), ft.colors.with_opacity(1, "#000000")])))
        row = ft.Row([
            ft.Column([ft.Text("Hue:"), ft.Text("Saturation:"), ft.Text("Brightness:")], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            ft.Column([hue, saturation, brightness], alignment=ft.MainAxisAlignment.CENTER, spacing=5)
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=5)
        coords = [255, 0]
        picker = ft.canvas.Circle(255, 0, 5, ft.Paint(gradient=ft.PaintRadialGradient((255, 0), 5, colors=[ft.colors.with_opacity(1, picked_color._get_children()[0].paint.color), ft.colors.with_opacity(1, "#000000")])))
        canvas = ft.canvas.Canvas([bg, primary, white, black, picker], width=255, height=255, content=ft.GestureDetector(
            on_pan_start=pan_start,
            on_pan_update=pan_update,
            drag_interval=10,))
        hex_color = ft.TextField(label="Hex Color", width=150, height=50, on_change=hex_change)
        hex_color.value = picked_color._get_children()[0].paint.color
        access_widgets["bg_color"] = hex_color
        next_button = ft.ElevatedButton("Continue", on_click=continue_to_files)
        page.add(ft.Row([ft.Container(canvas, height=255, width=255), ft.Column([hex_color, row, ft.Container(picked_color, width=310, height=20), next_button])], alignment=ft.MainAxisAlignment.CENTER, spacing=20))
        

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
    ft.app(target=main, assets_dir="assets")