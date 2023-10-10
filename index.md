---
layout: default
---

# Introducing PySonic

**The internet is full of online audio react visualizers, but they all place a  watermark over your video. You could pay for and learn Adobe After Effects, but that takes time and isn't what most people are looking for. PySonic is designed to be quick and easy, while still generating high quality videos, without that stupid watermark.**

## About

PySonic is an easy to use python application that create reactive video from an audio input. With an efficient render pipeline with multithreading, it is capable of rendering high quality video very quickly. 

Some features include:

- Reactive frequency band bars or waveform
- Solid color, static image, or video backgrounds
- Zoom and "snow" effects
- AI Super Sampling
- Lots of accepted audio and video formats with ffmpeg
- Fast rendering with multithreading
- Different bar placement locations
- Variable frame rate, resolution, and length

PySonic was created as a replacement for Sonic Candle, a similar application built in Java that was orphaned years ago. The purpose was to improve on some issues with Sonic Candle, like lack of support for a wide range of audio and video formats, while bringing it up to date with a faster render pipeline.

PySonic is currently in BETA. This is not a stable release. It has been tested, but thorough stability testing is necessary and new features are still planned. If you encounter bugs and glitches, please open an issue on github as it would greatly help me out in testing and polishing.

## Showcase

Below are two videos to showcase some of the capabilities of PySonic. All of these are strictly made in PySonic, with no external video software used.

Simply click and you'll be redirected to YouTube:

<div style="text-align: center;">

  <a href="http://www.youtube.com/watch?feature=player_embedded&v=DYgUEoXwa1Q
  " target="_blank"><img src="http://img.youtube.com/vi/DYgUEoXwa1Q/0.jpg" 
  alt="PySonic Showcase: Jeremy Zucker - Supercuts" width="427" height="240"/></a>

  <a href="http://www.youtube.com/watch?feature=player_embedded&v=39qQJ664yg8
  " target="_blank"><img src="http://img.youtube.com/vi/39qQJ664yg8/0.jpg" 
  alt="PySonic Showcase: Avicii - Levels" width="427" height="240"/></a>

</div>

## Roadmap

This is not meant to be a difinitive roadmap with all planned features exactly when they will release. It is intended to be a guide for approximately when specific upcoming features should be available. It is subject to change.

#### 0.9 - Released!

- New Flet (Flutter) based GUI
- Preparation for continuous integration
- Many bugfixes
- Improvements for future macOS and Linux releases

#### 1.0 - Under Development

- MacOS and Linux versions
- Continuous integration for Windows, macOS, and Linux
- Automated stability testing

#### 1.1

- Transparent video backgrounds
- Reintroduction of color picker (removed during the switch to Flet)
- Additional waveform react placements

#### 1.2

- Longer video backgrounds
- Render queue
- Render previews

#### 1.3

- Real-time render display

## More Information

PySonic comes bundled with ffmpeg so it accepts almost all audio and video inputs. It will, however, always output mp4 files. Video backgrounds are, at the moment, limited to 5 seconds due to the high memory requirements. While there is an easy solution, I thought the drop in render speed it came with was unacceptable. I am continuing to investigate an alternative solution. The AI supersampling can be slow. It is a cool feature, but due to quality and speed, I recommend using it if you wish to render at 4K but only have 16Gb of memory. Even with these settings, it is still possible to render at a native resolution.