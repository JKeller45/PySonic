---
layout: default
---

# Introducing PySonic

**The internet is full of online audio react visualizers, but they all place a  watermark over your video. You could pay for and learn Adobe After Effects, but that takes time and isn't what most people are looking for. PySonic is designed to be quick and easy, while still generating high-quality videos, without that stupid watermark.**

## About

PySonic is an easy-to-use Python application that creates reactive video from an audio input. With an efficient render pipeline with multithreading, it is capable of rendering high-quality video very quickly. 

Some features include:

- Reactive frequency band bars or waveform
- Solid color, static image, or video backgrounds
- Zoom and "snow" effects
- AI Super Sampling
- Lots of accepted audio and video formats with FFmpeg
- Fast rendering with multithreading
- Different bar placement locations
- Variable frame rate, resolution, and length

PySonic was created as a replacement for Sonic Candle, a similar application built in Java that was orphaned years ago. The purpose was to improve on some issues with Sonic Candle, like lack of support for a wide range of audio and video formats, while bringing it up to date with a faster render pipeline.

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

This is not meant to be a definitive roadmap with all planned features. It is intended to be a guide for approximately when specific upcoming features should be available. It is subject to change.

#### 1.0 - Released!

- MacOS and Linux versions
- Continuous integration for Windows, macOS, and Linux
- Automated stability testing

#### 1.1 - Under Development

- Transparent video backgrounds
- Reintroduction of color picker (removed during the switch to Flet)
- Additional waveform react placements
- Render previews

#### 1.2

- Longer video backgrounds
- Render queue
- Animated GIF backgrounds (only static GIFs are currently supported)

#### Future

- Real-time render display
- Opacity for react bars and waveforms
- Additional react modes
- FFmpeg improvements
- And more!

## Installation Guide:
PySonic relies on FFmpeg, however, due to licensing, you must install it yourself. On Windows, FFmpeg can be installed with the Chocolatey package manager, or with a static build. On Mac, I highly recommend using the MacPorts or Homebrew package managers to install. On Linux, apt-get/apt is the preferred way. Check out these install guides: [Windows Static Build](https://phoenixnap.com/kb/ffmpeg-windows) or [Chocolatety](https://community.chocolatey.org/packages/ffmpeg-shared), [Mac](https://phoenixnap.com/kb/ffmpeg-mac), [Linux](https://www.hostinger.com/tutorials/how-to-install-ffmpeg). 

Once FFmpeg is installed, simply run the executable and start rendering!

## More Information
Video backgrounds are, at the moment, limited to 5 seconds due to the high memory requirements. While there is an easy solution, the drop in render speed was unacceptable. I am continuing to investigate an alternative solution. The AI supersampling can be slow. It is a cool feature, but due to quality and speed, I recommend using it if you wish to render at 4K but only have 16Gb of memory. Even with these settings, it is still possible to render at a native resolution.
