# PySonic Audio Visualizer

## Overview
This app is an audio react visualizer built entirely in Python.

The internet is full of online audio react visualizers, but they all place a watermark over your video. You could pay for and learn Adobe After Effects, but that takes time and isn't what most people are looking for. PySonic is designed to be quick and easy, while still generating high-quality videos, without that stupid watermark.

This app is designed to replace Sonic Candle, a very similar app I used for years. It was orphaned 5 years ago, and I've since decided to develop a new version. At the moment, I am more or less at feature parity with Sonic Candle. I also have some new additions, and there are a few other cool features in the works, including transparent background videos. A UI refresh is also planned.

## Features
- Reactive frequency band bars or waveform
- Solid color, static image, or video backgrounds
- Zoom and "snow" effects
- AI Super Sampling
- Lots of accepted audio and video formats with ffmpeg
- Fast rendering with multithreading
- Different bar placement locations
- Variable frame rate, resolution, and length

## More information
PySonic comes bundled with ffmpeg so it accepts almost all audio and video inputs. It will, however, always output mp4 files. Video backgrounds are, at the moment, limited to 5 seconds due to the high memory requirements. While there is an easy solution, I thought the drop in render speed was unacceptable. I am continuing to investigate an alternative solution. The AI supersampling can be slow. It is a cool feature, but due to quality and speed, I recommend using it if you wish to render at 4K but only have 16Gb of memory. Even with these settings, it is still possible to render at a native resolution.

## A note on operating systems
At the moment, I have only built an executable for Windows. Due to some idiosyncrasies in the compiling program, I can only compile versions for my OS. I am investigating whether virtual machines will work. For now, I see no reason why PySonic would not work with WineHQ, however, this is not supported officially.

## Credits
Sound icon created by Culmbio - Flaticon
https://www.flaticon.com/free-icons/sound
