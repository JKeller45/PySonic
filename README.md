# PySonic Audio Visualizer

This app is a audio react visualizer built entirely in Python.

This app is designed to be a replacement for Sonic Candle, a very similar app that I used for years. It was orphaned 8 years ago, and I've since decided to develop an new version. At the moment, I am more or less at feature parity with Sonic Candle. I also have some new additions, and there are a few other cool features in the works, including transpartent background videos. A UI refresh is also planned.

Features include:
- Multiple audio react types
- Solid color, static image, or video background
- AI Super Sampling

PySonic comes bundled with ffmpeg so it accepts almost all audio and video inputs. It will, however, always output mp4 files.

Video background are, at the moment, limited to 5 seconds due to the high memory requirements. While there is an easy solution, I thought the drop in render speed was unacceptable. I am continuing to investigate an alternative solution.

The AI super sampling can be slow. It is a cool feature, but due to quality and speed, I would only recommend using it if you wish to render at 4K but only have 16gb of memory. Even with these settings, it is still possible to render at a native resolution.

Sound icon created by Culmbio - Flaticon
https://www.flaticon.com/free-icons/sound