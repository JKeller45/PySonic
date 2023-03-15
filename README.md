# PySonic Audio Visualizer

This app is a audio react visualizer built entirely in Python. It is fully functional, but is rapidly being developed and new features are being added. Currently, it runs slowly, but this will change once a specific Python package (Numba) is updated for 3.11. This should happen in the near future.

This app is designed to be a replacement for Sonic Candle, a very similar app that I used for years. It was orphaned 8 years ago, and I decided to develop an new version. At the moment, I am more or less at feature parity with Sonic Candle. I also have some new additions, and there are a few other cool features in the work, including transpartent background videos.

Features include:
- Multiple audio react types
- Solid color, static image, or video background
- AI Super Sampling
- Super Sampling Anti-Aliasing 
- In-memory compression
- CPU or GPU render backend

In its current state, PySonic only outputs as MP4 files. It accepts most common audio, video, and image inputs. There may be some mp3 or ogg files that are not compatible, but this an issue with either FFMPEG or Scipy, not PySonic.

For those wishing to overlay the react on their own video background, support for transparent videos doesn't exist at the moment, but is planned. You can either use your own video as the background, or render the audio react with a solid (green) background and then chromakey. The chromakey should be very good as there is no AA, bluring, or post processing unless you choose to use the AISS or SSAA options.

I would recommend using in-memory compression, especially at 4K or if you have 16gb of ram or less to be safe, as the hit to render times is minor. In my testing, the CPU render backend is generally faster, but this might change if you have other tasks running on the CPU, have a more power GPU, or have a lower core count CPU.
