---
sidebar_position: 8
---

# FFMPEG

> **Advanced command line toolkits for video editing and compression**

## What is FFmpeg?

FFmpeg is an open-source collection of tools and libraries that allows users to record, convert, and stream audio and
video in various formats. It has a command-line interface, which means it operates through a text-based command prompt,
offering a wide range of capabilities for processing multimedia content.

The genius of FFmpeg lies in its versatility. It supports a vast array of codecs and formats, making it a go-to solution
for format conversions, encoding, and even basic editing tasks like trimming and concatenating files. It's used by
software developers and content creators alike for tasks ranging from simple file conversions to complex streaming
solutions.

FFmpeg can be found on its official website: [FFmpeg Official Website](https://ffmpeg.org/)

## How to Install FFmpeg

### macOS

- The easiest way to install FFmpeg on macOS is through [Homebrew](https://brew.sh/), a package manager for macOS.
- Run `brew install ffmpeg`.

### Linux

- Ubuntu: `sudo apt update && sudo apt install ffmpeg`.
- Fedora: `sudo dnf install ffmpeg`.
- Arch Linux: `sudo pacman -S ffmpeg`.

Once installed, you can verify the installation by running `ffmpeg -version` in your command prompt or terminal. This
should display the version of FFmpeg installed on your system.

## Use FFmpeg

### Converting Video to `mp4` Format

To convert a video to `mp4` format, utilize the following command:

```shell
ffmpeg -i INPUT.mov -vcodec h264 -acodec aac OUTPUT.mp4
```

In case you want to convert all `MOV` files to `mp4` in a directory, you can use the following batch script:

```shell
FILES="*.MOV"
for f in $FILES
do
  FILENAME=$(echo "${f}" | awk -F . '{ print $1 }')
  echo "${FILENAME}.mp4"
  ffmpeg -i $f -vcodec h264 -acodec aac "${FILENAME}.mp4"
done
```

### Video Compression

Firstly, navigate (`cd`) to the directory containing the videos to compress. Then, create a new `compressed` directory
using `mkdir compressed`. Use the following script to generate compressed versions of each video into the new directory:

```shell
FILES="*.mp4"
for f in $FILES; do
  FILENAME=$(echo "$f" | awk -F . '{ print $1 }')
  echo "${FILENAME}.mp4"
  ffmpeg -i $f -vcodec h264 -crf 30 -vf scale=1280:720 "compressed/${FILENAME}.mp4"
done
```

Note: A higher `crf` value results in a higher compression effect and, consequently, lower video quality.

### Resizing Video

You can resize a video using the following command:

```shell
ffmpeg -i input.mp4 -vf scale=$w:$h output.mp4
```

### Cropping Video

The following command can be used to crop a video:

```shell
ffmpeg -i input.mp4 -filter:v "crop=w:h:x:y" output.mp4
```

Here, `w` and `h` represent the output video size, while `x` and `y` denote the top left corner of the cropping
rectangle.

### Trimming Video

To trim a video, use:

```shell
ffmpeg -i input.mp4 -ss 00:05:10 -to 00:15:30 -c:v copy -c:a copy output.mp4
```

Here, `-ss` is the start time and `-to` is the end time of the video.

### Removing Audio

To remove the audio from a video, use:

```shell
ffmpeg -i input.mp4 -c copy -an output.mp4
```

### Concatenating Videos

First, create a text file `video_concat.txt` to list the videos to be concatenated:

```text
file 'video1.mp4'
file 'video2.mp4'
file 'video3.mp4'
```

Then, concatenate all videos based on the order specified in the `video_concat.txt` file:

```shell
ffmpeg -f concat -i video_concat.txt -c copy concat.mp4
```

### Change Keyframe Interval

To change the keyframe interval (also known as GOP size) of a video, you can use the `-g` option with `ffmpeg`. The
following command will change the keyframe interval to every 30 frames:

```shell
ffmpeg -i input.mp4 -map v:0 -c:v libx264 -g 30 output.mp4
```

- `-map v:0` maps the first video stream from the input to the output. If the input has multiple video streams, and you
  want to process only the first one, this is necessary.
- `-g 30` sets the keyframe interval to 30 frames. This means a keyframe will be inserted every 30 frames.

A higher keyframe interval often corresponds to a **larger compression rate and smaller file size**, but also a higher
decoding time. 