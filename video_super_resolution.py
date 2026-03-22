# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from nvvfx import VideoSuperRes

HEVC_MAX = 8192

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Super Resolution using NVIDIA VFX SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(Path(__file__).parent / "assets" / "Drift_RUN_Master_Custom.mp4"),
        help="Input video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(Path(__file__).parent / "output" / "sample_sr.mp4"),
        help="Output video file",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Scale factor: 1 = same-resolution (denoise/deblur), 2/3/4 = upscale",
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=VideoSuperRes.QualityLevel.__members__.keys(),
        default="HIGH",
        help="Super resolution quality level",
    )
    return parser.parse_args()


def avframe_to_rgb_float(frame: av.VideoFrame, gpu: int) -> torch.Tensor:
    arr = frame.to_ndarray(format="rgb24")
    tensor = torch.from_numpy(arr).to(f"cuda:{gpu}")  # (H, W, 3) uint8
    tensor = tensor.permute(2, 0, 1).float() / 255.0  # (3, H, W) float32
    return tensor.contiguous()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality = VideoSuperRes.QualityLevel[args.quality]

    gpu = 0
    stream_ptr = torch.cuda.current_stream().cuda_stream
    bitrate = 16_000_000

    print("=" * 60)
    print("Video Super Resolution")
    print("=" * 60)
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    print(f"  Scale:   {args.scale}x")
    print(f"  Quality: {args.quality}")
    print()

    torch.cuda.set_device(gpu)

    input_container = av.open(str(input_path))
    input_stream = input_container.streams.video[0]
    input_stream.thread_type = "AUTO"

    input_width = input_stream.codec_context.width
    input_height = input_stream.codec_context.height
    total_frames = input_stream.frames or 0
    fps = float(input_stream.average_rate) if input_stream.average_rate else 0.0

    output_width = input_width * args.scale
    output_height = input_height * args.scale

    print("Video info:")
    print(f"  Resolution: {input_width}x{input_height} -> {output_width}x{output_height}")
    if total_frames:
        print(f"  Frames:     {total_frames}")
    print(f"  FPS:        {fps:.2f}")
    print()

    sr = VideoSuperRes(device=gpu, quality=quality)
    sr.input_width = input_width
    sr.input_height = input_height
    sr.output_width = output_width
    sr.output_height = output_height
    sr.load()
    print(f"Model loaded: {sr.is_loaded}")
    print()

    if output_height > HEVC_MAX or output_width > HEVC_MAX:
        raise Exception(f"Output resolution exceeds the HEVC maximum resolution of {HEVC_MAX}x{HEVC_MAX}")

    frame_rate = Fraction(fps if fps else 30).limit_denominator(10000)

    # Pick video codec: prefer HW (hevc_nvenc), fall back to SW (libx265).
    codec_candidates = ("hevc_nvenc", "libx265")
    output_container = None
    video_stream = None
    for name in codec_candidates:
        container = av.open(str(output_path), mode="w")
        try:
            stream = container.add_stream(name, rate=frame_rate)
            stream.width = output_width
            stream.height = output_height
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = bitrate
            stream.codec_context.open()
        except Exception:
            container.close()
            continue
        output_container, video_stream, codec_name = container, stream, name
        break

    if video_stream is None:
        print(f"Error: no usable H.265 encoder (tried {', '.join(codec_candidates)})")
        sys.exit(1)

    print(f"Encoder: {codec_name}")

    if total_frames:
        print(f"Processing {total_frames} frames...")
    else:
        print("Processing frames...")
    start_time = time.time()
    segment_start_time = start_time
    processed = 0
    for frame in input_container.decode(input_stream):
        rgb_input = avframe_to_rgb_float(frame, gpu)

        torch.cuda.nvtx.range_push("VideoSuperRes")
        output = sr.run(rgb_input, stream_ptr=stream_ptr)
        rgb_output = torch.from_dlpack(output.image).clone()
        torch.cuda.nvtx.range_pop()

        frame_np = (
            (rgb_output.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        )
        out_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in video_stream.encode(out_frame):
            output_container.mux(packet)
        processed += 1
        if processed % 100 == 0:
            ep_time = time.time() - segment_start_time
            print( f"Processed frame: {processed} current speed: {100/ep_time:.2f} fps")
            segment_start_time = time.time()

    for packet in video_stream.encode(None):
        output_container.mux(packet)
    output_container.close()
    input_container.close()

    elapsed = time.time() - start_time
    fps_proc = processed / elapsed if elapsed > 0 else 0

    print()
    print("Results:")
    print(f"  Frames processed: {processed}")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"  Processing FPS:   {fps_proc:.1f}")

    output_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Output size:      {output_size:.2f} MB")
    print(f"  Output file:      {output_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
