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
import warnings
from fractions import Fraction
from pathlib import Path

import av
import torch

from nvvfx import VideoFrameGeneration
from video_io_utils import (
    avframe_to_vfx_tensor,
    codec_candidates_for_bit_depth,
    copy_color_metadata,
    print_video_info,
    resolve_bit_depth,
    stream_fps,
    stream_pix_fmt,
    vfx_tensor_to_avframe,
)

HEVC_MAX = 8192


def parse_timesteps(value: str) -> list[float]:
    try:
        timesteps = [float(timestep.strip()) for timestep in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Timesteps must be comma-separated numbers"
        ) from exc

    if not timesteps or any(not 0.0 < timestep < 1.0 for timestep in timesteps):
        raise argparse.ArgumentTypeError(
            "Timesteps must be in the open interval (0.0, 1.0)"
        )
    if any(
        current >= following for current, following in zip(timesteps, timesteps[1:])
    ):
        warnings.warn(
            "Timesteps are not strictly increasing; frames will be generated in the supplied order.",
            stacklevel=2,
        )
    return timesteps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Frame Generation using NVIDIA VFX SDK",
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
        default=str(Path(__file__).parent / "output" / "sample_vfg.mp4"),
        help="Output video file",
    )
    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="Frame-rate multiplier",
    )
    generation_group.add_argument(
        "--timesteps",
        type=parse_timesteps,
        help="Comma-separated frame positions between consecutive input frames",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Output frame rate (0 = input frame rate multiplied by generated frames)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VideoFrameGeneration.Mode.__members__.keys(),
        default="MEDIUM",
        help="Frame generation quality mode",
    )
    parser.add_argument(
        "--automatic-shot-change-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic shot-change detection",
    )
    args = parser.parse_args()

    if args.multiplier < 2:
        parser.error("--multiplier must be at least 2")
    if args.target_fps < 0:
        parser.error("--target-fps must be non-negative")
    return args


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = VideoFrameGeneration.Mode[args.mode]
    timesteps = args.timesteps

    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. An NVIDIA GPU and working CUDA driver are required."
        )
        sys.exit(1)

    gpu = 0
    stream_ptr = torch.cuda.current_stream().cuda_stream
    bitrate = 16_000_000

    print("=" * 60)
    print("Video Frame Generation")
    print("=" * 60)
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    if timesteps:
        print(f"  Timesteps: {', '.join(str(timestep) for timestep in timesteps)}")
    else:
        print(f"  Multiplier: {args.multiplier}x")
    print(f"  Mode:    {args.mode}")
    print(f"  Automatic shot-change detection: {args.automatic_shot_change_detection}")
    print()

    torch.cuda.set_device(gpu)

    input_container = av.open(str(input_path))
    input_stream = input_container.streams.video[0]
    input_stream.thread_type = "AUTO"
    input_pix_fmt = stream_pix_fmt(input_stream)
    bit_depth = resolve_bit_depth(input_pix_fmt)

    input_width = input_stream.codec_context.width
    input_height = input_stream.codec_context.height
    total_frames = input_stream.frames or 0
    fps = stream_fps(input_stream)

    frames_per_pair = len(timesteps) + 1 if timesteps else args.multiplier
    output_fps = (
        args.target_fps if args.target_fps else (fps if fps else 30) * frames_per_pair
    )

    print("Video info:")
    print(f"  Resolution: {input_width}x{input_height}")
    print(f"  Input format: {input_pix_fmt or 'unknown'}")
    print(f"  Bit depth:  {bit_depth}-bit")
    if total_frames:
        print(f"  Frames:     {total_frames}")
    print(f"  Input FPS:  {fps:.2f}")
    print(f"  Output FPS: {output_fps:.2f}")
    print()

    image_encoding = (
        VideoFrameGeneration.ImageEncoding.RGB10A2
        if bit_depth == 10
        else VideoFrameGeneration.ImageEncoding.RGB8
    )
    vfg = VideoFrameGeneration(
        mode=mode,
        automatic_shot_change_detection_enabled=args.automatic_shot_change_detection,
        device=gpu,
        image_encoding=image_encoding,
    )
    vfg.input_width = input_width
    vfg.input_height = input_height
    vfg.frame_multiplier = args.multiplier
    vfg.load()
    print(f"Model loaded: {vfg.is_loaded}")
    print()

    if input_height > HEVC_MAX or input_width > HEVC_MAX:
        raise Exception(
            f"Output resolution exceeds the HEVC maximum resolution of {HEVC_MAX}x{HEVC_MAX}"
        )

    frame_rate = Fraction(output_fps if output_fps else 30).limit_denominator(10000)

    # Pick video codec: prefer HW (hevc_nvenc), fall back to SW (libx265).
    codec_candidates = codec_candidates_for_bit_depth(bit_depth)
    output_container = None
    video_stream = None
    for name, pix_fmt in codec_candidates:
        container = av.open(str(output_path), mode="w")
        try:
            stream = container.add_stream(name, rate=frame_rate)
            stream.width = input_width
            stream.height = input_height
            stream.pix_fmt = pix_fmt
            stream.bit_rate = bitrate
            copy_color_metadata(input_stream.codec_context, stream.codec_context)
            stream.codec_context.open()
        except Exception:
            container.close()
            continue
        output_container, video_stream, codec_name, output_pix_fmt = (
            container,
            stream,
            name,
            pix_fmt,
        )
        break

    if video_stream is None:
        tried = ", ".join(f"{name}/{pix_fmt}" for name, pix_fmt in codec_candidates)
        print(f"Error: no usable H.265 encoder (tried {tried})")
        sys.exit(1)

    print(f"Encoder: {codec_name} ({output_pix_fmt})")

    if total_frames:
        print(f"Processing {total_frames} frames...")
    else:
        print("Processing frames...")
    start_time = time.time()

    processed = 0
    generated = 0
    previous_frame = None
    for frame in input_container.decode(input_stream):
        current_frame = avframe_to_vfx_tensor(frame, gpu, bit_depth)
        if bit_depth == 10 and len(current_frame.shape) != 2:
            raise RuntimeError(
                f"10-bit VFG input packing failed: expected (H, W), got {tuple(current_frame.shape)}"
            )

        if previous_frame is None:
            output_frames = (current_frame,)
        else:
            output_frames = []
            if timesteps:
                for timestep in timesteps:
                    torch.cuda.nvtx.range_push("VideoFrameGeneration")
                    output = vfg.run_at_timestep(
                        previous_frame,
                        current_frame,
                        timestep,
                        stream_ptr=stream_ptr,
                    )
                    output_frames.append(torch.from_dlpack(output.image).clone())
                    torch.cuda.nvtx.range_pop()
                    generated += 1
            else:
                for frame_index in range(1, args.multiplier):
                    torch.cuda.nvtx.range_push("VideoFrameGeneration")
                    output = vfg.run(
                        previous_frame,
                        current_frame,
                        frame_index=frame_index,
                        stream_ptr=stream_ptr,
                    )
                    output_frames.append(torch.from_dlpack(output.image).clone())
                    torch.cuda.nvtx.range_pop()
                    generated += 1
            output_frames.append(current_frame)

        for output_tensor in output_frames:
            out_frame = vfx_tensor_to_avframe(output_tensor, bit_depth)
            copy_color_metadata(input_stream.codec_context, out_frame)
            for packet in video_stream.encode(out_frame):
                output_container.mux(packet)
        previous_frame = current_frame
        processed += 1

    for packet in video_stream.encode(None):
        output_container.mux(packet)
    output_container.close()
    input_container.close()
    vfg.close()

    print()
    print_video_info(
        "Output video info",
        output_path,
        format_label="Output format",
    )

    elapsed = time.time() - start_time
    fps_proc = processed / elapsed if elapsed > 0 else 0

    print()
    print("Results:")
    print(f"  Frames processed: {processed}")
    print(f"  Frames generated: {generated}")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"  Processing FPS:   {fps_proc:.1f}")

    output_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Output size:      {output_size:.2f} MB")
    print(f"  Output file:      {output_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
