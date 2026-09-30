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

"""
TrueHDR sample: convert an SDR video to an HDR10 (HEVC Main10, BT.2020 + PQ)
video using the NVIDIA VFX SDK. See ``README.md`` for a walkthrough and the
full command-line reference.
"""

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import torch
from nvvfx import TrueHDR

from video_io_utils import (
    avframe_to_rgb8_tensor,
    codec_candidates_for_bit_depth,
    print_video_info,
    rgb10a2_tensor_to_avframe,
    stream_fps,
)

HEVC_MAX = 8192

# HDR10 signalling for HEVC Main10 output: BT.2020 primaries, ST.2084 (PQ)
# transfer, BT.2020 non-constant luminance matrix, limited range. Values are
# the AVColor* enum integers from libavutil/pixfmt.h; PyAV's CodecContext
# and VideoFrame color-metadata setters accept these as ints.
HDR10_COLOR_METADATA = {
    "color_range": 1,  # AVCOL_RANGE_MPEG (limited / TV)
    "colorspace": 9,  # AVCOL_SPC_BT2020_NCL
    "color_trc": 16,  # AVCOL_TRC_SMPTE2084 (PQ)
    "color_primaries": 9,  # AVCOL_PRI_BT2020
}


def apply_hdr10_metadata(target) -> None:
    """
    Apply HDR10 color signalling (BT.2020 primaries, ST.2084 / PQ transfer,
    BT.2020 non-constant luminance matrix, limited range) to a PyAV target,
    typically ``stream.codec_context`` or a ``VideoFrame`` instance.

    All four fields in :data:`HDR10_COLOR_METADATA` must be settable on the
    target; a partial application produces an output that HDR-aware players
    will misinterpret. If any field cannot be set, this function collects
    the individual failures and raises a single :class:`RuntimeError` naming
    the offending fields and their underlying setter errors.
    """
    errors: list[str] = []
    for field, value in HDR10_COLOR_METADATA.items():
        try:
            setattr(target, field, value)
        except (TypeError, ValueError, AttributeError) as exc:
            errors.append(f"{field}={value!r}: {type(exc).__name__}: {exc}")
    if errors:
        target_name = type(target).__name__
        raise RuntimeError(
            f"Failed to apply HDR10 metadata to {target_name}: " + "; ".join(errors)
        )


def parse_args():
    """Parse the command-line arguments for the TrueHDR sample."""
    parser = argparse.ArgumentParser(
        description="TrueHDR SDR-to-HDR10 conversion using NVIDIA VFX SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(Path(__file__).parent / "assets" / "Drift_RUN_Master_Custom.mp4"),
        help="Input SDR video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(Path(__file__).parent / "output" / "sample_truehdr.mp4"),
        help="Output HDR10 (HEVC Main10) video file",
    )
    parser.add_argument(
        "--contrast",
        type=int,
        default=100,
        help="Contrast tunable [0..200]",
    )
    parser.add_argument(
        "--saturation",
        type=int,
        default=100,
        help="Saturation tunable [0..200]",
    )
    parser.add_argument(
        "--middle-gray",
        type=int,
        default=50,
        help="Middle-gray reference [10..100]",
    )
    parser.add_argument(
        "--luminance",
        type=int,
        default=650,
        help="Target HDR display peak luminance in nits [400..2000]",
    )
    parser.add_argument(
        "--debanding-off",
        action="store_true",
        help="Skip the DL-Debander pass",
    )
    return parser.parse_args()


def main():
    """
    Run the TrueHDR sample end-to-end: parse arguments, open the input,
    load the TrueHDR effect, encode each frame to HDR10 (HEVC Main10 with
    BT.2020 / PQ / BT.2020-NCL signalling), and print processing stats.
    """
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Range-check tunables so bad values don't surface as a wheel-side ValueError.
    for flag, value, lo, hi in (
        ("--contrast", args.contrast, 0, 200),
        ("--saturation", args.saturation, 0, 200),
        ("--middle-gray", args.middle_gray, 10, 100),
        ("--luminance", args.luminance, 400, 2000),
    ):
        if not lo <= value <= hi:
            print(f"Error: {flag} must be in [{lo}, {hi}], got {value}")
            sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. An NVIDIA GPU and working CUDA driver are required."
        )
        sys.exit(1)

    gpu = 0
    torch.cuda.set_device(gpu)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    bitrate = 16_000_000

    print("=" * 60)
    print("TrueHDR (SDR to HDR10 conversion)")
    print("=" * 60)
    print(f"  Input:        {input_path}")
    print(f"  Output:       {output_path}")
    print(f"  Contrast:     {args.contrast}")
    print(f"  Saturation:   {args.saturation}")
    print(f"  Middle gray:  {args.middle_gray}")
    print(f"  Luminance:    {args.luminance} nits")
    print(f"  Debanding:    {'off' if args.debanding_off else 'on'}")
    print()

    input_container = None
    output_container = None
    thdr = None
    processed = 0
    start_time = time.time()
    try:
        input_container = av.open(str(input_path))
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = "AUTO"

        input_width = input_stream.codec_context.width
        input_height = input_stream.codec_context.height
        total_frames = input_stream.frames or 0
        fps = stream_fps(input_stream)

        print_video_info(
            "Video info",
            input_stream,
            frames=total_frames or None,
            format_label="Input format",
        )
        print()

        thdr = TrueHDR(
            contrast=args.contrast,
            saturation=args.saturation,
            middle_gray=args.middle_gray,
            luminance=args.luminance,
            debanding_off=1 if args.debanding_off else 0,
            device=gpu,
        )
        thdr.load()
        print(f"Model loaded: {thdr.is_loaded}")
        print()

        if input_height > HEVC_MAX or input_width > HEVC_MAX:
            raise Exception(
                f"Output resolution exceeds the HEVC maximum resolution of {HEVC_MAX}x{HEVC_MAX}"
            )

        frame_rate = Fraction(fps if fps else 30).limit_denominator(10000)

        # Pick video codec: prefer HW (hevc_nvenc), fall back to SW (libx265).
        # TrueHDR emits packed RGB10A2, so we always write Main10.
        codec_candidates = codec_candidates_for_bit_depth(10)
        video_stream = None
        probe_errors: list[str] = []
        for name, pix_fmt in codec_candidates:
            container = av.open(str(output_path), mode="w")
            try:
                stream = container.add_stream(name, rate=frame_rate)
                stream.width = input_width
                stream.height = input_height
                stream.pix_fmt = pix_fmt
                stream.bit_rate = bitrate
                apply_hdr10_metadata(stream.codec_context)
                stream.codec_context.open()
            except Exception as exc:
                probe_errors.append(f"{name}/{pix_fmt}: {type(exc).__name__}: {exc}")
                try:
                    container.close()
                except Exception as close_exc:
                    probe_errors.append(
                        f"{name}/{pix_fmt} close: {type(close_exc).__name__}: {close_exc}"
                    )
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
            details = "; ".join(probe_errors) if probe_errors else "no candidates"
            print(f"Error: no usable H.265 encoder (tried {tried}); {details}")
            sys.exit(1)

        print(f"Encoder: {codec_name} ({output_pix_fmt})")

        if total_frames:
            print(f"Processing {total_frames} frames...")
        else:
            print("Processing frames...")
        start_time = time.time()

        for frame in input_container.decode(input_stream):
            rgb_input = avframe_to_rgb8_tensor(frame, gpu)

            torch.cuda.nvtx.range_push("TrueHDR")
            output = thdr.run(rgb_input, stream_ptr=stream_ptr)
            rgb_output = torch.from_dlpack(output.image).clone()
            torch.cuda.nvtx.range_pop()

            out_frame = rgb10a2_tensor_to_avframe(rgb_output)
            apply_hdr10_metadata(out_frame)
            for packet in video_stream.encode(out_frame):
                output_container.mux(packet)
            processed += 1

        for packet in video_stream.encode(None):
            output_container.mux(packet)
    finally:
        # Release resources in reverse order of acquisition. Collect close()
        # failures so a partial failure doesn't stop the remaining releases,
        # then surface them: if an exception from the try-block is already
        # propagating, preserve it and just warn; otherwise raise the
        # collected errors so they aren't silently lost.
        cleanup_errors: list[str] = []
        for label, resource in (
            ("output_container", output_container),
            ("input_container", input_container),
            ("thdr", thdr),
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            joined = "; ".join(cleanup_errors)
            if sys.exc_info()[0] is not None:
                print(
                    f"Warning: additional errors during cleanup: {joined}",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(f"Errors during cleanup: {joined}")

    print()
    print_video_info(
        "Output video info",
        output_path,
        frames=processed,
        format_label="Output format",
    )

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
