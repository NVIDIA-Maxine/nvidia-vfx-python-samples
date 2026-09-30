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

from pathlib import Path

import av
import numpy as np
import torch

COLOR_METADATA_FIELDS = ("color_range", "colorspace", "color_trc", "color_primaries")


def is_10_bit_pix_fmt(pix_fmt: str | None) -> bool:
    if not pix_fmt:
        return False
    normalized = pix_fmt.lower()
    return "10" in normalized or normalized.startswith("p010")


def resolve_bit_depth(input_pix_fmt: str | None) -> int:
    return 10 if is_10_bit_pix_fmt(input_pix_fmt) else 8


def codec_candidates_for_bit_depth(bit_depth: int) -> tuple[tuple[str, str], ...]:
    if bit_depth == 10:
        return (
            ("hevc_nvenc", "p010le"),
            ("libx265", "yuv420p10le"),
        )
    return (
        ("hevc_nvenc", "yuv420p"),
        ("libx265", "yuv420p"),
    )


def stream_pix_fmt(stream) -> str | None:
    pix_fmt = stream.codec_context.pix_fmt
    if pix_fmt:
        return pix_fmt
    stream_format = getattr(stream, "format", None)
    return getattr(stream_format, "name", None)


def stream_fps(stream) -> float:
    return float(stream.average_rate) if stream.average_rate else 0.0


def copy_color_metadata(source, destination) -> None:
    """Copy standard color interpretation metadata between PyAV objects."""
    for field in COLOR_METADATA_FIELDS:
        value = getattr(source, field, None)
        if value is not None:
            setattr(destination, field, value)


def _stream_dimensions(stream) -> tuple[int, int]:
    width = stream.codec_context.width or stream.width
    height = stream.codec_context.height or stream.height
    return width, height


def print_video_info(
    title: str,
    source,
    *,
    frames: int | None = None,
    format_label: str = "Format",
) -> None:
    try:
        if isinstance(source, (str, Path)):
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                width, height = _stream_dimensions(stream)
                pix_fmt = stream_pix_fmt(stream)
                bit_depth = resolve_bit_depth(pix_fmt)
                fps = stream_fps(stream)
        else:
            stream = source
            width, height = _stream_dimensions(stream)
            pix_fmt = stream_pix_fmt(stream)
            bit_depth = resolve_bit_depth(pix_fmt)
            fps = stream_fps(stream)
    except Exception as exc:
        print(f"{title}: unavailable ({exc})")
        return

    print(f"{title}:")
    print(f"  Resolution: {width}x{height}")
    print(f"  {format_label}: {pix_fmt or 'unknown'}")
    print(f"  Bit depth:  {bit_depth}-bit")
    print(f"  Frames:     {frames}" if frames is not None else "  Frames:     unknown")
    print(f"  FPS:        {fps:.2f}" if fps else "  FPS:        unknown")


def avframe_to_rgb8_tensor(frame: av.VideoFrame, gpu: int) -> torch.Tensor:
    """Convert a decoded frame to normalized planar RGB8 input."""
    arr = frame.to_ndarray(format="rgb24")
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(
        f"cuda:{gpu}"
    )  # (H, W, 3) uint8
    tensor = tensor.permute(2, 0, 1).float() / 255.0  # (3, H, W) float32
    return tensor.contiguous()


def avframe_to_rgb10a2_tensor(frame: av.VideoFrame, gpu: int) -> torch.Tensor:
    """Convert a decoded frame directly to packed RGB10A2 via FFmpeg."""
    if not hasattr(torch, "uint32"):
        raise RuntimeError("10-bit RGB10A2 mode requires torch.uint32 support")

    packed_frame = frame.reformat(format="x2bgr10le")
    plane = packed_frame.planes[0]
    if plane.line_size % 4 != 0:
        raise RuntimeError(f"Unexpected x2bgr10le line size: {plane.line_size}")

    row_words = plane.line_size // 4
    words = np.frombuffer(
        plane,
        dtype="<u4",
        count=packed_frame.height * row_words,
    ).reshape(packed_frame.height, row_words)
    packed = np.ascontiguousarray(words[:, : packed_frame.width])
    packed |= np.uint32(3 << 30)
    return torch.from_numpy(packed).to(f"cuda:{gpu}").contiguous()


def avframe_to_vfx_tensor(
    frame: av.VideoFrame, gpu: int, bit_depth: int
) -> torch.Tensor:
    if bit_depth == 10:
        return avframe_to_rgb10a2_tensor(frame, gpu)
    return avframe_to_rgb8_tensor(frame, gpu)


def rgb8_tensor_to_avframe(output: torch.Tensor) -> av.VideoFrame:
    """Convert normalized planar RGB8 output to a PyAV frame."""
    if output.ndim != 3 or output.shape[0] != 3 or output.dtype != torch.float32:
        raise ValueError("RGB8 output must be a float32 tensor with shape (3, H, W)")

    frame_np = (
        (output.clamp(0.0, 1.0) * 255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return av.VideoFrame.from_ndarray(frame_np, format="rgb24")


def rgb10a2_tensor_to_avframe(output: torch.Tensor) -> av.VideoFrame:
    """Wrap packed RGB10A2 output in an FFmpeg x2bgr10le frame."""
    if output.ndim != 2 or output.dtype != torch.uint32:
        raise ValueError("RGB10A2 output must be a uint32 tensor with shape (H, W)")

    height, width = output.shape
    frame = av.VideoFrame(width=width, height=height, format="x2bgr10le")
    plane = frame.planes[0]
    if plane.line_size % 4 != 0:
        raise RuntimeError(f"Unexpected x2bgr10le line size: {plane.line_size}")

    row_words = plane.line_size // 4
    packed = np.ascontiguousarray(output.detach().cpu().numpy(), dtype="<u4")
    storage = np.zeros((height, row_words), dtype="<u4")
    storage[:, :width] = packed | np.uint32(3 << 30)
    plane.update(storage.tobytes())
    return frame


def vfx_tensor_to_avframe(output: torch.Tensor, bit_depth: int) -> av.VideoFrame:
    if bit_depth == 10:
        return rgb10a2_tensor_to_avframe(output)
    if bit_depth == 8:
        return rgb8_tensor_to_avframe(output)
    raise ValueError(f"Unsupported bit depth: {bit_depth}; expected 8 or 10")


def unpack_rgb10a2(packed, *, normalize: bool = False):
    """
    Unpack (H, W) uint32 R10G10B10A2 words into per-channel planes.

    Accepts either a PyTorch tensor or a NumPy array; the returned type
    matches the input.

    Args:
        packed: (H, W) uint32 tensor / array of packed R10G10B10A2 words.
        normalize: If False (default), return (4, H, W) uint16 raw code
            points (R/G/B in [0, 1023], A in [0, 3]) for bit-exact
            inspection. If True, return (3, H, W) float32 in [0, 1] via
            divide-by-1023 (10-bit UNORM convention). Alpha is dropped
            in the normalized form.
    """
    if isinstance(packed, np.ndarray):
        if packed.ndim != 2:
            raise ValueError(
                f"expected a 2-D (H, W) array of packed R10G10B10A2 words, "
                f"got shape {packed.shape}"
            )
        if packed.dtype != np.uint32:
            raise TypeError(
                f"expected uint32 packed R10G10B10A2 words, got {packed.dtype}"
            )
        r = (packed & 0x3FF).astype(np.uint16)
        g = ((packed >> 10) & 0x3FF).astype(np.uint16)
        b = ((packed >> 20) & 0x3FF).astype(np.uint16)
        if normalize:
            return np.stack([c.astype(np.float32) / 1023.0 for c in (r, g, b)], axis=0)
        a = ((packed >> 30) & 0x3).astype(np.uint16)
        return np.stack([r, g, b, a], axis=0)

    if isinstance(packed, torch.Tensor):
        if packed.ndim != 2:
            raise ValueError(
                f"expected a 2-D (H, W) tensor of packed R10G10B10A2 words, "
                f"got shape {tuple(packed.shape)}"
            )
        if packed.dtype != torch.uint32:
            raise TypeError(
                f"expected uint32 packed R10G10B10A2 words, got {packed.dtype}"
            )
        # Torch bitwise ops don't support uint32; widen to int64.
        p = packed.to(torch.int64)
        r = (p & 0x3FF).to(torch.uint16)
        g = ((p >> 10) & 0x3FF).to(torch.uint16)
        b = ((p >> 20) & 0x3FF).to(torch.uint16)
        if normalize:
            return torch.stack([c.to(torch.float32) / 1023.0 for c in (r, g, b)], dim=0)
        a = ((p >> 30) & 0x3).to(torch.uint16)
        return torch.stack([r, g, b, a], dim=0)

    raise TypeError(
        f"expected a torch.Tensor or numpy.ndarray of packed R10G10B10A2 words, "
        f"got {type(packed).__name__}"
    )
