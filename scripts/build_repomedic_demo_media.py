from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1600
HEIGHT = 896
FPS = 24
BG = "#08121B"
SURFACE = "#122431"
FG = "#E8F0F2"
ACCENT = "#33C3B8"
SUCCESS = "#87D68D"
MUTED = "#8DA6B3"


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/msyh.ttc")]
    elif bold:
        candidates = [Path("C:/Windows/Fonts/msyhbd.ttc"), Path("C:/Windows/Fonts/segoeuib.ttf")]
    else:
        candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/segoeui.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


TITLE = _font(58, bold=True)
HEADING = _font(34, bold=True)
BODY = _font(25)
MONO = _font(22, mono=True)
SMALL = _font(19)


def _rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=3)


def _flow_frame(row: dict[str, Any], progress: float = 1.0) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((90, 60), "RepoMedic 真实工具闭环", font=TITLE, fill=FG)
    draw.text((93, 135), "固定案例 · decimal-price · 可审计回放", font=BODY, fill=MUTED)
    labels = [
        ("用户问题", "字符串价格被拼接"),
        ("大模型", "选择工具"),
        ("搜索源码", "total_price"),
        ("读取文件", "pricing.py + test"),
        ("运行测试", "pytest returncode=1"),
        ("结构化报告", "Top-1 app/pricing.py"),
    ]
    left = 75
    top = 280
    box_w = 220
    gap = 35
    visible = max(1, math.ceil(progress * len(labels)))
    for index, (title, detail) in enumerate(labels[:visible]):
        x1 = left + index * (box_w + gap)
        x2 = x1 + box_w
        _rounded(draw, (x1, top, x2, top + 180), SURFACE, ACCENT)
        draw.text((x1 + 22, top + 32), title, font=HEADING, fill=FG)
        draw.multiline_text(
            (x1 + 22, top + 94), detail, font=SMALL, fill=SUCCESS, spacing=6
        )
        if index < min(visible - 1, len(labels) - 1):
            start = x2 + 7
            end = x2 + gap - 7
            y = top + 90
            draw.line((start, y, end, y), fill=ACCENT, width=5)
            draw.polygon([(end, y), (end - 12, y - 8), (end - 12, y + 8)], fill=ACCENT)
    report = row["report"]
    draw.text((90, 560), "真实运行结果", font=HEADING, fill=FG)
    stats = [
        f"状态  {report['status']}",
        f"Token  {report['token_usage']}",
        f"延迟  {report['elapsed_ms'] / 1000:.2f}s",
        f"工具尝试  {report['attempted_tool_calls']}",
    ]
    for index, text in enumerate(stats):
        x = 90 + index * 360
        draw.text((x, 635), text, font=BODY, fill=SUCCESS if index == 0 else FG)
    draw.text(
        (90, 800),
        "离线回放：数据来自此前真实百炼API运行，不冒充现场联网结果",
        font=SMALL,
        fill=MUTED,
    )
    return image


def _title_frame(alpha: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    y = int(330 + (1 - alpha) * 35)
    title_fill = tuple(int(int(FG[i : i + 2], 16) * alpha) for i in (1, 3, 5))
    accent_fill = tuple(int(int(ACCENT[i : i + 2], 16) * alpha) for i in (1, 3, 5))
    draw.text((120, y), "RepoMedic", font=_font(96, bold=True), fill=title_fill)
    draw.text(
        (125, y + 125),
        "真实模型 · 真实工具 · 真实验证",
        font=HEADING,
        fill=accent_fill,
    )
    return image


def build(log: Path, screenshot: Path, video: Path) -> None:
    payload = json.loads(log.read_text(encoding="utf-8"))
    row = payload.get("case", payload)
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    _flow_frame(row).save(screenshot)
    with imageio.get_writer(video, fps=FPS, codec="libx264", quality=8) as writer:
        total_frames = 12 * FPS
        for frame_index in range(total_frames):
            second = frame_index / FPS
            if second < 2.2:
                frame = _title_frame(min(1.0, second / 0.8))
            elif second < 3.0:
                transition = (second - 2.2) / 0.8
                incoming = _flow_frame(row, max(0.1, transition * 0.2))
                frame = Image.blend(_title_frame(1.0), incoming, transition)
            else:
                progress = min(1.0, (second - 3.0) / 5.5)
                frame = _flow_frame(row, progress)
            writer.append_data(np.asarray(frame))


def main() -> int:
    parser = argparse.ArgumentParser(description="生成RepoMedic离线演示截图和备用视频")
    parser.add_argument("log", type=Path)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    build(args.log, args.screenshot, args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
