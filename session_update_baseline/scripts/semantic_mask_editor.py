#!/usr/bin/env python3
"""Small semantic-mask correction GUI for the Azure Kinect keyframe runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import yaml


COMMON_LABELS = [
    (0, "wall"),
    (3, "floor"),
    (5, "ceiling"),
    (7, "bed"),
    (8, "window"),
    (10, "cabinet"),
    (12, "PERSON-DYNAMIC"),
    (14, "door"),
    (15, "table"),
    (19, "chair"),
    (23, "sofa"),
    (24, "shelf"),
    (33, "desk"),
    (35, "wardrobe"),
    (39, "cushion"),
    (41, "box"),
    (57, "pillow"),
    (62, "bookcase"),
    (74, "computer"),
    (75, "swivel-chair"),
    (92, "apparel"),
    (112, "basket"),
    (115, "bag"),
    (131, "blanket"),
    (138, "trashcan"),
    (139, "fan"),
    (149, "IGNORE-ERASE"),
]


def make_palette() -> np.ndarray:
    ids = np.arange(256, dtype=np.uint32)
    palette = np.stack(
        ((ids * 37 + 53) % 255, (ids * 67 + 97) % 255, (ids * 101 + 193) % 255),
        axis=1,
    ).astype(np.uint8)
    palette[0] = [110, 110, 110]
    palette[12] = [255, 55, 75]
    return palette


class SemanticMaskEditor:
    def __init__(
        self,
        root: tk.Tk,
        input_dir: Path,
        output_dir: Path,
        labelspace: Path,
        title: str,
    ) -> None:
        self.root = root
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames = sorted(input_dir.glob("*_color.png"))
        if not self.frames:
            raise RuntimeError(f"No *_color.png files in {input_dir}")
        self.palette = make_palette()
        self.palette[149] = [255, 0, 255]
        config = yaml.safe_load(labelspace.read_text(encoding="utf-8"))
        self.label_names = {
            int(row["label"]): str(row["name"]).strip() for row in config["label_names"]
        }
        self.label_names[149] = "IGNORE-ERASE"
        self.label_values = [f"{label:03d}  {name}" for label, name in COMMON_LABELS]
        self.index = 0
        self.rgb: Image.Image
        self.mask: np.ndarray
        self.auto_mask: np.ndarray
        self.photo: ImageTk.PhotoImage | None = None
        self.undo_stack: list[np.ndarray] = []
        self.polygon: list[tuple[int, int]] = []
        self.dirty = False
        self.brushing = False
        self.show_original = False
        self.review_path = output_dir / "review_status.json"
        self.reviewed: set[str] = set()
        self._load_review_status()

        root.title(f"Semantic Mask Editor - {title}")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui(title)
        self._bind_keys()
        self.load_frame(self.index)

    @property
    def frame_id(self) -> str:
        return self.frames[self.index].name[: -len("_color.png")]

    @property
    def output_mask_path(self) -> Path:
        return self.output_dir / f"{self.frame_id}_segmentation.png"

    def _load_review_status(self) -> None:
        if not self.review_path.exists():
            return
        try:
            data = json.loads(self.review_path.read_text(encoding="utf-8"))
            self.reviewed = set(data.get("reviewed", []))
            self.index = min(max(int(data.get("current_index", 0)), 0), len(self.frames) - 1)
        except (OSError, ValueError, TypeError):
            self.reviewed = set()

    def _write_review_status(self) -> None:
        payload = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "current_index": self.index,
            "reviewed": sorted(self.reviewed),
            "reviewed_count": len(self.reviewed),
            "frame_count": len(self.frames),
        }
        self.review_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_ui(self, title: str) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=title).grid(row=0, column=0, padx=(0, 12))
        ttk.Button(toolbar, text="1  PREV (A)", command=lambda: self.navigate(-1)).grid(row=0, column=1)
        ttk.Button(toolbar, text="2  NEXT (D)", command=lambda: self.navigate(1)).grid(row=0, column=2, padx=(4, 12))
        ttk.Button(toolbar, text="7  NEXT PERSON", command=self.next_person).grid(row=0, column=3, padx=(0, 12))
        ttk.Button(toolbar, text="3  SAVE (S)", command=self.save).grid(row=0, column=4)
        ttk.Button(toolbar, text="4  REVIEWED (R)", command=self.mark_reviewed).grid(row=0, column=5, padx=4)
        ttk.Button(toolbar, text="5  UNDO (Ctrl+Z)", command=self.undo).grid(row=0, column=6)
        ttk.Button(toolbar, text="6  RESET AUTO", command=self.reset_auto).grid(row=0, column=7, padx=4)

        tools = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        tools.pack(fill="x")
        self.tool = tk.StringVar(value="component")
        for column, (value, label) in enumerate(
            (("component", "A  COMPONENT"), ("brush", "B  BRUSH"), ("polygon", "C  POLYGON"))
        ):
            ttk.Radiobutton(tools, text=label, value=value, variable=self.tool).grid(
                row=0, column=column, padx=(0, 8)
            )
        ttk.Label(tools, text="CLASS").grid(row=0, column=3)
        self.label_combo = ttk.Combobox(tools, values=self.label_values, state="readonly", width=25)
        self.label_combo.current(6)
        self.label_combo.grid(row=0, column=4, padx=(4, 12))
        ttk.Button(tools, text="ERASE = IGNORE", command=self.select_ignore).grid(row=0, column=5, padx=(0, 12))
        ttk.Label(tools, text="BRUSH SIZE").grid(row=0, column=6)
        self.brush_size = tk.IntVar(value=24)
        ttk.Scale(tools, from_=2, to=100, variable=self.brush_size, orient="horizontal", length=120).grid(
            row=0, column=7, padx=(4, 12)
        )
        ttk.Label(tools, text="OVERLAY ALPHA").grid(row=0, column=8)
        self.alpha = tk.IntVar(value=48)
        ttk.Scale(
            tools,
            from_=0,
            to=85,
            variable=self.alpha,
            command=lambda _value: self.render(),
            orient="horizontal",
            length=140,
        ).grid(row=0, column=9, padx=4)

        content = ttk.Frame(self.root, padding=(6, 0, 6, 0))
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(content, width=960, height=540, cursor="crosshair", highlightthickness=0)
        self.canvas.pack(side="left")
        legend_panel = ttk.Frame(content, padding=(10, 0, 0, 0))
        legend_panel.pack(side="left", fill="y")
        ttk.Label(legend_panel, text="COLORS IN THIS FRAME").pack(anchor="w")
        self.legend = tk.Canvas(legend_panel, width=245, height=515, highlightthickness=0)
        self.legend.pack(fill="y")
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Double-Button-1>", self.commit_polygon)
        self.canvas.bind("<Button-3>", self.pick_label)
        self.canvas.bind("<Motion>", self.on_motion)

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, padding=6).pack(fill="x")
        ttk.Label(
            self.root,
            text="Left click: edit | Right click: pick class | Polygon: Enter to fill, Esc to cancel | Hold SPACE: original RGB",
            padding=(6, 0, 6, 6),
        ).pack(fill="x")

    def _bind_keys(self) -> None:
        self.root.bind("a", lambda _event: self.navigate(-1))
        self.root.bind("d", lambda _event: self.navigate(1))
        self.root.bind("s", lambda _event: self.save())
        self.root.bind("r", lambda _event: self.mark_reviewed())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Return>", self.commit_polygon)
        self.root.bind("<Escape>", lambda _event: self.cancel_polygon())
        self.root.bind("<KeyPress-space>", self.original_on)
        self.root.bind("<KeyRelease-space>", self.original_off)

    def selected_label(self) -> int:
        return int(self.label_combo.get().split()[0])

    def label_name(self, label: int) -> str:
        return self.label_names.get(label, f"class-{label}")

    def select_ignore(self) -> None:
        self.label_combo.current(len(COMMON_LABELS) - 1)

    def update_legend(self) -> None:
        self.legend.delete("all")
        labels, counts = np.unique(self.mask, return_counts=True)
        order = np.argsort(counts)[::-1]
        total = float(self.mask.size)
        for row_index, source_index in enumerate(order[:22]):
            label = int(labels[source_index])
            count = int(counts[source_index])
            color = self.palette[label]
            color_hex = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
            y = 8 + row_index * 22
            self.legend.create_rectangle(4, y, 24, y + 15, fill=color_hex, outline="#444444")
            self.legend.create_text(
                32,
                y + 8,
                anchor="w",
                text=f"{label:03d} {self.label_name(label)[:17]}  {100.0 * count / total:4.1f}%",
            )

    def next_person(self) -> None:
        start = self.index
        for offset in range(1, len(self.frames) + 1):
            candidate = (start + offset) % len(self.frames)
            frame_id = self.frames[candidate].name[: -len("_color.png")]
            edited = self.output_dir / f"{frame_id}_segmentation.png"
            source = edited if edited.exists() else self.input_dir / f"{frame_id}_segmentation.png"
            mask = np.asarray(Image.open(source), dtype=np.uint8)
            if np.any(mask == 12):
                if self.dirty:
                    self.save()
                self.load_frame(candidate)
                self._write_review_status()
                return
        messagebox.showinfo("Person frames", "No frame contains class 012 person.")

    def load_frame(self, index: int) -> None:
        self.index = index
        self.rgb = Image.open(self.frames[index]).convert("RGB")
        source_mask = self.input_dir / f"{self.frame_id}_segmentation.png"
        self.auto_mask = np.asarray(Image.open(source_mask), dtype=np.uint8).copy()
        if self.output_mask_path.exists():
            self.mask = np.asarray(Image.open(self.output_mask_path), dtype=np.uint8).copy()
        else:
            self.mask = self.auto_mask.copy()
        if self.rgb.size != (self.mask.shape[1], self.mask.shape[0]):
            raise RuntimeError(f"RGB/mask size mismatch for {self.frame_id}")
        self.canvas.config(width=self.rgb.width, height=self.rgb.height)
        self.undo_stack.clear()
        self.polygon.clear()
        self.dirty = False
        self.render()

    def render(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        if self.show_original:
            display = rgb
        else:
            semantic = self.palette[self.mask]
            alpha = float(self.alpha.get()) / 100.0
            display = np.clip(rgb * (1.0 - alpha) + semantic * alpha, 0, 255).astype(np.uint8)
        self.photo = ImageTk.PhotoImage(Image.fromarray(display))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        if self.polygon:
            flattened = [value for point in self.polygon for value in point]
            if len(self.polygon) > 1:
                self.canvas.create_line(*flattened, fill="#ffffff", width=2)
            for x, y in self.polygon:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ffffff", outline="#111111")
        self.update_legend()
        checked = "REVIEWED" if self.frame_id in self.reviewed else "NOT REVIEWED"
        dirty = " | UNSAVED" if self.dirty else ""
        self.status.set(
            f"{self.index + 1}/{len(self.frames)}  {self.frame_id}  [{checked}]  "
            f"reviewed {len(self.reviewed)}/{len(self.frames)}{dirty}"
        )

    def push_undo(self) -> None:
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.mask = self.undo_stack.pop()
        self.dirty = True
        self.render()

    def save(self) -> None:
        Image.fromarray(self.mask, mode="L").save(self.output_mask_path)
        self.dirty = False
        self._write_review_status()
        self.render()

    def mark_reviewed(self) -> None:
        self.save()
        self.reviewed.add(self.frame_id)
        self._write_review_status()
        self.render()

    def navigate(self, delta: int) -> None:
        if self.dirty:
            self.save()
        target = min(max(self.index + delta, 0), len(self.frames) - 1)
        if target != self.index:
            self.load_frame(target)
            self._write_review_status()

    def reset_auto(self) -> None:
        if not messagebox.askyesno("Reset automatic mask", "Discard manual edits for this frame?"):
            return
        self.push_undo()
        self.mask = self.auto_mask.copy()
        self.dirty = True
        self.render()

    def valid_xy(self, event: tk.Event) -> tuple[int, int] | None:
        x, y = int(event.x), int(event.y)
        if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
            return x, y
        return None

    def on_left_down(self, event: tk.Event) -> None:
        point = self.valid_xy(event)
        if point is None:
            return
        tool = self.tool.get()
        if tool == "component":
            self.push_undo()
            image = Image.fromarray(self.mask, mode="L")
            ImageDraw.floodfill(image, point, self.selected_label(), thresh=0)
            self.mask = np.asarray(image, dtype=np.uint8).copy()
            self.dirty = True
            self.render()
        elif tool == "brush":
            self.push_undo()
            self.brushing = True
            self.paint(point)
        elif tool == "polygon":
            self.polygon.append(point)
            self.render()

    def on_left_drag(self, event: tk.Event) -> None:
        if self.tool.get() != "brush" or not self.brushing:
            return
        point = self.valid_xy(event)
        if point is not None:
            self.paint(point)

    def on_left_up(self, _event: tk.Event) -> None:
        self.brushing = False

    def paint(self, point: tuple[int, int]) -> None:
        radius = max(1, int(self.brush_size.get()) // 2)
        image = Image.fromarray(self.mask, mode="L")
        draw = ImageDraw.Draw(image)
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=self.selected_label())
        self.mask = np.asarray(image, dtype=np.uint8).copy()
        self.dirty = True
        self.render()

    def commit_polygon(self, _event: tk.Event | None = None) -> None:
        if self.tool.get() != "polygon" or len(self.polygon) < 3:
            return
        self.push_undo()
        image = Image.fromarray(self.mask, mode="L")
        ImageDraw.Draw(image).polygon(self.polygon, fill=self.selected_label())
        self.mask = np.asarray(image, dtype=np.uint8).copy()
        self.polygon.clear()
        self.dirty = True
        self.render()

    def cancel_polygon(self) -> None:
        self.polygon.clear()
        self.render()

    def pick_label(self, event: tk.Event) -> None:
        point = self.valid_xy(event)
        if point is None:
            return
        label = int(self.mask[point[1], point[0]])
        match = next((i for i, value in enumerate(COMMON_LABELS) if value[0] == label), None)
        if match is not None:
            self.label_combo.current(match)
        self.status.set(
            f"Picked class {label} {self.label_name(label)}. "
            "Keep the automatic mask if it is not in the common list."
        )

    def on_motion(self, event: tk.Event) -> None:
        point = self.valid_xy(event)
        if point is None:
            return
        label = int(self.mask[point[1], point[0]])
        checked = "REVIEWED" if self.frame_id in self.reviewed else "NOT REVIEWED"
        self.status.set(
            f"{self.index + 1}/{len(self.frames)}  {self.frame_id}  [{checked}]  "
            f"x={point[0]} y={point[1]} class={label} {self.label_name(label)}  "
            f"reviewed {len(self.reviewed)}/{len(self.frames)}"
        )

    def original_on(self, _event: tk.Event) -> None:
        if not self.show_original:
            self.show_original = True
            self.render()

    def original_off(self, _event: tk.Event) -> None:
        if self.show_original:
            self.show_original = False
            self.render()

    def close(self) -> None:
        if self.dirty and messagebox.askyesno("Save edits", "Save the current frame before closing?"):
            self.save()
        self._write_review_status()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labelspace", type=Path, required=True)
    parser.add_argument("--title", default="Azure Kinect session")
    args = parser.parse_args()
    root = tk.Tk()
    for font_name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkFixedFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            tkfont.nametofont(font_name).configure(family="Microsoft YaHei", size=10)
        except tk.TclError:
            pass
    SemanticMaskEditor(root, args.input_dir, args.output_dir, args.labelspace, args.title)
    root.mainloop()


if __name__ == "__main__":
    main()
