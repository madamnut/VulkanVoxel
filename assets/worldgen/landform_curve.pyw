import argparse
import json
import math
import struct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
POINTS_PATH = SCRIPT_DIR / "landform_curve_points.json"
BIN_PATH = SCRIPT_DIR / "landform_curve.bin"
LUT_PAIR_COUNT = 1000

DEFAULT_POINTS = [
    (-1.00, -0.125),
    (-0.85, -0.10625),
    (-0.60, -0.0421875),
    (-0.40, -0.015625),
    (-0.15, -0.0017578125),
    (0.00, 0.0),
    (0.15, 0.0005859375),
    (0.40, 0.01875),
    (0.60, 0.075),
    (0.85, 0.159375),
    (1.00, 0.25),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def load_points():
    if not POINTS_PATH.exists():
        return list(DEFAULT_POINTS)
    try:
        data = json.loads(POINTS_PATH.read_text(encoding="utf-8"))
        points = [(float(item["x"]), float(item["y"])) for item in data["points"]]
        return normalize_points(points)
    except Exception:
        return list(DEFAULT_POINTS)


def save_points(points):
    data = {
        "points": [
            {"x": round(x, 6), "y": round(y, 6)}
            for x, y in normalize_points(points)
        ]
    }
    POINTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def normalize_points(points):
    by_x = {}
    for x, y in points:
        by_x[clamp(float(x), -1.0, 1.0)] = clamp(float(y), -0.5, 0.5)
    by_x[-1.0] = by_x.get(-1.0, DEFAULT_POINTS[0][1])
    by_x[1.0] = by_x.get(1.0, DEFAULT_POINTS[-1][1])
    return sorted(by_x.items(), key=lambda point: point[0])


def monotone_slopes(points):
    count = len(points)
    if count <= 1:
        return [0.0] * count
    dx = [points[i + 1][0] - points[i][0] for i in range(count - 1)]
    dy = [points[i + 1][1] - points[i][1] for i in range(count - 1)]
    slopes = [0.0 if dx[i] == 0.0 else dy[i] / dx[i] for i in range(count - 1)]
    tangents = [0.0] * count
    tangents[0] = slopes[0]
    tangents[-1] = slopes[-1]
    for i in range(1, count - 1):
        if slopes[i - 1] * slopes[i] <= 0.0:
            tangents[i] = 0.0
        else:
            tangents[i] = (slopes[i - 1] + slopes[i]) * 0.5

    for i in range(count - 1):
        if slopes[i] == 0.0:
            tangents[i] = 0.0
            tangents[i + 1] = 0.0
            continue
        a = tangents[i] / slopes[i]
        b = tangents[i + 1] / slopes[i]
        length = math.hypot(a, b)
        if length > 3.0:
            scale = 3.0 / length
            tangents[i] = scale * a * slopes[i]
            tangents[i + 1] = scale * b * slopes[i]
    return tangents


def sample_curve(points, x):
    points = normalize_points(points)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    tangents = monotone_slopes(points)
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x <= x1:
            width = x1 - x0
            t = (x - x0) / width
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            return (
                h00 * y0 +
                h10 * width * tangents[i] +
                h01 * y1 +
                h11 * width * tangents[i + 1]
            )
    return points[-1][1]


def bake_curve(points, output_path=BIN_PATH):
    points = normalize_points(points)
    samples = []
    for i in range(LUT_PAIR_COUNT):
        x = -1.0 + 2.0 * i / (LUT_PAIR_COUNT - 1)
        samples.append((x, float(sample_curve(points, x))))

    with output_path.open("wb") as file:
        file.write(struct.pack("<I", LUT_PAIR_COUNT))
        for x, y in samples:
            file.write(struct.pack("<ff", x, y))


class CurveEditor:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.root = root
        self.points = load_points()
        self.selected = None
        self.width = 980
        self.height = 620
        self.pad = 58

        root.title("Landform Curve")
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="#111318")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X)
        tk.Button(toolbar, text="Bake BIN", command=self.bake).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Save Points", command=lambda: save_points(self.points)).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Load Points", command=self.load).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT)
        self.status = tk.Label(toolbar, text="")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Double-Button-1>", self.on_double)
        self.draw()

    def y_bounds(self):
        return -0.5, 0.5

    def to_screen(self, x, y):
        y_min, y_max = self.y_bounds()
        sx = self.pad + (x + 1.0) * 0.5 * (self.width - self.pad * 2)
        sy = self.height - self.pad - (y - y_min) / (y_max - y_min) * (self.height - self.pad * 2)
        return sx, sy

    def from_screen(self, sx, sy):
        y_min, y_max = self.y_bounds()
        x = ((sx - self.pad) / (self.width - self.pad * 2)) * 2.0 - 1.0
        y = y_min + (self.height - self.pad - sy) / (self.height - self.pad * 2) * (y_max - y_min)
        return clamp(x, -1.0, 1.0), clamp(y, -0.5, 0.5)

    def draw(self):
        c = self.canvas
        c.delete("all")
        y_min, y_max = self.y_bounds()

        for i in range(11):
            x = -1.0 + i * 0.2
            sx, _ = self.to_screen(x, 0.0)
            c.create_line(sx, self.pad, sx, self.height - self.pad, fill="#242832")
            c.create_text(sx, self.height - self.pad + 18, text=f"{x:.1f}", fill="#8d96a8", font=("Consolas", 9))

        for i in range(9):
            y = y_min + (y_max - y_min) * i / 8
            _, sy = self.to_screen(0.0, y)
            c.create_line(self.pad, sy, self.width - self.pad, sy, fill="#242832")
            c.create_text(self.pad - 8, sy, text=f"{y:.2f}", fill="#8d96a8", font=("Consolas", 9), anchor="e")

        x0, zero_y = self.to_screen(0.0, 0.0)
        c.create_line(x0, self.pad, x0, self.height - self.pad, fill="#566070", width=2)
        c.create_line(self.pad, zero_y, self.width - self.pad, zero_y, fill="#566070", width=2)

        curve_points = []
        for i in range(512):
            x = -1.0 + 2.0 * i / 511
            y = sample_curve(self.points, x)
            curve_points.extend(self.to_screen(x, y))
        c.create_line(*curve_points, fill="#62d6ff", width=3, smooth=True)

        for index, (x, y) in enumerate(self.points):
            sx, sy = self.to_screen(x, y)
            color = "#ffcf5a" if index == self.selected else "#f2f4f8"
            c.create_oval(sx - 6, sy - 6, sx + 6, sy + 6, fill=color, outline="#111318", width=2)
            c.create_text(sx, sy - 18, text=f"{x:.2f},{y:.3f}", fill=color, font=("Consolas", 8))

        self.status.config(text=f"{BIN_PATH.name} | points: {len(self.points)}")

    def nearest_point(self, sx, sy):
        best = None
        best_dist = 14.0
        for index, (x, y) in enumerate(self.points):
            px, py = self.to_screen(x, y)
            dist = math.hypot(px - sx, py - sy)
            if dist < best_dist:
                best = index
                best_dist = dist
        return best

    def on_down(self, event):
        self.selected = self.nearest_point(event.x, event.y)
        self.draw()

    def on_drag(self, event):
        if self.selected is None:
            return
        x, y = self.from_screen(event.x, event.y)
        if self.selected == 0:
            x = -1.0
        elif self.selected == len(self.points) - 1:
            x = 1.0
        else:
            x = clamp(x, self.points[self.selected - 1][0] + 0.001, self.points[self.selected + 1][0] - 0.001)
        self.points[self.selected] = (x, y)
        self.points = normalize_points(self.points)
        self.draw()

    def on_up(self, _):
        save_points(self.points)

    def on_double(self, event):
        x, y = self.from_screen(event.x, event.y)
        if x <= -0.999 or x >= 0.999:
            return
        self.points.append((x, y))
        self.points = normalize_points(self.points)
        self.selected = min(range(len(self.points)), key=lambda i: abs(self.points[i][0] - x))
        save_points(self.points)
        self.draw()

    def delete_selected(self):
        if self.selected is None or self.selected in (0, len(self.points) - 1):
            return
        del self.points[self.selected]
        self.selected = None
        save_points(self.points)
        self.draw()

    def load(self):
        self.points = load_points()
        self.selected = None
        self.draw()

    def reset(self):
        self.points = list(DEFAULT_POINTS)
        self.selected = None
        save_points(self.points)
        self.draw()

    def bake(self):
        save_points(self.points)
        bake_curve(self.points)
        self.messagebox.showinfo("Landform Curve", f"Baked {BIN_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", action="store_true")
    parser.add_argument("--samples", type=int, default=LUT_PAIR_COUNT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.bake:
        bake_curve(load_points())
        return

    import tkinter as tk
    root = tk.Tk()
    CurveEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
