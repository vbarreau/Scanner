"""
ScannerApp — integrated CT scan processing UI.

All interactive panels render inside a fixed embedded matplotlib canvas.
Controls live in a left panel. The window size never changes.
No separate matplotlib windows are opened.
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import cv2
from scipy.interpolate import splev

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import Image as ImageModule  # noqa: E402
from Image import show_slice  # noqa: E402
from func import clip as func_clip  # noqa: E402
from spline_editor import SplineEditor  # noqa: E402

# ── Window geometry ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 1300, 780
LEFT_W = 270
FIG_DPI = 100


class RangeSlider(tk.Frame):
    """Horizontal slider with two draggable handles for a [low, high] range."""

    _R   = 7    # handle radius (px)
    _PAD = 12   # left/right padding before track ends

    def __init__(self, parent, from_: int, to: int,
                 init_low: int, init_high: int,
                 command=None, color: str = '#4499ff', **kw):
        super().__init__(parent, bg="#ebebeb", **kw)
        self._from    = from_
        self._to      = to
        self._low     = init_low
        self._high    = init_high
        self._command = command
        self._color   = color
        self._drag    = None   # 'low' | 'high'

        self._cv = tk.Canvas(self, height=36, bg="#ebebeb",
                             highlightthickness=0, bd=0)
        self._cv.pack(fill=tk.X)
        self._cv.bind("<Configure>",       lambda _: self._draw())
        self._cv.bind("<ButtonPress-1>",   self._press)
        self._cv.bind("<B1-Motion>",       self._move)
        self._cv.bind("<ButtonRelease-1>", lambda _: setattr(self, '_drag', None))

    # ── geometry ────────────────────────────────────────────────────────

    def _val2x(self, val):
        p, w = self._PAD, self._cv.winfo_width()
        return p + (val - self._from) / max(self._to - self._from, 1) * (w - 2 * p)

    def _x2val(self, x):
        p, w = self._PAD, self._cv.winfo_width()
        raw = (x - p) / max(w - 2 * p, 1) * (self._to - self._from) + self._from
        return int(round(max(self._from, min(self._to, raw))))

    # ── drawing ─────────────────────────────────────────────────────────

    def _draw(self):
        c = self._cv
        w = c.winfo_width()
        if w < 4:
            return
        c.delete("all")
        h, r, p = 36, self._R, self._PAD
        cy = h - r - 4          # track / handle centre
        x0, x1 = self._val2x(self._low), self._val2x(self._high)
        # tracks
        c.create_line(p, cy, w - p, cy, fill="#cccccc", width=3, capstyle="round")
        c.create_line(x0, cy, x1, cy, fill=self._color, width=3, capstyle="round")
        # value labels above handles
        c.create_text(x0, cy - r - 2, text=str(self._low),
                      fill="#555555", font=("Helvetica", 7), anchor="s")
        c.create_text(x1, cy - r - 2, text=str(self._high),
                      fill="#555555", font=("Helvetica", 7), anchor="s")
        # handles
        for x in (x0, x1):
            c.create_oval(x - r, cy - r, x + r, cy + r,
                          fill=self._color, outline="white", width=1.5)

    # ── interaction ─────────────────────────────────────────────────────

    def _press(self, ev):
        x0, x1 = self._val2x(self._low), self._val2x(self._high)
        self._drag = 'low' if abs(ev.x - x0) <= abs(ev.x - x1) else 'high'

    def _move(self, ev):
        if self._drag is None:
            return
        val = self._x2val(ev.x)
        if self._drag == 'low':
            self._low  = min(val, self._high)
        else:
            self._high = max(val, self._low)
        self._draw()
        if self._command:
            self._command()

    # ── public API ──────────────────────────────────────────────────────

    def get_low(self):  return self._low
    def get_high(self): return self._high

    def set(self, low: int, high: int):
        self._low, self._high = low, high
        self._draw()


class ScannerApp:
    """Main application window for the CT scan processing pipeline."""

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self):
        self.scan: ImageModule.Image3D | None = None
        self._loading = False
        self._mode: str | None = None   # 'plot' | 'animation' | 'projection'

        self.root = tk.Tk()
        self.root.title("CT Scanner")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(True, True)

        self._default_sl_var: tk.IntVar | None = None
        self._active_tool: str = 'pointer'
        self._measure_pts: list = []
        self._tool_btns: dict = {}

        self._build_layout()
        self._show_welcome()
        # Bind scroll on the matplotlib canvas (works in all modes)
        self._canvas.get_tk_widget().bind("<MouseWheel>", self._on_canvas_scroll)
        self._fig.canvas.mpl_connect('button_press_event', self._on_measure_click)

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    def _build_layout(self):
        # ── Left panel ──────────────────────────────────────────────────
        self._left = tk.Frame(self.root, width=LEFT_W, bg="#ebebeb",
                              relief=tk.FLAT, bd=0)
        self._left.pack(side=tk.LEFT, fill=tk.Y)
        self._left.pack_propagate(False)

        # ── Right panel: embedded matplotlib canvas + slice sidebar ─────
        right = tk.Frame(self.root, bg="#111111")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        fig_w = (WIN_W - LEFT_W) / FIG_DPI
        fig_h = WIN_H / FIG_DPI
        self._fig = Figure(figsize=(fig_w, fig_h), dpi=FIG_DPI,
                           facecolor="#111111")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Slice sidebar — far right strip, hidden until a scan is loaded
        self._slice_sidebar = tk.Frame(right, width=28, bg="#111111")
        self._slice_sidebar.grid_propagate(False)
        self._slice_sidebar.grid(row=0, column=1, sticky="ns")
        self._slice_sidebar.grid_remove()

        # Tool toolbar — vertical strip on the far right
        self._tool_bar_frame = tk.Frame(right, width=44, bg="#1c1c1c")
        self._tool_bar_frame.grid_propagate(False)
        self._tool_bar_frame.grid(row=0, column=2, sticky="ns")
        self._build_tool_toolbar(self._tool_bar_frame)

        # ── Left panel: static top + scrollable middle + fixed bottom ───
        self._build_left_static()
        tk.Frame(self._left, height=1, bg="#cccccc").pack(fill=tk.X)
        # Bottom bar: packed first (side=BOTTOM) so it anchors correctly.
        # Starts empty → zero height. Populated by _enter_mode.
        self._bottom_bar = tk.Frame(self._left, bg="#ebebeb")
        self._bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_scrollable_dyn()

    def _build_scrollable_dyn(self):
        """Create a vertically scrollable area for the dynamic controls."""
        outer = tk.Frame(self._left, bg="#ebebeb")
        outer.pack(fill=tk.BOTH, expand=True)

        self._dyn_canvas = tk.Canvas(outer, bg="#ebebeb",
                                     highlightthickness=0, bd=0)
        sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                          command=self._dyn_canvas.yview)
        self._dyn_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._dyn_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Content frame lives inside the canvas
        self._dyn = tk.Frame(self._dyn_canvas, bg="#ebebeb")
        self._dyn_win = self._dyn_canvas.create_window(
            (0, 0), window=self._dyn, anchor="nw"
        )

        # Keep content frame width in sync with canvas width
        self._dyn_canvas.bind(
            "<Configure>",
            lambda e: self._dyn_canvas.itemconfig(self._dyn_win, width=e.width),
        )
        # Update scroll region whenever content changes
        self._dyn.bind(
            "<Configure>",
            lambda e: self._dyn_canvas.configure(
                scrollregion=self._dyn_canvas.bbox("all")
            ),
        )
        # Mouse-wheel scrolling (Windows)
        self._dyn_canvas.bind("<MouseWheel>", self._on_dyn_scroll)
        self._dyn.bind("<MouseWheel>", self._on_dyn_scroll)

    def _on_dyn_scroll(self, event):
        self._dyn_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_scroll(self, widget):
        """Recursively bind mouse-wheel to a widget tree so scrolling works everywhere."""
        widget.bind("<MouseWheel>", self._on_dyn_scroll)
        for child in widget.winfo_children():
            self._bind_scroll(child)

    # ── Static left content ──────────────────────────────────────────────

    def _build_left_static(self):
        P = 8
        L = self._left

        tk.Label(L, text="CT Scanner", font=("Helvetica", 14, "bold"),
                 bg="#ebebeb").pack(pady=(10, 6))

        # Folder
        frm = tk.LabelFrame(L, text="DICOM folder", bg="#ebebeb",
                             font=("Helvetica", 8), padx=P, pady=4)
        frm.pack(fill=tk.X, padx=P, pady=(0, 4))
        self._folder_var = tk.StringVar(value="(none selected)")
        tk.Label(frm, textvariable=self._folder_var, wraplength=210,
                 anchor="w", bg="#ebebeb", font=("Helvetica", 8),
                 fg="#444444").pack(fill=tk.X)
        tk.Button(frm, text="Browse…", command=self._on_browse).pack(pady=(4, 0))

        # Status + progress
        self._status_var = tk.StringVar(value="Select a folder to begin.")
        tk.Label(L, textvariable=self._status_var, bg="#ebebeb",
                 fg="#666666", font=("Helvetica", 8),
                 wraplength=240).pack(padx=P, pady=(2, 0))
        self._progress = ttk.Progressbar(L, mode='determinate', length=230, maximum=100, value=0)
        self._progress.pack(pady=(2, 4))

        # Scan info
        self._info_frm = tk.LabelFrame(L, text="Scan info", bg="#ebebeb",
                                       font=("Helvetica", 8), padx=P, pady=4)
        self._info_frm.pack(fill=tk.X, padx=P, pady=(0, 4))
        self._info_var = tk.StringVar(value="—")
        tk.Label(self._info_frm, textvariable=self._info_var, bg="#ebebeb",
                 font=("Courier", 8), anchor="w", justify=tk.LEFT).pack(fill=tk.X)

        # Action buttons
        act_frm = tk.LabelFrame(L, text="Actions", bg="#ebebeb",
                                font=("Helvetica", 8), padx=P, pady=4)
        act_frm.pack(fill=tk.X, padx=P, pady=(0, 4))
        self._btn_plot = tk.Button(act_frm, text="Edit Image",
                                   command=self._on_mode_plot, width=22)
        self._btn_rot = tk.Button(act_frm, text="Rotate 3D",
                                  command=self._on_mode_rotate, width=22)
        self._btn_anim = tk.Button(act_frm, text="Create Animation",
                                   command=self._on_mode_animation, width=22)
        self._btn_proj = tk.Button(act_frm, text="Create pseudo radiography",
                                   command=self._on_mode_projection, width=22)
        self._btn_crop = tk.Button(act_frm, text="Crop",
                                   command=self._on_mode_crop, width=22)
        self._btn_spline = tk.Button(act_frm, text="Define Spline",
                                     command=self._on_mode_spline, width=22)
        self._action_buttons = [self._btn_plot, self._btn_rot, self._btn_crop,
                                 self._btn_anim, self._btn_proj, self._btn_spline]
        for btn in self._action_buttons:
            btn.pack(fill=tk.X, pady=2)

        self._act_frm = act_frm   # keep reference for show/hide
        self._refresh_buttons()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _refresh_buttons(self):
        ok = self.scan is not None and not self._loading
        for btn in self._action_buttons:
            btn.config(state=tk.NORMAL if ok else tk.DISABLED)

    # ------------------------------------------------------------------ #
    #  Mode enter / exit                                                   #
    # ------------------------------------------------------------------ #

    def _enter_mode(self, save_label: str, save_cmd, cancel_cmd=None):
        """Hide action buttons; show Save + Cancel bar at the bottom."""
        if cancel_cmd is None:
            cancel_cmd = self._exit_mode
        self._default_sl_var = None   # disable canvas scroll in sub-modes
        self._measure_pts = []
        self._active_tool = 'pointer'
        self._update_tool_buttons()
        self._canvas.get_tk_widget().config(cursor='')
        self._slice_sidebar.grid_remove()
        self._act_frm.pack_forget()
        for w in self._bottom_bar.winfo_children():
            w.destroy()
        tk.Frame(self._bottom_bar, height=1, bg="#cccccc").pack(fill=tk.X)
        inner = tk.Frame(self._bottom_bar, bg="#ebebeb", padx=6, pady=6)
        inner.pack(fill=tk.X)
        tk.Button(inner, text=save_label, bg="#d0f0d0",
                  font=("Helvetica", 9, "bold"),
                  command=save_cmd).pack(fill=tk.X, pady=(0, 3))
        tk.Button(inner, text="Cancel", bg="#ffe0e0",
                  command=cancel_cmd).pack(fill=tk.X)

    def _exit_mode(self):
        """Restore action buttons; empty the bottom bar; return to welcome."""
        self._mode = None
        self._measure_pts = []
        self._active_tool = 'pointer'
        self._update_tool_buttons()
        self._canvas.get_tk_widget().config(cursor='')
        for w in self._bottom_bar.winfo_children():
            w.destroy()
        # Re-pack actions only if they were hidden (pack_forget removes from manager)
        if not self._act_frm.winfo_manager():
            self._act_frm.pack(fill=tk.X, padx=8, pady=(0, 4),
                               after=self._info_frm)
        self._clear_dyn()
        self._show_welcome()
        if self.scan is not None:
          self._show_first_slice()

    def _clear_dyn(self):
        for w in self._dyn.winfo_children():
            w.destroy()

    def _redraw(self):
        self._canvas.draw_idle()

    def _lframe(self, title):
        """Convenience: returns a new LabelFrame inside the dynamic panel."""
        f = tk.LabelFrame(self._dyn, text=title, bg="#ebebeb",
                          font=("Helvetica", 8), padx=6, pady=4)
        f.pack(fill=tk.X, padx=6, pady=3)
        return f

    def _slider(self, parent, label, from_, to, init, resolution=1):
        tk.Label(parent, text=label, bg="#ebebeb",
                 font=("Helvetica", 8)).pack(anchor="w")
        sl = tk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL,
                      resolution=resolution, length=220, bg="#ebebeb",
                      sliderlength=12, font=("Helvetica", 7),
                      highlightthickness=0)
        sl.set(init)
        sl.pack(fill=tk.X)
        return sl

    # ------------------------------------------------------------------ #
    #  Welcome                                                             #
    # ------------------------------------------------------------------ #

    def _show_welcome(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#111111")
        ax.text(0.5, 0.56, "CT Scanner", ha='center', va='center',
                fontsize=30, color='white', fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.44, "Select a DICOM folder to begin.",
                ha='center', va='center', fontsize=13, color='#888888',
                transform=ax.transAxes)
        ax.axis('off')
        self._redraw()

    # ------------------------------------------------------------------ #
    #  Folder browse & load                                                #
    # ------------------------------------------------------------------ #

    def _on_browse(self):
        if self._mode is not None:
            self._exit_mode()
        folder = filedialog.askdirectory(title="Select DICOM folder")
        if not folder:
            return
        self._folder_var.set(os.path.basename(folder) or folder)
        self._load_scan(folder)

    def _load_scan(self, folder: str):
        self._exit_mode()          # restore actions bar, clear dyn, welcome screen
        self._loading = True
        self.scan = None
        self._refresh_buttons()
        self._status_var.set("Loading DICOM files…")
        self._info_var.set("—")
        self._progress.config(value=0, maximum=100)

        def _progress_cb(current, total):
            pct = int(current / total * 100)
            self.root.after(0, lambda v=pct: self._progress.config(value=v))

        def _worker():
            try:
                scan = ImageModule.Image3D(folder, progress_callback=_progress_cb)
                self.root.after(0, lambda: self._on_load_done(scan))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_load_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_load_done(self, scan: ImageModule.Image3D):
        self.scan = scan
        self._loading = False
        self._progress.config(value=0)
        p = scan.properties()
        self._status_var.set("Ready.")
        self._info_var.set(
            f"Shape : {p['dim']}\n"
            f"Min   : {p['min']:.0f}\n"
            f"Max   : {p['max']:.0f}\n"
            f"Mean  : {p['valeur moyenne']:.1f}\n"
            f"Zeros : {p['Pixels nuls']}"
        )
        self._refresh_buttons()
        self._show_first_slice()

    def _show_first_slice(self):
        """Display the middle slice of the loaded scan on the main canvas."""
        assert self.scan is not None
        scan = self.scan
        axis = scan.normal_axis
        n = scan.shape[axis]
        mid = n // 2
        self._default_sl_var = tk.IntVar(value=mid)
        self._build_default_controls(n)
        self._draw_default_slice(mid)

    def _build_default_controls(self, n: int):
        """Show the flat vertical slice slider in the right sidebar."""
        self._clear_dyn()
        for w in self._slice_sidebar.winfo_children():
            w.destroy()
        self._slice_sidebar.grid()  # reveal
        assert self._default_sl_var is not None

        tk.Label(self._slice_sidebar, textvariable=self._default_sl_var,
                 bg="#111111", fg="#555555",
                 font=("Helvetica", 7)).pack(side=tk.BOTTOM, pady=4)

        sl = tk.Scale(self._slice_sidebar, from_=n - 1, to=0,
                      orient=tk.VERTICAL,
                      variable=self._default_sl_var,
                      resolution=1,
                      bg="#111111", fg="#111111",
                      troughcolor="#2c2c2c", activebackground="#555555",
                      highlightthickness=0, bd=0, relief=tk.FLAT,
                      showvalue=False, sliderlength=18,
                      command=lambda v: self._draw_default_slice(int(v)))
        sl.pack(fill=tk.Y, expand=True, padx=4, pady=(6, 2))
        self._default_sl = sl

    def _draw_default_slice(self, idx: int):
        scan = self.scan
        assert scan is not None
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#111111")
        show_slice(ax, scan.img, idx, scan.normal_axis, scan.x, scan.y, scan.z)
        self._draw_measure_overlay(ax)
        self._redraw()

    def _on_canvas_scroll(self, event):
        """Scroll through slices with the mouse wheel on the default page."""
        if self._mode is not None or self.scan is None or self._default_sl_var is None:
            return
        axis = self.scan.normal_axis
        n = self.scan.shape[axis]
        delta = event.delta // 120  # standardize scroll speed across platforms
        new_idx = max(0, min(n - 1, self._default_sl_var.get() + delta))
        self._default_sl_var.set(new_idx)
        self._draw_default_slice(new_idx)

    # ------------------------------------------------------------------ #
    #  Tool toolbar                                                        #
    # ------------------------------------------------------------------ #

    def _build_tool_toolbar(self, parent: tk.Frame):
        tk.Frame(parent, height=8, bg="#1c1c1c").pack(fill=tk.X)
        tools = [
            ('pointer', '↖', 'Select'),
            ('measure', '↔', 'Measure distance'),
        ]
        for tool_id, symbol, _ in tools:
            btn = tk.Button(
                parent, text=symbol,
                font=("Helvetica", 15),
                width=2, height=1,
                relief=tk.FLAT, bd=0,
                bg='#2a2a2a', fg='white',
                activebackground='#3a5aaa', activeforeground='white',
                cursor='hand2',
                command=lambda t=tool_id: self._set_tool(t),
            )
            btn.pack(pady=(2, 0), padx=4, fill=tk.X)
            self._tool_btns[tool_id] = btn
            tk.Frame(parent, height=1, bg='#333333').pack(fill=tk.X, padx=4)
        self._update_tool_buttons()

    def _set_tool(self, tool: str):
        self._active_tool = tool
        self._canvas.get_tk_widget().config(
            cursor='crosshair' if tool == 'measure' else ''
        )
        if tool != 'measure':
            self._measure_pts = []
            if self._mode is None and self._default_sl_var is not None:
                self._draw_default_slice(self._default_sl_var.get())
        self._update_tool_buttons()

    def _update_tool_buttons(self):
        for tool_id, btn in self._tool_btns.items():
            active = (tool_id == self._active_tool)
            btn.config(
                bg='#4477cc' if active else '#2a2a2a',
                relief=tk.SUNKEN if active else tk.FLAT,
            )

    # ------------------------------------------------------------------ #
    #  Measure distance tool                                               #
    # ------------------------------------------------------------------ #

    def _on_measure_click(self, event):
        if self._active_tool != 'measure':
            return
        if self._mode is not None or self.scan is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        if len(self._measure_pts) >= 2:
            self._measure_pts = []
        self._measure_pts.append((event.xdata, event.ydata))
        if self._default_sl_var is not None:
            self._draw_default_slice(self._default_sl_var.get())

    def _draw_measure_overlay(self, ax):
        if not self._measure_pts:
            return
        xs = [p[0] for p in self._measure_pts]
        ys = [p[1] for p in self._measure_pts]
        ax.plot(xs, ys, 'o', color='#ff4444', markersize=7, zorder=10,
                markeredgecolor='white', markeredgewidth=0.8)
        if len(self._measure_pts) == 2:
            ax.plot(xs, ys, color='#ff4444', linewidth=1.5,
                    linestyle='--', zorder=9)
            dist = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
            mx, my = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
            ax.text(mx, my, f' {dist:.1f} mm', color='white', fontsize=9,
                    zorder=11,
                    bbox=dict(facecolor='#1a1a1a', alpha=0.75,
                              pad=2, edgecolor='none'))

    def _on_load_error(self, exc: Exception):
        self._loading = False
        self._progress.config(value=0)
        self._status_var.set(f"Error: {exc}")
        messagebox.showerror("Load error", str(exc))
        self._refresh_buttons()

    # ================================================================== #
    #  MODE — Interactive Plot                                             #
    # ================================================================== #

    def _on_mode_plot(self):
        if self.scan is None:
            return
        self._mode = 'plot'
        self._plot_img_backup = self.scan.img.copy()
        self._enter_mode("Save", self._plot_save,
                         cancel_cmd=self._plot_cancel)
        self._build_plot_controls()
        self._build_plot_figure()

    # ── Left panel controls ──────────────────────────────────────────────

    def _build_plot_controls(self):
        self._clear_dyn()
        assert self.scan is not None
        s = self.scan.shape

        frm_sl = self._lframe("Slices")
        self._sl_x = self._slider(frm_sl, "X", 0, s[0] - 1, s[0] // 2)
        self._sl_y = self._slider(frm_sl, "Y", 0, s[1] - 1, s[1] // 2)
        self._sl_z = self._slider(frm_sl, "Z", 0, s[2] - 1, s[2] // 2)
        for sl in (self._sl_x, self._sl_y, self._sl_z):
            sl.config(command=lambda _: self._plot_update_slices())

        frm_pr = self._lframe("Processing")
        self._sl_contrast = self._slider(frm_pr, "Contrast", -100, 100, 0)
        tk.Button(frm_pr, text="Apply contrast",
                  command=self._plot_apply_contrast).pack(fill=tk.X, pady=1)

        self._sl_seuil = self._slider(frm_pr, "Compress seuil",
                                       0, int(self.scan.img.max()), 0)
        tk.Button(frm_pr, text="Apply compress",
                  command=self._plot_apply_compress).pack(fill=tk.X, pady=1)

        img_min = int(self.scan.img.min())
        img_max = int(self.scan.img.max())
        tk.Label(frm_pr, text="Clip range", bg="#ebebeb",
                 font=("Helvetica", 8)).pack(anchor="w")
        self._rs_clip = RangeSlider(frm_pr, img_min, img_max,
                                    img_min, img_max)
        self._rs_clip.pack(fill=tk.X)
        tk.Button(frm_pr, text="Apply clip",
                  command=self._plot_apply_clip).pack(fill=tk.X, pady=1)

        self._bind_scroll(self._dyn)

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_plot_figure(self):
        assert self.scan is not None
        self._fig.clear()
        gs = self._fig.add_gridspec(
            2, 3, height_ratios=[4, 1],
            hspace=0.06, wspace=0.03,
            left=0.01, right=0.99, top=0.97, bottom=0.03,
        )
        self._ax_px = self._fig.add_subplot(gs[0, 0])
        self._ax_py = self._fig.add_subplot(gs[0, 1])
        self._ax_pz = self._fig.add_subplot(gs[0, 2])
        self._ax_hist = self._fig.add_subplot(gs[1, :])

        for ax, title in zip(
            (self._ax_px, self._ax_py, self._ax_pz), ("Slice X", "Slice Y", "Slice Z")
        ):
            ax.set_facecolor("black")
            ax.set_title(title, color="white", fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")

        self._ax_hist.set_facecolor("#1a1a1a")
        for sp in self._ax_hist.spines.values():
            sp.set_edgecolor("#444444")

        ix, iy, iz = self._sl_x.get(), self._sl_y.get(), self._sl_z.get()
        img = self.scan.img
        vmin, vmax = 0, img.max()
        aspects = [(self.scan.z[-1]-self.scan.z[0])/(self.scan.y[-1]-self.scan.y[0]),
                   (self.scan.z[-1]-self.scan.z[0])/(self.scan.x[-1]-self.scan.x[0]),
                   (self.scan.y[-1]-self.scan.y[0])/(self.scan.x[-1]-self.scan.x[0])]
        self._im_px = self._ax_px.imshow(
            img[ix, :, :].T, cmap='gray', aspect=aspects[0], origin='lower', vmin=vmin, vmax=vmax)
        self._im_py = self._ax_py.imshow(
            img[:, iy, :].T, cmap='gray', aspect=aspects[1], origin='lower', vmin=vmin, vmax=vmax)
        self._im_pz = self._ax_pz.imshow(
            img[:, :, iz], cmap='gray', aspect=aspects[2], origin='upper', vmin=vmin, vmax=vmax)

        lw, a = 0.8, 0.65
        self._chl_px = [
            self._ax_px.axvline(iy, color='red',    lw=lw, alpha=a),
            self._ax_px.axhline(iz, color='cyan',   lw=lw, alpha=a),
        ]
        self._chl_py = [
            self._ax_py.axvline(ix, color='yellow', lw=lw, alpha=a),
            self._ax_py.axhline(iz, color='cyan',   lw=lw, alpha=a),
        ]
        self._chl_pz = [
            self._ax_pz.axhline(ix, color='yellow', lw=lw, alpha=a),
            self._ax_pz.axvline(iy, color='red',    lw=lw, alpha=a),
        ]
        self._plot_draw_hist()

        # Pixel-value overlay label in the lower-right corner of the canvas
        if not hasattr(self, '_pixel_label'):
            self._pixel_label = tk.Label(
                self._canvas.get_tk_widget(),
                text="", bg="#111111", fg="white",
                font=("Courier", 9), padx=4, pady=2,
            )
        self._pixel_label.place(relx=1.0, rely=1.0, anchor="se")
        self._pixel_label.lift()

        # Connect hover event
        if hasattr(self, '_plot_hover_cid'):
            self._canvas.mpl_disconnect(self._plot_hover_cid)
        self._plot_hover_cid = self._canvas.mpl_connect(
            'motion_notify_event', self._plot_on_hover)

        self._redraw()

    def _plot_on_hover(self, event):
        axes_data = {
            id(self._ax_px): ('X', 'y', 'z'),
            id(self._ax_py): ('Y', 'x', 'z'),
            id(self._ax_pz): ('Z', 'x', 'y'),
        }
        if event.inaxes is None or id(event.inaxes) not in axes_data:
            self._pixel_label.config(text="")
            return
        assert self.scan is not None
        if event.xdata is None or event.ydata is None:
            self._pixel_label.config(text="")
            return
        col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
        img = self.scan.img
        ix = int(self._sl_x.get())
        iy = int(self._sl_y.get())
        iz = int(self._sl_z.get())
        label, ax_id = axes_data[id(event.inaxes)][0], id(event.inaxes)
        try:
            if ax_id == id(self._ax_px):
                # imshow(img[ix, :, :].T) → col=y, row=z
                c, r = np.clip(col, 0, img.shape[1]-1), np.clip(row, 0, img.shape[2]-1)
                val = img[ix, c, r]
            elif ax_id == id(self._ax_py):
                # imshow(img[:, iy, :].T) → col=x, row=z
                c, r = np.clip(col, 0, img.shape[0]-1), np.clip(row, 0, img.shape[2]-1)
                val = img[c, iy, r]
            else:
                # imshow(img[:, :, iz]) → col=y, row=x
                c, r = np.clip(col, 0, img.shape[1]-1), np.clip(row, 0, img.shape[0]-1)
                val = img[r, c, iz]
            self._pixel_label.config(text=f"Slice {label}  val={val:.1f}")
        except Exception:
            self._pixel_label.config(text="")

    def _plot_update_slices(self):
        if not hasattr(self, '_im_px'):
            return
        assert self.scan is not None
        ix, iy, iz = int(self._sl_x.get()), int(self._sl_y.get()), int(self._sl_z.get())
        img = self.scan.img
        self._im_px.set_data(img[ix, :, :].T)
        self._im_py.set_data(img[:, iy, :].T)
        self._im_pz.set_data(img[:, :, iz])
        self._im_px.set_clim(0, img.max())
        self._chl_px[0].set_xdata([iy])
        self._chl_px[1].set_ydata([iz])
        self._chl_py[0].set_xdata([ix])
        self._chl_py[1].set_ydata([iz])
        self._chl_pz[0].set_ydata([ix])
        self._chl_pz[1].set_xdata([iy])
        self._canvas.draw_idle()

    def _plot_draw_hist(self):
        assert self.scan is not None
        ax = self._ax_hist
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.hist(self.scan.img.flatten(), bins=256, color='#4488ff',
                alpha=0.85, log=True)
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    def _plot_apply_contrast(self):
        assert self.scan is not None
        self.scan.change_contrast(self._sl_contrast.get())
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_apply_compress(self):
        assert self.scan is not None
        self.scan.compress(self._sl_seuil.get(), 0.8)
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_apply_clip(self):
        assert self.scan is not None
        self.scan.img = func_clip(self.scan.img,
                                  self._rs_clip.get_low(),
                                  self._rs_clip.get_high())
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_reset(self):
        assert self.scan is not None
        self.scan.img = self._plot_img_backup.copy()
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_cancel(self):
        """Reset the image to its original state then exit the mode."""
        assert self.scan is not None
        self.scan.img = self._plot_img_backup.copy()
        self._plot_cleanup_hover()
        self._exit_mode()

    def _plot_save(self):
        self._plot_cleanup_hover()
        self._exit_mode()

    def _plot_cleanup_hover(self):
        if hasattr(self, '_plot_hover_cid'):
            self._canvas.mpl_disconnect(self._plot_hover_cid)
            del self._plot_hover_cid
        if hasattr(self, '_pixel_label'):
            self._pixel_label.place_forget()

    # ================================================================== #
    #  MODE — Animation                                                    #
    # ================================================================== #

    def _on_mode_animation(self):
        if self.scan is None:
            return
        self._mode = 'animation'
        self._anim = {'axis': 'z', 'pov': 'x', 'grad': False}
        self._enter_mode("Generate video…", self._anim_generate)
        self._build_anim_controls()
        self._build_anim_figure()

    # ── Left panel controls ──────────────────────────────────────────────

    def _build_anim_controls(self):
        self._clear_dyn()

        # Rotation axis
        frm_ax = self._lframe("Rotation axis")
        row = tk.Frame(frm_ax, bg="#ebebeb")
        row.pack()
        self._anim_ax_btns = {}
        for a in ('x', 'y', 'z'):
            b = tk.Button(row, text=a.upper(), width=5,
                          command=lambda v=a: self._anim_set('axis', v))
            b.pack(side=tk.LEFT, padx=2)
            self._anim_ax_btns[a] = b

        # Camera POV
        frm_pov = self._lframe("Camera POV")
        row2 = tk.Frame(frm_pov, bg="#ebebeb")
        row2.pack()
        self._anim_pov_btns = {}
        for a in ('x', 'y', 'z'):
            b = tk.Button(row2, text=a.upper(), width=5,
                          command=lambda v=a: self._anim_set('pov', v))
            b.pack(side=tk.LEFT, padx=2)
            self._anim_pov_btns[a] = b

        # Settings
        frm_set = self._lframe("Settings")
        self._sl_frames = self._slider(frm_set, "Frames", 8, 144, 72, resolution=4)
        self._anim_grad_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frm_set, text="Additive (gradient) mode",
                       variable=self._anim_grad_var, bg="#ebebeb",
                       command=lambda: self._anim['update']
                       ).pack(anchor="w")
        # fix: use a real command
        self._anim_grad_var.trace_add('write',
                                      lambda *_: self._anim.__setitem__('grad', self._anim_grad_var.get()))

        # Buttons
        frm_btn = tk.Frame(self._dyn, bg="#ebebeb")
        frm_btn.pack(fill=tk.X, padx=6, pady=3)
        # tk.Button(frm_btn, text="Preview frame",
        #           command=self._anim_preview).pack(fill=tk.X, pady=2)

        self._anim_refresh_colors()
        self._bind_scroll(self._dyn)

    def _anim_set(self, key: str, val: str):
        self._anim[key] = val
        self._anim_refresh_colors()
        if key in ('axis', 'pov'):
            self._build_anim_figure()

    def _anim_refresh_colors(self):
        for a, b in self._anim_ax_btns.items():
            b.config(bg='#90ee90' if a == self._anim['axis'] else 'SystemButtonFace')
        for a, b in self._anim_pov_btns.items():
            b.config(bg='#add8e6' if a == self._anim.get('pov', 'x') else 'SystemButtonFace')

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_anim_figure(self):
        assert self.scan is not None
        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")
        mip = {
            'x': self.scan.img.max(axis=0),
            'y': self.scan.img.max(axis=1),
            'z': self.scan.img.max(axis=2),
        }
        gs = self._fig.add_gridspec(
            1, 3, 
            hspace=0.12, wspace=0.03,
            left=0.01, right=0.99, top=0.97, bottom=0.03,
        )
        self._anim_mip_ax = {}
        aspects = [(self.scan.z[-1]-self.scan.z[0])/(self.scan.y[-1]-self.scan.y[0]),
            (self.scan.z[-1]-self.scan.z[0])/(self.scan.x[-1]-self.scan.x[0]),
            (self.scan.y[-1]-self.scan.y[0])/(self.scan.x[-1]-self.scan.x[0])]
        nX, nY, nZ = self.scan.img.shape
        # For each view: (row_axis, col_axis) — what patient axis maps to rows/cols
        _view_axes = {'x': ('z', 'y'), 'y': ('z', 'x'), 'z': ('x', 'y')}
        _view_shape = {'x': (nZ, nY), 'y': (nZ, nX), 'z': (nX, nY)}
        _axis_style = dict(color='#ff6600', lw=1.2, ls='--', alpha=0.85, zorder=5)
        rot_axis = self._anim['axis']
        for i, key in enumerate(('x', 'y', 'z')):
            ax = self._fig.add_subplot(gs[0, i])
            data = mip[key].T if key in ('x', 'y') else mip[key]
            origin = 'lower' if key in ('x', 'y') else 'upper'
            ax.imshow(data, cmap='gray', aspect=aspects[i], origin=origin)
            # Draw rotation axis
            nrows, ncols = _view_shape[key]
            row_ax, col_ax = _view_axes[key]
            if rot_axis == key:
                # Viewing along the rotation axis: show as a crosshair dot
                ax.plot((ncols - 1) / 2, (nrows - 1) / 2,
                        marker='+', color='#ff6600', ms=14, mew=1.8,
                        alpha=0.9, zorder=5)
            elif rot_axis == row_ax:
                # Rotation axis runs along rows → vertical line at center column
                ax.axvline((ncols - 1) / 2, **_axis_style)
            else:
                # Rotation axis runs along columns → horizontal line at center row
                ax.axhline((nrows - 1) / 2, **_axis_style)
            ax.set_title(f"View along {key.upper()}", color='white', fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            pov = self._anim.get('pov', 'x')
            color = '#add8e6' if key == pov else '#333333'
            lw = 3 if key == pov else 1
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(lw)
            self._anim_mip_ax[key] = ax

        # # Preview area (bottom row spans all 3 columns)
        # self._ax_anim_prev = self._fig.add_subplot(gs[1, :])
        # self._ax_anim_prev.set_facecolor("#0d0d0d")
        # self._ax_anim_prev.tick_params(left=False, bottom=False,
        #                                labelleft=False, labelbottom=False)
        # self._ax_anim_prev.text(
        #     0.5, 0.5, 'Click "Preview frame" to render a vispy frame.',
        #     ha='center', va='center', color='#666666',
        #     fontsize=11, transform=self._ax_anim_prev.transAxes,
        # )
        # for sp in self._ax_anim_prev.spines.values():
        #     sp.set_edgecolor('#333333')
        # self._redraw()

    # def _anim_preview(self):
    #     self._status_var.set("Rendering preview…")
    #     self.root.update()
    #     try:
    #         frame = self.scan.vispy_preview_frame(
    #             grad=self._anim['grad'],
    #             axis=self._anim['axis'],
    #             pov=self._anim.get('pov', 'x'),
    #         )
    #         ax = self._ax_anim_prev
    #         ax.clear()
    #         ax.imshow(frame, aspect='auto')
    #         ax.set_title("Vispy preview", color='white', fontsize=9, pad=3)
    #         ax.tick_params(left=False, bottom=False,
    #                        labelleft=False, labelbottom=False)
    #         self._redraw()
    #         self._status_var.set("Preview ready.")
    #     except Exception as exc:
    #         self._status_var.set(f"Preview error: {exc}")
    #         tk.messagebox.showerror("Preview error", str(exc))

    def _anim_generate(self):
        outpath = filedialog.asksaveasfilename(
            title="Save animation as…",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
            initialfile="scan_anim.mp4",
        )
        if not outpath:
            return
        n_frames = int(self._sl_frames.get())
        axis = self._anim['axis']
        pov = self._anim.get('pov', 'x')
        grad = self._anim['grad']

        self._status_var.set(f"Rendering {n_frames} frames… (UI will freeze)")
        self.root.update()
        # vispy (Qt OpenGL) must run on the main thread
        self.root.after(20, lambda: self._do_render(outpath, n_frames, axis, pov, grad))

    def _do_render(self, outpath, n_frames, axis, pov, grad):
        assert self.scan is not None
        self._progress.config(maximum=n_frames, value=0)

        def _progress_cb(current, total):
            self._progress.config(value=current)
            self.root.update_idletasks()

        try:
            self.scan.rotation_video(
                outpath=outpath, n_frames=n_frames,
                grad=grad, axis=axis, pov=pov,
                progress_callback=_progress_cb,
            )
            self._status_var.set(f"Saved → {os.path.basename(outpath)}")
        except Exception as exc:
            self._status_var.set(f"Render error: {exc}")
            messagebox.showerror("Render error", str(exc))
        finally:
            self._progress.config(value=0, maximum=100)

    # ================================================================== #
    #  MODE — Projection Viewer                                            #
    # ================================================================== #

    def _on_mode_projection(self):
        if self.scan is None:
            return
        self._mode = 'projection'
        self._proj_grad = False
        self._enter_mode("Save images", self._proj_save)
        self._build_proj_controls()
        self._build_proj_figure()

    # ── Left panel controls ──────────────────────────────────────────────

    def _build_proj_controls(self):
        self._clear_dyn()

        frm = self._lframe("Projection settings")
        self._proj_grad_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frm, text="Gradient (additive) mode",
                       variable=self._proj_grad_var, bg="#ebebeb",
                       command=self._proj_toggle).pack(anchor="w")

        self._bind_scroll(self._dyn)

    def _proj_toggle(self):
        self._proj_grad = self._proj_grad_var.get()
        self._build_proj_figure()

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_proj_figure(self):
        assert self.scan is not None
        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")
        projs = [self.scan.projection(ax, grad=self._proj_grad) for ax in range(3)]
        titles = ["Projection X", "Projection Y", "Projection Z"]
        gs = self._fig.add_gridspec(
            1, 3, wspace=0.03,
            left=0.01, right=0.99, top=0.95, bottom=0.04,
        )
        aspects = [(self.scan.z[-1]-self.scan.z[0])/(self.scan.y[-1]-self.scan.y[0]),
            (self.scan.z[-1]-self.scan.z[0])/(self.scan.x[-1]-self.scan.x[0]),
            (self.scan.y[-1]-self.scan.y[0])/(self.scan.x[-1]-self.scan.x[0])
        ]
        for i, (proj, title) in enumerate(zip(projs, titles)):
            ax = self._fig.add_subplot(gs[0, i])

            # Projections 0 (X) and 1 (Y): .T puts Z as row axis → origin='lower' puts
            # inferior at bottom. Projection 2 (Z) is an axial view: no transpose needed.
            if i < 2:
                ax.imshow(proj.T, cmap='gray', aspect=aspects[i], origin='lower')
            else:
                ax.imshow(proj, cmap='gray', aspect=aspects[i])
            ax.set_title(title, color='white', fontsize=10, pad=4)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor('#333333')
        self._proj_data = projs
        self._redraw()

    def _proj_save(self):
        folder = filedialog.askdirectory(title="Save projections to folder…")
        if not folder:
            return
        for i, proj in enumerate(self._proj_data):
            p = proj.astype(float)
            p = (p - p.min()) / (p.max() - p.min() + 1e-8) * 255
            cv2.imwrite(
                os.path.join(folder, f"projection_{'XYZ'[i]}.png"),
                p.astype(np.uint8),
            )
        messagebox.showinfo("Saved", f"3 projection images saved to:\n{folder}")

    # ================================================================== #
    #  MODE — Rotate 3D                                                    #
    # ================================================================== #

    def _on_mode_rotate(self):
        if self.scan is None:
            return
        self._mode = 'rotate'
        self._rotate_img_backup = self.scan.img.copy()
        self._rotate_xyz_backup = (self.scan.x.copy(), self.scan.y.copy(), self.scan.z.copy())
        self._enter_mode("Save", self._rotate_save, cancel_cmd=self._rotate_cancel)
        self._build_rotate_controls()
        self._build_rotate_figure()

    # ── Left panel controls ──────────────────────────────────────────────

    def _build_rotate_controls(self):
        self._clear_dyn()

        frm = self._lframe("Rotate 90°")
        tk.Label(frm, text="Around X axis", bg="#ebebeb",
                 font=("Helvetica", 8)).pack(anchor="w")
        row_x = tk.Frame(frm, bg="#ebebeb")
        row_x.pack(fill=tk.X, pady=1)
        tk.Button(row_x, text="X +90°", width=9,
                  command=lambda: self._rotate_apply((1, 2), 1)).pack(side=tk.LEFT, padx=2)
        tk.Button(row_x, text="X −90°", width=9,
                  command=lambda: self._rotate_apply((1, 2), -1)).pack(side=tk.LEFT, padx=2)

        tk.Label(frm, text="Around Y axis", bg="#ebebeb",
                 font=("Helvetica", 8)).pack(anchor="w")
        row_y = tk.Frame(frm, bg="#ebebeb")
        row_y.pack(fill=tk.X, pady=1)
        tk.Button(row_y, text="Y +90°", width=9,
                  command=lambda: self._rotate_apply((0, 2), 1)).pack(side=tk.LEFT, padx=2)
        tk.Button(row_y, text="Y −90°", width=9,
                  command=lambda: self._rotate_apply((0, 2), -1)).pack(side=tk.LEFT, padx=2)

        tk.Label(frm, text="Around Z axis", bg="#ebebeb",
                 font=("Helvetica", 8)).pack(anchor="w")
        row_z = tk.Frame(frm, bg="#ebebeb")
        row_z.pack(fill=tk.X, pady=1)
        tk.Button(row_z, text="Z +90°", width=9,
                  command=lambda: self._rotate_apply((0, 1), 1)).pack(side=tk.LEFT, padx=2)
        tk.Button(row_z, text="Z −90°", width=9,
                  command=lambda: self._rotate_apply((0, 1), -1)).pack(side=tk.LEFT, padx=2)

        frm_r = tk.Frame(self._dyn, bg="#ebebeb")
        frm_r.pack(fill=tk.X, padx=6, pady=4)
        tk.Button(frm_r, text="Reset", bg="#fff0d0",
                  command=self._rotate_reset).pack(fill=tk.X)

        self._bind_scroll(self._dyn)

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_rotate_figure(self):
        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")
        gs = self._fig.add_gridspec(
            1, 3, wspace=0.03,
            left=0.01, right=0.99, top=0.95, bottom=0.04,
        )
        self._rot_axes = []
        mip_labels = ("MIP along X", "MIP along Y", "MIP along Z")
        for i in range(3):
            ax = self._fig.add_subplot(gs[0, i])
            ax.set_facecolor("black")
            ax.set_title(mip_labels[i], color="white", fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            self._rot_axes.append(ax)
        self._rotate_redraw_mips()

    def _rotate_redraw_mips(self):
        assert self.scan is not None
        img = self.scan.img
        for i, ax in enumerate(self._rot_axes):
            ax.clear()
            ax.set_facecolor("black")
            mip_labels = ("MIP along X", "MIP along Y", "MIP along Z")
            ax.set_title(mip_labels[i], color="white", fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            ax.imshow(img.max(axis=i), cmap="gray", aspect="auto", origin="upper")
        self._redraw()

    # ── Rotation logic ───────────────────────────────────────────────────

    def _rotate_apply(self, axes: tuple, k: int):
        """Apply np.rot90 with given axes/k and update x/y/z accordingly."""
        assert self.scan is not None
        self.scan.img = np.rot90(self.scan.img, k=k, axes=axes)
        self.scan.x, self.scan.y, self.scan.z = self._rotated_coords(
            self.scan.x, self.scan.y, self.scan.z, axes, k)
        self.scan.shape = self.scan.img.shape
        self._rotate_redraw_mips()

    @staticmethod
    def _rotated_coords(x, y, z, axes, k):
        """Return updated (x, y, z) after np.rot90(img, k, axes).

        np.rot90 with k=1 in the (a0, a1) plane:
          result[..., new_a0, ..., new_a1, ...] comes from [-a1, +a0].
          Concretely the new a0-axis gets old a1's coords,
          and the new a1-axis gets reversed old a0's coords.
        k=-1 (or k=3) is the opposite: new a0 ← reversed old a1, new a1 ← old a0.
        """
        coords = [x, y, z]
        a0, a1 = axes
        k = k % 4
        if k == 1:
            new_a0 = coords[a1]
            new_a1 = coords[a0][::-1]
        elif k == 3:  # same as k=-1
            new_a0 = coords[a1][::-1]
            new_a1 = coords[a0]
        elif k == 2:
            new_a0 = coords[a0][::-1]
            new_a1 = coords[a1][::-1]
        else:  # k == 0, no-op
            return x, y, z
        coords[a0] = new_a0
        coords[a1] = new_a1
        return coords[0], coords[1], coords[2]

    def _rotate_reset(self):
        assert self.scan is not None
        self.scan.img = self._rotate_img_backup.copy()
        self.scan.x, self.scan.y, self.scan.z = (
            self._rotate_xyz_backup[0].copy(),
            self._rotate_xyz_backup[1].copy(),
            self._rotate_xyz_backup[2].copy(),
        )
        self.scan.shape = self.scan.img.shape
        self._rotate_redraw_mips()

    def _rotate_cancel(self):
        self._rotate_reset()
        self._exit_mode()

    def _rotate_save(self):
        self._exit_mode()

    # ================================================================== #
    #  MODE — Crop                                                         #
    # ================================================================== #

    def _on_mode_crop(self):
        if self.scan is None:
            return
        self._mode = 'crop'
        self._enter_mode("Apply Crop", self._crop_save, cancel_cmd=self._exit_mode)
        self._build_crop_controls()
        self._build_crop_figure()

    # ── Left panel controls ──────────────────────────────────────────────

    def _build_crop_controls(self):
        self._clear_dyn()
        assert self.scan is not None
        s = self.scan.shape  # (nX, nY, nZ)
        cx, cy, cz = '#00ff99', '#ff9900', '#4499ff'  # match line colours

        frm = self._lframe("Crop bounds (indices)")
        for label, n, color, attr in (
            ("X", s[0], cx, '_rs_crop_x'),
            ("Y", s[1], cy, '_rs_crop_y'),
            ("Z", s[2], cz, '_rs_crop_z'),
        ):
            tk.Label(frm, text=label, bg="#ebebeb",
                     font=("Helvetica", 8), fg=color).pack(anchor="w")
            rs = RangeSlider(frm, 0, n - 1, 0, n - 1,
                             command=self._crop_update_lines, color=color)
            rs.pack(fill=tk.X)
            setattr(self, attr, rs)

        self._bind_scroll(self._dyn)

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_crop_figure(self):
        assert self.scan is not None
        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")
        img = self.scan.img
        nX, nY, nZ = img.shape
        gs = self._fig.add_gridspec(
            1, 3, wspace=0.03,
            left=0.01, right=0.99, top=0.95, bottom=0.04,
        )
        # Pixel spacings in mm (use actual step, not just total range / nVoxels)
        dx = abs(float(self.scan.x[1] - self.scan.x[0])) if nX > 1 else 1.0
        dy = abs(float(self.scan.y[1] - self.scan.y[0])) if nY > 1 else 1.0
        dz = abs(float(self.scan.z[1] - self.scan.z[0])) if nZ > 1 else 1.0
        crop_aspects = [dy / dz, dx / dz, dx / dy]

        mip_labels = ("MIP along X", "MIP along Y", "MIP along Z")
        self._crop_axes = []
        for i in range(3):
            ax = self._fig.add_subplot(gs[0, i])
            ax.set_facecolor("black")
            ax.set_title(mip_labels[i], color="white", fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            ax.imshow(img.max(axis=i), cmap="gray", aspect=crop_aspects[i], origin="upper")
            self._crop_axes.append(ax)

        # Colour legend: green=X, orange=Y, blue=Z
        lw, a = 1.2, 0.85
        cx, cy, cz = '#00ff99', '#ff9900', '#4499ff'
        # MIP along X (axis=0): shape (nY, nZ) → rows=Y, cols=Z
        # MIP along Y (axis=1): shape (nX, nZ) → rows=X, cols=Z
        # MIP along Z (axis=2): shape (nX, nY) → rows=X, cols=Y
        self._crop_lines = {
            # X bounds: visible in panels 1 (row) and 2 (row)
            'x0_1': self._crop_axes[1].axhline(0,       color=cx, lw=lw, alpha=a, ls='--'),
            'x1_1': self._crop_axes[1].axhline(nX - 1,  color=cx, lw=lw, alpha=a, ls='--'),
            'x0_2': self._crop_axes[2].axhline(0,       color=cx, lw=lw, alpha=a, ls='--'),
            'x1_2': self._crop_axes[2].axhline(nX - 1,  color=cx, lw=lw, alpha=a, ls='--'),
            # Y bounds: visible in panel 0 (row) and panel 2 (col)
            'y0_0': self._crop_axes[0].axhline(0,       color=cy, lw=lw, alpha=a, ls='--'),
            'y1_0': self._crop_axes[0].axhline(nY - 1,  color=cy, lw=lw, alpha=a, ls='--'),
            'y0_2': self._crop_axes[2].axvline(0,       color=cy, lw=lw, alpha=a, ls='--'),
            'y1_2': self._crop_axes[2].axvline(nY - 1,  color=cy, lw=lw, alpha=a, ls='--'),
            # Z bounds: visible in panel 0 (col) and panel 1 (col)
            'z0_0': self._crop_axes[0].axvline(0,       color=cz, lw=lw, alpha=a, ls='--'),
            'z1_0': self._crop_axes[0].axvline(nZ - 1,  color=cz, lw=lw, alpha=a, ls='--'),
            'z0_1': self._crop_axes[1].axvline(0,       color=cz, lw=lw, alpha=a, ls='--'),
            'z1_1': self._crop_axes[1].axvline(nZ - 1,  color=cz, lw=lw, alpha=a, ls='--'),
        }
        self._redraw()

    def _crop_update_lines(self):
        x0, x1 = self._rs_crop_x.get_low(), self._rs_crop_x.get_high()
        y0, y1 = self._rs_crop_y.get_low(), self._rs_crop_y.get_high()
        z0, z1 = self._rs_crop_z.get_low(), self._rs_crop_z.get_high()

        self._crop_lines['x0_1'].set_ydata([x0])
        self._crop_lines['x1_1'].set_ydata([x1])
        self._crop_lines['x0_2'].set_ydata([x0])
        self._crop_lines['x1_2'].set_ydata([x1])
        self._crop_lines['y0_0'].set_ydata([y0])
        self._crop_lines['y1_0'].set_ydata([y1])
        self._crop_lines['y0_2'].set_xdata([y0])
        self._crop_lines['y1_2'].set_xdata([y1])
        self._crop_lines['z0_0'].set_xdata([z0])
        self._crop_lines['z1_0'].set_xdata([z1])
        self._crop_lines['z0_1'].set_xdata([z0])
        self._crop_lines['z1_1'].set_xdata([z1])
        self._canvas.draw_idle()

    def _crop_save(self):
        x0, x1 = self._rs_crop_x.get_low(), self._rs_crop_x.get_high() + 1
        y0, y1 = self._rs_crop_y.get_low(), self._rs_crop_y.get_high() + 1
        z0, z1 = self._rs_crop_z.get_low(), self._rs_crop_z.get_high() + 1
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            messagebox.showerror("Invalid crop",
                                 "Min must be strictly less than Max for each axis.")
            return
        self.scan.crop_index([x0, x1], [y0, y1], [z0, z1])
        self._exit_mode()

    # ================================================================== #
    #  MODE — Define Spline                                               #
    # ================================================================== #

    def _on_mode_spline(self):
        if self.scan is None:
            return
        self._mode = 'spline'
        self._enter_mode("Done", self._spline_save, cancel_cmd=self._spline_cancel)
        self._build_spline_figure()
        self._build_spline_controls()
        self._spline_editor.set_axis(0)

    # ── Left panel controls ──────────────────────────────────────────────

    def _build_spline_controls(self):
        self._clear_dyn()

        frm_ax = self._lframe("View axis")
        row = tk.Frame(frm_ax, bg="#ebebeb")
        row.pack()
        self._spline_axis_btns = {}
        for i, label in enumerate(('X', 'Y', 'Z')):
            b = tk.Button(row, text=label, width=5,
                          command=lambda a=i: self._spline_set_axis(a))
            b.pack(side=tk.LEFT, padx=2)
            self._spline_axis_btns[i] = b

        frm_pts = self._lframe("Points")
        self._spline_count_var = tk.StringVar(value="Points: 0")
        tk.Label(frm_pts, textvariable=self._spline_count_var,
                 bg="#ebebeb", font=("Courier", 8)).pack(anchor="w")
        tk.Button(frm_pts, text="Remove last",
                  command=self._spline_remove_last).pack(fill=tk.X, pady=2)
        tk.Button(frm_pts, text="Clear all",
                  command=self._spline_clear_all).pack(fill=tk.X, pady=1)

        frm_step = self._lframe("Export STEP")
        self._sl_step_radius = self._slider(frm_step, "Tube radius (mm)", 1, 50, 5)
        tk.Button(frm_step, text="To STEP…",
                  command=self._spline_export_step).pack(fill=tk.X, pady=2)

        self._spline_refresh_axis_colors()
        self._bind_scroll(self._dyn)

    def _spline_set_axis(self, axis: int):
        self._spline_editor.set_axis(axis)
        self._spline_refresh_axis_colors()
        self._spline_update_nav_label()

    def _spline_cycle_axis(self, delta: int):
        new_axis = (self._spline_editor._axis + delta) % 3
        self._spline_set_axis(new_axis)

    def _spline_refresh_axis_colors(self):
        current = self._spline_editor._axis
        colors = {0: '#90ee90', 1: '#add8e6', 2: '#ffcc88'}
        for i, b in self._spline_axis_btns.items():
            b.config(bg=colors[i] if i == current else 'SystemButtonFace')

    def _spline_update_nav_label(self):
        labels = {0: 'View along X', 1: 'View along Y', 2: 'View along Z'}
        self._spline_nav_text.set_text(labels[self._spline_editor._axis])
        self._canvas.draw_idle()

    def _spline_remove_last(self):
        pts = self._spline_editor._points
        if pts:
            pts.pop()
            self._spline_editor.redraw()

    def _spline_clear_all(self):
        self._spline_editor._points.clear()
        self._spline_editor.redraw()

    def _spline_export_step(self):
        from spline_to_step import spline_to_step
        result = self._spline_editor.get_spline()
        if result is None:
            messagebox.showwarning("Export STEP", "Need at least 2 points to export.")
            return
        tck, u = result
        n_pts = max(100, len(self._spline_editor._points) * 20)
        u_fine = np.linspace(u[0], u[-1], n_pts)
        xi, yi, zi = splev(u_fine, tck)
        # Convert voxel indices → physical mm using the scan coordinate arrays
        assert self.scan is not None
        ix = np.arange(len(self.scan.x), dtype=float)
        iy = np.arange(len(self.scan.y), dtype=float)
        iz = np.arange(len(self.scan.z), dtype=float)
        pts_mm = list(zip(
            np.interp(xi, ix, self.scan.x.astype(float)),
            np.interp(yi, iy, self.scan.y.astype(float)),
            np.interp(zi, iz, self.scan.z.astype(float)),
        ))
        outpath = filedialog.asksaveasfilename(
            title="Save STEP file",
            defaultextension=".step",
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
            initialfile="spline.step",
        )
        if not outpath:
            return
        radius = float(self._sl_step_radius.get())
        try:
            self._status_var.set("Exporting STEP…")
            self.root.update_idletasks()
            spline_to_step(pts_mm, outpath, radius=radius, downsample=4)
            self._status_var.set(f"STEP saved → {os.path.basename(outpath)}")
        except Exception as exc:
            self._status_var.set(f"STEP export error: {exc}")
            messagebox.showerror("STEP export error", str(exc))

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_spline_figure(self):
        from matplotlib.widgets import Button as MplButton

        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")

        # Main image axes — leave bottom 10 % for the navigation bar
        self._ax_spline = self._fig.add_axes([0.01, 0.11, 0.98, 0.87])

        # Navigation bar background
        ax_nav = self._fig.add_axes([0.0, 0.0, 1.0, 0.10])
        ax_nav.set_facecolor("#1a1a1a")
        ax_nav.axis('off')

        # ◄ and ► buttons
        ax_prev = self._fig.add_axes([0.10, 0.015, 0.14, 0.065])
        ax_next = self._fig.add_axes([0.76, 0.015, 0.14, 0.065])
        ax_lbl  = self._fig.add_axes([0.35, 0.015, 0.30, 0.065])

        self._spline_btn_prev = MplButton(ax_prev, '◄  Prev',
                                          color='#222222', hovercolor='#3a3a3a')
        self._spline_btn_next = MplButton(ax_next, 'Next  ►',
                                          color='#222222', hovercolor='#3a3a3a')
        for btn in (self._spline_btn_prev, self._spline_btn_next):
            btn.label.set_color('white')
            btn.label.set_fontsize(10)

        ax_lbl.set_facecolor("#1a1a1a")
        ax_lbl.axis('off')
        self._spline_nav_text = ax_lbl.text(
            0.5, 0.5, 'View along X',
            ha='center', va='center',
            color='white', fontsize=11, fontweight='bold',
            transform=ax_lbl.transAxes,
        )

        self._spline_btn_prev.on_clicked(lambda _: self._spline_cycle_axis(-1))
        self._spline_btn_next.on_clicked(lambda _: self._spline_cycle_axis(+1))

        # Build editor and wire the point-count callback
        self._spline_editor = SplineEditor(
            self._fig, self._ax_spline, self.scan,
            on_changed=lambda n: self._spline_count_var.set(f"Points: {n}"),
        )
        self._spline_editor.connect()

    # ── Save / cancel ────────────────────────────────────────────────────

    def _spline_save(self):
        self._spline_editor.disconnect()
        self._spline_btn_prev.disconnect_events()
        self._spline_btn_next.disconnect_events()
        self.scan.spline_points = self._spline_editor.get_points()
        self.scan.spline        = self._spline_editor.get_spline()
        self._export_spline_to_file()
        self._exit_mode()

    def _export_spline_to_file(self):
        spline = self.scan.spline
        if spline is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save spline points",
        )
        if not path:
            return
        tck, u = spline
        u_fine = np.linspace(u[0], u[-1], max(500, len(u) * 10))
        x, y, z = splev(u_fine, tck)
        data = np.column_stack([x, y, z])
        np.savetxt(path, data, delimiter=",", header="x,y,z", comments="")

    def _spline_cancel(self):
        self._spline_editor.disconnect()
        self._spline_btn_prev.disconnect_events()
        self._spline_btn_next.disconnect_events()
        self._exit_mode()

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    ScannerApp().run()
