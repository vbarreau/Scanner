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
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import cv2

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import Image as ImageModule  # noqa: E402
from Image import show_slice  # noqa: E402
from func import clip as func_clip  # noqa: E402

# ── Window geometry ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 1300, 780
LEFT_W = 270
FIG_DPI = 100


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
        self.root.resizable(False, False)

        self._build_layout()
        self._show_welcome()

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    def _build_layout(self):
        # ── Left panel ──────────────────────────────────────────────────
        self._left = tk.Frame(self.root, width=LEFT_W, bg="#ebebeb",
                              relief=tk.FLAT, bd=0)
        self._left.pack(side=tk.LEFT, fill=tk.Y)
        self._left.pack_propagate(False)

        # ── Right panel: embedded matplotlib canvas ──────────────────────
        right = tk.Frame(self.root, bg="#111111")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig_w = (WIN_W - LEFT_W) / FIG_DPI
        fig_h = WIN_H / FIG_DPI
        self._fig = Figure(figsize=(fig_w, fig_h), dpi=FIG_DPI,
                           facecolor="#111111")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        self._progress = ttk.Progressbar(L, mode='indeterminate', length=230)
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
        self._action_buttons = [self._btn_plot, self._btn_rot, self._btn_anim, self._btn_proj]
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
        self._progress.start(10)

        def _worker():
            try:
                scan = ImageModule.Image3D(folder)
                self.root.after(0, lambda: self._on_load_done(scan))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_load_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_load_done(self, scan: ImageModule.Image3D):
        self.scan = scan
        self._loading = False
        self._progress.stop()
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
        scan = self.scan
        mid = scan.shape[0] // 2
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#111111")
        show_slice(ax, scan.img, mid, self.scan.normal_axis, scan.x, scan.y, scan.z)
        self._redraw()

    def _on_load_error(self, exc: Exception):
        self._loading = False
        self._progress.stop()
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
        self._sl_clip_min = self._slider(frm_pr, "Clip min",
                                         img_min, img_max, img_min)
        self._sl_clip_max = self._slider(frm_pr, "Clip max",
                                         img_min, img_max, img_max)
        tk.Button(frm_pr, text="Apply clip",
                  command=self._plot_apply_clip).pack(fill=tk.X, pady=1)

        self._bind_scroll(self._dyn)

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_plot_figure(self):
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

        self._im_px = self._ax_px.imshow(
            img[ix, :, :].T, cmap='gray', aspect='auto', origin='upper')
        self._im_py = self._ax_py.imshow(
            img[:, iy, :].T, cmap='gray', aspect='auto', origin='upper')
        self._im_pz = self._ax_pz.imshow(
            img[:, :, iz], cmap='gray', aspect='auto', origin='upper')

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
        self._redraw()

    def _plot_update_slices(self):
        if not hasattr(self, '_im_px'):
            return
        ix, iy, iz = int(self._sl_x.get()), int(self._sl_y.get()), int(self._sl_z.get())
        img = self.scan.img
        self._im_px.set_data(img[ix, :, :].T)
        self._im_py.set_data(img[:, iy, :].T)
        self._im_pz.set_data(img[:, :, iz])
        self._chl_px[0].set_xdata([iy])
        self._chl_px[1].set_ydata([iz])
        self._chl_py[0].set_xdata([ix])
        self._chl_py[1].set_ydata([iz])
        self._chl_pz[0].set_ydata([ix])
        self._chl_pz[1].set_xdata([iy])
        self._canvas.draw_idle()

    def _plot_draw_hist(self):
        ax = self._ax_hist
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.hist(self.scan.img.flatten(), bins=256, color='#4488ff',
                alpha=0.85, log=True)
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    def _plot_apply_contrast(self):
        self.scan.change_contrast(self._sl_contrast.get())
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_apply_compress(self):
        self.scan.compress(self._sl_seuil.get(), 0.8)
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_apply_clip(self):
        self.scan.img = func_clip(self.scan.img,
                                  self._sl_clip_min.get(),
                                  self._sl_clip_max.get())
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_reset(self):
        self.scan.img = self._plot_img_backup.copy()
        self._plot_update_slices()
        self._plot_draw_hist()
        self._redraw()

    def _plot_cancel(self):
        """Reset the image to its original state then exit the mode."""
        self.scan.img = self._plot_img_backup.copy()
        self._exit_mode()

    def _plot_save(self):
        self._exit_mode()

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
        tk.Button(frm_btn, text="Preview frame",
                  command=self._anim_preview).pack(fill=tk.X, pady=2)

        self._anim_refresh_colors()
        self._bind_scroll(self._dyn)

    def _anim_set(self, key: str, val: str):
        self._anim[key] = val
        self._anim_refresh_colors()
        if key == 'axis':
            # Refresh MIP border highlights
            self._build_anim_figure()

    def _anim_refresh_colors(self):
        for a, b in self._anim_ax_btns.items():
            b.config(bg='#90ee90' if a == self._anim['axis'] else 'SystemButtonFace')
        for a, b in self._anim_pov_btns.items():
            b.config(bg='#add8e6' if a == self._anim.get('pov', 'x') else 'SystemButtonFace')

    # ── Right canvas ─────────────────────────────────────────────────────

    def _build_anim_figure(self):
        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")
        mip = {
            'x': self.scan.img.max(axis=0),
            'y': self.scan.img.max(axis=1),
            'z': self.scan.img.max(axis=2),
        }
        gs = self._fig.add_gridspec(
            2, 3, height_ratios=[3, 2],
            hspace=0.12, wspace=0.03,
            left=0.01, right=0.99, top=0.97, bottom=0.03,
        )
        self._anim_mip_ax = {}
        for i, key in enumerate(('x', 'y', 'z')):
            ax = self._fig.add_subplot(gs[0, i])
            ax.imshow(mip[key], cmap='gray', aspect='auto')
            ax.set_title(f"View along {key.upper()}", color='white', fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            color = '#00cc44' if key == self._anim['axis'] else '#333333'
            lw = 3 if key == self._anim['axis'] else 1
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(lw)
            self._anim_mip_ax[key] = ax

        # Preview area (bottom row spans all 3 columns)
        self._ax_anim_prev = self._fig.add_subplot(gs[1, :])
        self._ax_anim_prev.set_facecolor("#0d0d0d")
        self._ax_anim_prev.tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
        self._ax_anim_prev.text(
            0.5, 0.5, 'Click "Preview frame" to render a vispy frame.',
            ha='center', va='center', color='#666666',
            fontsize=11, transform=self._ax_anim_prev.transAxes,
        )
        for sp in self._ax_anim_prev.spines.values():
            sp.set_edgecolor('#333333')
        self._redraw()

    def _anim_preview(self):
        from vispy import scene
        self._status_var.set("Rendering preview…")
        self.root.update()
        try:
            img_r = self.scan.img.astype(np.float32)
            img_r = (img_r - img_r.min()) / (img_r.max() - img_r.min() + 1e-8)
            axis = self._anim['axis']
            if axis == 'x':
                img_r = np.rot90(img_r, k=1, axes=(0, 2))
            elif axis == 'y':
                img_r = np.rot90(img_r, k=1, axes=(1, 2))
            method = 'additive' if self._anim['grad'] else 'mip'
            _pov_map = {'x': (0, 0), 'y': (90, 0), 'z': (0, 90)}
            az, el = _pov_map[self._anim.get('pov', 'x')]
            canvas = scene.SceneCanvas(size=(600, 360), show=True, bgcolor='black')
            canvas.show(False)
            view = canvas.central_widget.add_view()
            scene.visuals.Volume(img_r, parent=view.scene,
                                 method=method, cmap='grays', clim=(0, 1))
            cam = scene.cameras.TurntableCamera(fov=0, elevation=el, azimuth=az)
            view.camera = cam
            cam.set_range()
            frame = canvas.render(alpha=False)
            canvas.close()

            ax = self._ax_anim_prev
            ax.clear()
            ax.imshow(frame, aspect='auto')
            ax.set_title("Vispy preview", color='white', fontsize=9, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            self._redraw()
            self._status_var.set("Preview ready.")
        except Exception as exc:
            self._status_var.set(f"Preview error: {exc}")
            messagebox.showerror("Preview error", str(exc))

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
        try:
            self.scan.rotation_video(
                outpath=outpath, n_frames=n_frames,
                grad=grad, axis=axis, pov=pov,
            )
            self._status_var.set(f"Saved → {os.path.basename(outpath)}")
        except Exception as exc:
            self._status_var.set(f"Render error: {exc}")
            messagebox.showerror("Render error", str(exc))

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
        self._fig.clear()
        self._fig.patch.set_facecolor("#111111")
        projs = [self.scan.projection(ax, grad=self._proj_grad) for ax in range(3)]
        titles = ["Projection X", "Projection Y", "Projection Z"]
        gs = self._fig.add_gridspec(
            1, 3, wspace=0.03,
            left=0.01, right=0.99, top=0.95, bottom=0.04,
        )
        for i, (proj, title) in enumerate(zip(projs, titles)):
            ax = self._fig.add_subplot(gs[0, i])
            ax.imshow(proj, cmap='gray', aspect='auto')
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

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    ScannerApp().run()
