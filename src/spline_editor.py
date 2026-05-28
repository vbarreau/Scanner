"""
SplineEditor — interactive 3D spline definition inside an embedded matplotlib axes.

The editor operates on a single-axis MIP view at a time.
Volume shape convention: (nX, nY, nZ).

View axis → image mapping
  axis 0 (view along X): imshow(vol.max(0).T, origin='lower')  col=Y, row=Z
  axis 1 (view along Y): imshow(vol.max(1).T, origin='lower')  col=X, row=Z
  axis 2 (view along Z): imshow(vol.max(2),   origin='upper')  col=Y, row=X
"""

import numpy as np
from scipy.interpolate import splprep, splev


# view_axis → (col_array_idx, row_array_idx, depth_array_idx)
_COORD_MAP = {
    0: (1, 2, 0),
    1: (0, 2, 1),
    2: (1, 0, 2),
}

_VIEW_TITLES = {
    0: 'View along X',
    1: 'View along Y',
    2: 'View along Z',
}


class SplineEditor:
    """
    Interactive 3D spline editor embedded in a single matplotlib axes.

    Usage:
        editor = SplineEditor(fig, ax, scan)
        editor.connect()
        editor.set_axis(0)         # show first view
        ...
        tck_u = editor.get_spline()
        editor.disconnect()
    """

    def __init__(self, fig, ax, scan, on_changed=None):
        """
        Parameters
        ----------
        fig       : matplotlib Figure
        ax        : matplotlib Axes (the single image axes)
        scan      : Image3D — provides .img, .x, .y, .z
        on_changed: callable(n_points) — called on every redraw
        """
        self._fig = fig
        self._ax = ax
        self._scan = scan
        self._on_changed = on_changed

        self._axis = 0
        self._points = []      # list of [xi, yi, zi] as floats (voxel indices)
        self._drag_idx = None
        self._cids = []

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self):
        self._cids = [
            self._fig.canvas.mpl_connect('button_press_event',   self._on_press),
            self._fig.canvas.mpl_connect('motion_notify_event',  self._on_motion),
            self._fig.canvas.mpl_connect('button_release_event', self._on_release),
        ]

    def disconnect(self):
        for cid in self._cids:
            self._fig.canvas.mpl_disconnect(cid)
        self._cids = []

    # ------------------------------------------------------------------ #
    #  Axis switching                                                      #
    # ------------------------------------------------------------------ #

    def set_axis(self, axis: int):
        self._axis = axis
        self.redraw()

    # ------------------------------------------------------------------ #
    #  Coordinate helpers                                                  #
    # ------------------------------------------------------------------ #

    def _mip_data(self):
        """Return (data_2d, origin, aspect) for the current view axis."""
        vol = self._scan.img
        x, y, z = self._scan.x, self._scan.y, self._scan.z
        x_range = abs(float(x[-1] - x[0])) or 1.0
        y_range = abs(float(y[-1] - y[0])) or 1.0
        z_range = abs(float(z[-1] - z[0])) or 1.0
        if self._axis == 0:
            return vol.max(axis=0).T, 'lower', z_range / y_range
        elif self._axis == 1:
            return vol.max(axis=1).T, 'lower', z_range / x_range
        else:
            return vol.max(axis=2),   'upper', y_range / x_range

    def _pt_to_display(self, pt):
        """[x,y,z] voxel → (col, row) in imshow data coordinates."""
        ci, ri, _ = _COORD_MAP[self._axis]
        return pt[ci], pt[ri]

    def _display_to_pt(self, col, row, depth_from_idx=None):
        """(col, row) → [x,y,z] voxel; depth coord taken from existing point or defaults to centre."""
        ci, ri, di = _COORD_MAP[self._axis]
        coords = [0.0, 0.0, 0.0]
        coords[ci] = float(col)
        coords[ri] = float(row)
        if depth_from_idx is not None and depth_from_idx < len(self._points):
            coords[di] = self._points[depth_from_idx][di]
        else:
            coords[di] = float(self._scan.img.shape[di] // 2)
        return coords

    def _clamp(self, pt):
        s = self._scan.img.shape
        return [max(0.0, min(float(s[i] - 1), pt[i])) for i in range(3)]

    def _hit_test(self, col, row, threshold=12):
        """Return index of nearest point within threshold px, or None."""
        best, best_d = None, float(threshold)
        for i, pt in enumerate(self._points):
            pc, pr = self._pt_to_display(pt)
            d = ((pc - col) ** 2 + (pr - row) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, i
        return best

    # ------------------------------------------------------------------ #
    #  Event handlers                                                      #
    # ------------------------------------------------------------------ #

    def _on_press(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        col, row = event.xdata, event.ydata
        idx = self._hit_test(col, row)
        if idx is not None:
            self._drag_idx = idx
        else:
            self._points.append(self._clamp(self._display_to_pt(col, row)))
            self._drag_idx = len(self._points) - 1
        self.redraw()

    def _on_motion(self, event):
        if self._drag_idx is None:
            return
        if event.inaxes is not self._ax or event.xdata is None:
            return
        updated = self._display_to_pt(event.xdata, event.ydata, self._drag_idx)
        self._points[self._drag_idx] = self._clamp(updated)
        self.redraw()

    def _on_release(self, event):
        self._drag_idx = None

    # ------------------------------------------------------------------ #
    #  Drawing                                                             #
    # ------------------------------------------------------------------ #

    def redraw(self):
        ax = self._ax
        ax.clear()
        ax.set_facecolor('black')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')

        data, origin, aspect = self._mip_data()
        vmax = float(data.max()) if data.size else 1.0
        ax.imshow(data, cmap='gray', origin=origin, aspect=aspect,
                  vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_title(_VIEW_TITLES[self._axis], color='white', fontsize=10, pad=4)

        # Smooth spline curve in this 2D projection
        if len(self._points) >= 2:
            try:
                pts = np.array(self._points)
                ci, ri, _ = _COORD_MAP[self._axis]
                cols_arr = pts[:, ci]
                rows_arr = pts[:, ri]
                k = min(3, len(self._points) - 1)
                tck, _ = splprep([cols_arr, rows_arr], s=0, k=k)
                u_fine = np.linspace(0, 1, 300)
                c_fine, r_fine = splev(u_fine, tck)
                ax.plot(c_fine, r_fine, color='#00aaff', lw=1.5, alpha=0.85, zorder=4)
            except Exception:
                pass

        # Control points with index labels
        for i, pt in enumerate(self._points):
            col, row = self._pt_to_display(pt)
            fc = '#ff4444' if i == self._drag_idx else '#ffdd00'
            ax.plot(col, row, 'o', color=fc, ms=8, mew=1.5,
                    markeredgecolor='white', zorder=5)
            ax.text(col + 4, row - 4, str(i + 1), color='white',
                    fontsize=7, zorder=6, va='top')

        if self._on_changed is not None:
            self._on_changed(len(self._points))

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Public output                                                       #
    # ------------------------------------------------------------------ #

    def get_points(self):
        """Return a copy of the control points as list of [x, y, z] voxel indices."""
        return [pt[:] for pt in self._points]

    def get_spline(self):
        """
        Compute a 3D parametric spline from the control points.
        Returns (tck, u) from scipy.interpolate.splprep, or None if < 2 points.
        """
        if len(self._points) < 2:
            return None
        pts = np.array(self._points, dtype=float)
        try:
            k = min(3, len(pts) - 1)
            tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=0, k=k)
            return tck, u
        except Exception:
            return None
