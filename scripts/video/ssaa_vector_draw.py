"""True 4x Supersampled Anti-Aliased Vector Drawing Engine (`SSAAVectorDraw`) for Pillow.

Eliminates all 1-bit Bresenham stair-stepping / jagged edges on rounded rectangles, pills,
glass card borders, circles, polygons, and connector lines by rasterizing shape primitives on a
tight 4x-supersampled ROI patch and downsampling with area-weighted LANCZOS/BOX filtering.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

_ORIG_DRAW_CLS = ImageDraw.ImageDraw
_ORIG_DRAW_FN = ImageDraw.Draw
SSAA_SCALE = 4


def _norm_rgba(col, default_alpha: int = 255) -> tuple[int, int, int, int] | None:
    if col is None:
        return None
    if isinstance(col, int):
        return (col, col, col, default_alpha)
    if len(col) == 3:
        return (int(col[0]), int(col[1]), int(col[2]), default_alpha)
    return (int(col[0]), int(col[1]), int(col[2]), int(col[3]))


def _flatten_xy(xy) -> list[tuple[float, float]]:
    if not xy:
        return []
    if isinstance(xy[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in xy]
    return [(float(xy[i]), float(xy[i + 1])) for i in range(0, len(xy), 2)]


class SSAAVectorDraw:
    """Drop-in replacement for `PIL.ImageDraw.ImageDraw` that renders geometric primitives
    (`rounded_rectangle`, `ellipse`, `line`, `polygon`) at 4x supersampled resolution (`SSAA 4x`).
    """

    def __init__(self, im: Image.Image, mode: str | None = None) -> None:
        self._im = im
        self._raw = _ORIG_DRAW_CLS(im, mode)

    def __getattr__(self, name: str):
        return getattr(self._raw, name)

    def _composite_4x_patch(self, rx0: int, ry0: int, rw: int, rh: int, draw_cb) -> None:
        if rw <= 0 or rh <= 0:
            return
        s = SSAA_SCALE
        if self._im.mode == "L":
            patch_4x = Image.new("L", (rw * s, rh * s), 0)
            raw_4x = _ORIG_DRAW_CLS(patch_4x)
            draw_cb(raw_4x, s, is_l=True)
            patch_1x = patch_4x.resize((rw, rh), Image.Resampling.LANCZOS)
            # Max-blend onto L mask so anti-aliased edge feathers cleanly
            cur_crop = self._im.crop((rx0, ry0, rx0 + rw, ry0 + rh))
            from PIL import ImageChops
            self._im.paste(ImageChops.lighter(cur_crop, patch_1x), (rx0, ry0))
            return

        patch_4x = Image.new("RGBA", (rw * s, rh * s), (0, 0, 0, 0))
        raw_4x = _ORIG_DRAW_CLS(patch_4x)
        draw_cb(raw_4x, s, is_l=False)
        patch_1x = patch_4x.resize((rw, rh), Image.Resampling.LANCZOS)
        if self._im.mode == "RGBA":
            self._im.alpha_composite(patch_1x, dest=(rx0, ry0))
        else:
            self._im.paste(patch_1x.convert(self._im.mode), (rx0, ry0), patch_1x.split()[3])

    def rounded_rectangle(self, xy, radius: float = 0, fill=None, outline=None, width: int = 1, corners=None) -> None:
        pts = _flatten_xy(xy)
        if len(pts) < 2:
            return
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        W, H = self._im.size
        pad = max(4, int(width) + 3)
        rx0 = max(0, int(x0) - pad)
        ry0 = max(0, int(y0) - pad)
        rx1 = min(W, int(round(x1)) + pad + 1)
        ry1 = min(H, int(round(y1)) + pad + 1)
        rw, rh = rx1 - rx0, ry1 - ry0

        def _cb(raw_4x, s: int, is_l: bool) -> None:
            bx0 = int(round((x0 - rx0) * s))
            by0 = int(round((y0 - ry0) * s))
            bx1 = int(round((x1 - rx0) * s))
            by1 = int(round((y1 - ry0) * s))
            f_col = fill if is_l else _norm_rgba(fill)
            o_col = outline if is_l else _norm_rgba(outline)
            raw_4x.rounded_rectangle(
                [bx0, by0, bx1, by1],
                radius=int(round(radius * s)),
                fill=f_col,
                outline=o_col,
                width=max(1, int(round(width * s))) if o_col is not None else 0,
                corners=corners,
            )

        self._composite_4x_patch(rx0, ry0, rw, rh, _cb)

    def ellipse(self, xy, fill=None, outline=None, width: int = 1) -> None:
        pts = _flatten_xy(xy)
        if len(pts) < 2:
            return
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        W, H = self._im.size
        pad = max(4, int(width) + 3)
        rx0 = max(0, int(x0) - pad)
        ry0 = max(0, int(y0) - pad)
        rx1 = min(W, int(round(x1)) + pad + 1)
        ry1 = min(H, int(round(y1)) + pad + 1)
        rw, rh = rx1 - rx0, ry1 - ry0

        def _cb(raw_4x, s: int, is_l: bool) -> None:
            bx0 = int(round((x0 - rx0) * s))
            by0 = int(round((y0 - ry0) * s))
            bx1 = int(round((x1 - rx0) * s))
            by1 = int(round((y1 - ry0) * s))
            f_col = fill if is_l else _norm_rgba(fill)
            o_col = outline if is_l else _norm_rgba(outline)
            raw_4x.ellipse(
                [bx0, by0, bx1, by1],
                fill=f_col,
                outline=o_col,
                width=max(1, int(round(width * s))) if o_col is not None else 0,
            )

        self._composite_4x_patch(rx0, ry0, rw, rh, _cb)

    def line(self, xy, fill=None, width: int = 1, joint=None) -> None:
        pts = _flatten_xy(xy)
        if len(pts) < 2:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        W, H = self._im.size
        pad = max(4, int(width) + 3)
        rx0 = max(0, int(min(xs)) - pad)
        ry0 = max(0, int(min(ys)) - pad)
        rx1 = min(W, int(round(max(xs))) + pad + 1)
        ry1 = min(H, int(round(max(ys))) + pad + 1)
        rw, rh = rx1 - rx0, ry1 - ry0

        def _cb(raw_4x, s: int, is_l: bool) -> None:
            pts_4x = [(int(round((px - rx0) * s)), int(round((py - ry0) * s))) for px, py in pts]
            f_col = fill if is_l else _norm_rgba(fill)
            raw_4x.line(pts_4x, fill=f_col, width=max(1, int(round(width * s))), joint=joint)

        self._composite_4x_patch(rx0, ry0, rw, rh, _cb)

    def polygon(self, xy, fill=None, outline=None, width: int = 1) -> None:
        pts = _flatten_xy(xy)
        if len(pts) < 3:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        W, H = self._im.size
        pad = max(4, int(width) + 3)
        rx0 = max(0, int(min(xs)) - pad)
        ry0 = max(0, int(min(ys)) - pad)
        rx1 = min(W, int(round(max(xs))) + pad + 1)
        ry1 = min(H, int(round(max(ys))) + pad + 1)
        rw, rh = rx1 - rx0, ry1 - ry0

        def _cb(raw_4x, s: int, is_l: bool) -> None:
            pts_4x = [(int(round((px - rx0) * s)), int(round((py - ry0) * s))) for px, py in pts]
            f_col = fill if is_l else _norm_rgba(fill)
            o_col = outline if is_l else _norm_rgba(outline)
            raw_4x.polygon(
                pts_4x,
                fill=f_col,
                outline=o_col,
                width=max(1, int(round(width * s))) if o_col is not None else 1,
            )

        self._composite_4x_patch(rx0, ry0, rw, rh, _cb)


def ssaa_draw_glass_card(
    base_rgba: Image.Image,
    box: list[int],
    radius: int,
    border_col: tuple[int, int, int, int] = (0, 129, 251, 85),
    shadow_col: tuple[int, int, int, int] = (0, 80, 180, 22),
    shadow_blur: int = 14,
    shadow_offset_y: int = 7,
    fill_col: tuple[int, int, int, int] = (255, 255, 255, 248),
    border_w: int = 2,
) -> None:
    """Render an elevated glass card with a soft Gaussian drop shadow and a 4x-supersampled
    anti-aliased rounded border (`SSAA 4x`).
    """
    W, H = base_rgba.size
    x0, y0, x1, y1 = box
    pad = shadow_blur * 2 + 12
    rx0 = max(0, x0 - pad)
    ry0 = max(0, y0 - pad)
    rx1 = min(W, x1 + pad)
    ry1 = min(H, y1 + pad + shadow_offset_y)
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw <= 4 or rh <= 4:
        return

    sh_roi = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    sh_d = _ORIG_DRAW_CLS(sh_roi)
    sh_d.rounded_rectangle(
        [x0 - rx0 + 2, y0 - ry0 + shadow_offset_y, x1 - rx0 - 2, y1 - ry0 + shadow_offset_y + 4],
        radius=radius,
        fill=_norm_rgba(shadow_col),
    )
    sh_roi = sh_roi.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

    # Draw card body & border using 4x supersampling on the tight card region
    SSAAVectorDraw(sh_roi).rounded_rectangle(
        [x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0],
        radius=radius,
        fill=_norm_rgba(fill_col),
        outline=_norm_rgba(border_col),
        width=border_w,
    )
    base_rgba.alpha_composite(sh_roi, dest=(rx0, ry0))


def install_ssaa_vector_draw() -> None:
    """Monkey-patch `PIL.ImageDraw.Draw` globally so all vector primitives are 4x supersampled."""
    def _patched_draw(im: Image.Image, mode: str | None = None):
        return SSAAVectorDraw(im, mode)

    ImageDraw.Draw = _patched_draw
