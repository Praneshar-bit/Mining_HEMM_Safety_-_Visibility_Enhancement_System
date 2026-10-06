
import argparse
from pathlib import Path
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


# ============================================================
# NMDC LiDAR DRIVER VIEW + POTHOLE / BUMP DETECTION
#
# Input : KITTI-style .bin  (x,y,z,intensity)
# Output: forward driver-view image
#
# The display does NOT show LiDAR dots and does NOT assign
# object names. Elevated objects are generic solid blocks.
#
# Road defects:
#   - POTHOLE = local depression relative to surrounding road
#   - BUMP    = local elevation relative to surrounding road
#
# Detection is based on measured LiDAR elevation, not a
# pre-drawn pothole.
# ============================================================


def load_bin(path):
    data = np.fromfile(path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError("Expected KITTI-style BIN with x,y,z,intensity.")
    return data.reshape(-1, 4)


def clean_points(points, max_range=65.0):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = (
        (x > 2.0) &
        (x < max_range) &
        (np.abs(y) < 10.0) &
        (z > -4.5) &
        (z < 4.0)
    )
    return points[keep]


def build_ground_grid(points):
    """
    Build a measured elevation grid.
    A low percentile is used per cell so elevated objects
    do not become the road surface.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (np.abs(y) < 9.0) & (z < 1.5)
    p = points[mask]

    if len(p) < 300:
        return None

    x0, x1 = 2.0, min(65.0, float(np.percentile(p[:, 0], 99)))
    y0, y1 = -9.0, 9.0
    dx, dy = 0.35, 0.35

    nx = int((x1 - x0) / dx) + 1
    ny = int((y1 - y0) / dy) + 1

    grid = np.full((ny, nx), np.nan, dtype=np.float32)

    ix = ((p[:, 0] - x0) / dx).astype(int)
    iy = ((p[:, 1] - y0) / dy).astype(int)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, zz = ix[valid], iy[valid], p[:, 2][valid]

    # Robust local low surface.
    for yy in range(ny):
        yy_mask = iy == yy
        if not np.any(yy_mask):
            continue
        for xx in range(nx):
            vals = zz[yy_mask & (ix == xx)]
            if len(vals) >= 2:
                grid[yy, xx] = np.percentile(vals, 20)

    # Fill gaps horizontally and vertically.
    for yy in range(ny):
        row = grid[yy]
        good = np.where(np.isfinite(row))[0]
        if len(good) >= 2:
            grid[yy] = np.interp(np.arange(nx), good, row[good])

    for xx in range(nx):
        col = grid[:, xx]
        good = np.where(np.isfinite(col))[0]
        if len(good) >= 2:
            grid[:, xx] = np.interp(np.arange(ny), good, col[good])

    med = np.nanmedian(grid)
    grid[~np.isfinite(grid)] = med

    # Light smoothing, retaining road defects.
    grid = cv2.GaussianBlur(grid, (5, 5), 0.7)

    return x0, y0, dx, dy, grid


def sample_grid(surface, x, y):
    x0, y0, dx, dy, grid = surface

    ix = (x - x0) / dx
    iy = (y - y0) / dy

    fx = np.floor(ix)
    fy = np.floor(iy)

    ix0 = np.clip(fx.astype(int), 0, grid.shape[1] - 1)
    iy0 = np.clip(fy.astype(int), 0, grid.shape[0] - 1)
    ix1 = np.clip(ix0 + 1, 0, grid.shape[1] - 1)
    iy1 = np.clip(iy0 + 1, 0, grid.shape[0] - 1)

    tx = np.clip(ix - fx, 0, 1)
    ty = np.clip(iy - fy, 0, 1)

    return (
        (1 - tx) * (1 - ty) * grid[iy0, ix0] +
        tx * (1 - ty) * grid[iy0, ix1] +
        (1 - tx) * ty * grid[iy1, ix0] +
        tx * ty * grid[iy1, ix1]
    )


def detect_road_defects(surface):
    """
    Detect local road depressions/elevations using:
        local defect = measured surface - broad local surface

    POTHOLE: negative residual
    BUMP:    positive residual

    Several neighboring cells must agree before a region is accepted.
    """
    x0, y0, dx, dy, grid = surface

    # Broad surface = expected surrounding road.
    broad = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.2, sigmaY=2.2)

    residual = grid - broad

    # Thresholds deliberately conservative.
    pothole_mask = residual < -0.12
    bump_mask = residual > 0.12

    # Remove tiny isolated noise.
    kernel = np.ones((3, 3), np.uint8)
    pothole_mask = cv2.morphologyEx(
        pothole_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    pothole_mask = cv2.morphologyEx(
        pothole_mask, cv2.MORPH_CLOSE, kernel
    )

    bump_mask = cv2.morphologyEx(
        bump_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    bump_mask = cv2.morphologyEx(
        bump_mask, cv2.MORPH_CLOSE, kernel
    )

    defects = []

    def extract(mask, kind):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.18:
                continue

            xpix, ypix, wpix, hpix = cv2.boundingRect(cnt)

            xmin = x0 + xpix * dx
            xmax = x0 + (xpix + wpix) * dx
            ymin = y0 + ypix * dy
            ymax = y0 + (ypix + hpix) * dy

            # Only defects in the forward travel corridor.
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            if cx < 3 or cx > 55 or abs(cy) > 7:
                continue

            yy, xx = np.where(mask > 0)
            inside = (
                (xx >= xpix) & (xx <= xpix + wpix) &
                (yy >= ypix) & (yy <= ypix + hpix)
            )
            if not np.any(inside):
                continue

            vals = residual[yy[inside], xx[inside]]
            if len(vals) < 3:
                continue

            depth = float(abs(np.percentile(vals, 20)))

            # Size from actual LiDAR grid.
            width = max(xmax - xmin, 0.35)
            length = max(ymax - ymin, 0.35)

            # Reject implausibly huge regions.
            if width > 8 or length > 8:
                continue

            defects.append({
                "kind": kind,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "depth": depth,
                "strength": depth * np.sqrt(max(area, 0.1))
            })

    extract(pothole_mask, "POTHOLE")
    extract(bump_mask, "BUMP")

    # Strongest nearby defects first.
    defects.sort(key=lambda d: (
        d["xmin"],
        -d["strength"]
    ))

    # Remove overlapping duplicate regions.
    result = []
    for d in defects:
        overlap = False
        for r in result:
            if d["kind"] != r["kind"]:
                continue
            cx = (d["xmin"] + d["xmax"]) / 2
            cy = (d["ymin"] + d["ymax"]) / 2
            rx = (r["xmin"] + r["xmax"]) / 2
            ry = (r["ymin"] + r["ymax"]) / 2

            if abs(cx - rx) < 1.0 and abs(cy - ry) < 1.0:
                overlap = True
                break

        if not overlap:
            result.append(d)

        if len(result) >= 8:
            break

    return result


def detect_obstacle_blocks(points, surface):
    if surface is None:
        return []

    ground = sample_grid(
        surface,
        points[:, 0],
        points[:, 1]
    )
    relative = points[:, 2] - ground

    mask = (
        (relative > 0.45) &
        (points[:, 0] > 4.0) &
        (points[:, 0] < 60.0) &
        (np.abs(points[:, 1]) < 18)
    )

    p = points[mask]
    if len(p) < 30:
        return []

    if len(p) > 6000:
        idx = np.random.default_rng(4).choice(len(p), 6000, replace=False)
        q = p[idx]
    else:
        q = p

    labels = DBSCAN(eps=0.65, min_samples=12).fit_predict(q[:, :3])

    blocks = []

    for lab in sorted(set(labels)):
        if lab < 0:
            continue

        c = q[labels == lab]
        if len(c) < 15:
            continue

        xmin, xmax = np.percentile(c[:, 0], [2, 98])
        ymin, ymax = np.percentile(c[:, 1], [2, 98])
        zmax = np.percentile(c[:, 2], 98)

        if xmax - xmin < 0.35 or ymax - ymin < 0.25:
            continue
        if xmax - xmin > 15 or ymax - ymin > 10:
            continue

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        base = float(sample_grid(
            surface,
            np.array([cx], dtype=np.float32),
            np.array([cy], dtype=np.float32)
        )[0])

        blocks.append({
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": base,
            "zmax": max(zmax, base + 0.5)
        })

    blocks.sort(key=lambda b: (b["xmax"] - b["xmin"]) *
                              (b["ymax"] - b["ymin"]) *
                              (b["zmax"] - b["zmin"]),
                reverse=True)

    return blocks[:8]


def project(points3d, W, H, fx=735, fy=790, camera_z=1.55):
    # Slightly lower/forward camera gives a more natural driver-eye perspective.
    cx, cy = W * 0.5, H * 0.56

    x = points3d[:, 0]
    y = points3d[:, 1]
    z = points3d[:, 2] - camera_z

    good = x > 0.5

    u = fx * y / np.maximum(x, 0.5) + cx
    v = cy - fy * z / np.maximum(x, 0.5)

    return np.column_stack([u, v]), good


def background(img):
    H, W = img.shape[:2]

    # Smooth atmospheric depth gradient: distant horizon is lighter,
    # foreground is darker. This is still a LiDAR visualization, not
    # a photographic reconstruction.
    top = np.array([30, 33, 38], dtype=np.float32)
    horizon = np.array([72, 77, 84], dtype=np.float32)
    bottom = np.array([22, 25, 29], dtype=np.float32)

    for y in range(H):
        t = y / max(H - 1, 1)

        if t < 0.55:
            a = t / 0.55
            col = top * (1 - a) + horizon * a
        else:
            a = (t - 0.55) / 0.45
            col = horizon * (1 - a) + bottom * a

        img[y, :] = np.clip(col, 0, 255).astype(np.uint8)

    # Very soft horizon haze.
    haze_y = int(H * 0.53)
    haze = np.zeros_like(img, dtype=np.uint8)
    cv2.ellipse(
        haze,
        (W // 2, haze_y),
        (int(W * 0.42), int(H * 0.20)),
        0, 0, 360,
        (45, 50, 56),
        -1
    )
    haze = cv2.GaussianBlur(haze, (0, 0), 35)
    img[:] = cv2.addWeighted(img, 0.86, haze, 0.14, 0)


def draw_road(img, surface, W, H):
    x0, y0, dx, dy, grid = surface

    # Dense lateral samples create a smoother continuous road.
    ys = np.linspace(-9, 9, 90)

    for xi in range(grid.shape[1] - 2, -1, -1):
        xa = x0 + xi * dx
        xb = x0 + (xi + 1) * dx

        if xa < 3 or xa > 65:
            continue

        for j in range(len(ys) - 1):
            ya, yb = ys[j], ys[j + 1]

            za1 = sample_grid(surface, np.array([xa], np.float32),
                              np.array([ya], np.float32))[0]
            za2 = sample_grid(surface, np.array([xa], np.float32),
                              np.array([yb], np.float32))[0]
            zb1 = sample_grid(surface, np.array([xb], np.float32),
                              np.array([ya], np.float32))[0]
            zb2 = sample_grid(surface, np.array([xb], np.float32),
                              np.array([yb], np.float32))[0]

            world = np.array([
                [xa, ya, za1],
                [xa, yb, za2],
                [xb, yb, zb2],
                [xb, ya, zb1]
            ], dtype=np.float32)

            p, good = project(world, W, H)

            if not np.all(good):
                continue

            poly = np.round(p).astype(np.int32)

            # Depth + local relief shading.
            mean_z = float(np.mean(world[:, 2]))
            relief = mean_z - float(np.median(grid))

            # Approximate surface normal from neighboring points.
            dzdx = (zb1 + zb2 - za1 - za2) / max(2 * dx, 1e-3)
            dzdy = (za2 + zb2 - za1 - zb1) / max(2 * (yb - ya), 1e-3)

            # Soft directional lighting.
            nx, ny, nz = -dzdx, -dzdy, 1.0
            norm = max((nx * nx + ny * ny + nz * nz) ** 0.5, 1e-6)
            light = (0.20 * nx + 0.10 * ny + 0.97 * nz) / norm
            light = np.clip(light, 0.55, 1.12)

            # Perspective: foreground gets more texture/contrast,
            # distant road becomes softer and lighter.
            depth_factor = np.clip(1.0 - xa / 70.0, 0.0, 1.0)
            base = 56 + 28 * depth_factor
            shade = int(np.clip(base * light - 20 * relief, 38, 108))

            cv2.fillConvexPoly(
                img, poly,
                (shade, int(shade * 1.01), int(shade * 1.06))
            )

    # Soft road-edge blending. This avoids the previous "cut-out polygon"
    # appearance while keeping the actual measured road geometry.
    overlay = img.copy()

    left = []
    right = []
    for x in np.linspace(5, 60, 70):
        zl = sample_grid(surface, np.array([x], np.float32),
                         np.array([-8.8], np.float32))[0]
        zr = sample_grid(surface, np.array([x], np.float32),
                         np.array([8.8], np.float32))[0]

        pl, ok1 = project(np.array([[x, -8.8, zl]], np.float32), W, H)
        pr, ok2 = project(np.array([[x, 8.8, zr]], np.float32), W, H)

        if ok1[0]:
            left.append(pl[0].astype(int))
        if ok2[0]:
            right.append(pr[0].astype(int))

    if len(left) > 2:
        cv2.polylines(overlay, [np.array(left)], False,
                      (92, 94, 96), 2, cv2.LINE_AA)
    if len(right) > 2:
        cv2.polylines(overlay, [np.array(right)], False,
                      (92, 94, 96), 2, cv2.LINE_AA)

    img[:] = cv2.addWeighted(img, 0.92, overlay, 0.08, 0)


def draw_defect(img, d, surface, W, H):
    cx = (d["xmin"] + d["xmax"]) / 2
    cy = (d["ymin"] + d["ymax"]) / 2

    base = sample_grid(
        surface,
        np.array([cx], np.float32),
        np.array([cy], np.float32)
    )[0]

    if d["kind"] == "POTHOLE":
        depth = min(max(d["depth"], 0.10), 0.42)
        center_z = base - depth
        edge_z = base
        main = (54, 55, 58)
        edge = (112, 115, 118)
    else:
        height = min(max(d["depth"], 0.10), 0.32)
        center_z = base + height
        edge_z = base
        main = (105, 108, 112)
        edge = (142, 145, 148)

    # Use a rounded 8-point footprint rather than a rectangular CAD box.
    rx = max((d["xmax"] - d["xmin"]) * 0.50, 0.3)
    ry = max((d["ymax"] - d["ymin"]) * 0.50, 0.3)

    world = []
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        world.append([
            cx + rx * np.cos(a),
            cy + ry * np.sin(a),
            edge_z
        ])

    edge_world = np.array(world, dtype=np.float32)
    center_world = np.array([[cx, cy, center_z]], dtype=np.float32)

    pe, ok = project(edge_world, W, H)
    pc, okc = project(center_world, W, H)

    if not np.all(ok) or not okc[0]:
        return

    # Triangular shaded facets create a gentle 3D depression/elevation.
    overlay = img.copy()
    center = pc[0]

    for i in range(len(pe)):
        poly = np.array([
            pe[i],
            pe[(i + 1) % len(pe)],
            center
        ], dtype=np.int32)

        cv2.fillConvexPoly(
            overlay, poly,
            tuple(int(v) for v in main)
        )

    img[:] = cv2.addWeighted(img, 0.62, overlay, 0.38, 0)

    border = np.round(pe).astype(np.int32)
    cv2.polylines(
        img, [border], True, edge, 1, cv2.LINE_AA
    )

    # Keep the warning marker tiny and generic.
    if d["xmin"] < 25:
        cv2.circle(
            img,
            tuple(np.round(center).astype(int)),
            6,
            (55, 55, 220) if d["kind"] == "POTHOLE"
            else (55, 170, 220),
            -1,
            cv2.LINE_AA
        )


def draw_block(img, b, W, H):
    v = np.array([
        [b["xmin"], b["ymin"], b["zmin"]],
        [b["xmax"], b["ymin"], b["zmin"]],
        [b["xmax"], b["ymax"], b["zmin"]],
        [b["xmin"], b["ymax"], b["zmin"]],
        [b["xmin"], b["ymin"], b["zmax"]],
        [b["xmax"], b["ymin"], b["zmax"]],
        [b["xmax"], b["ymax"], b["zmax"]],
        [b["xmin"], b["ymax"], b["zmax"]],
    ], dtype=np.float32)

    p, good = project(v, W, H)
    if not np.all(good):
        return

    # Draw farther surfaces first. Subtle translucency makes the
    # geometry feel integrated into the scene instead of like a CAD box.
    faces = [
        ([0, 1, 5, 4], (102, 106, 111)),
        ([1, 2, 6, 5], (118, 121, 126)),
        ([2, 3, 7, 6], (91, 95, 100)),
        ([3, 0, 4, 7], (82, 86, 91)),
        ([4, 5, 6, 7], (135, 139, 144)),
    ]

    overlay = img.copy()

    for face, color in faces:
        poly = np.round(p[face]).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, color)

    img[:] = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

    # Very subtle outline, not a bright wireframe.
    for face, _ in faces:
        poly = np.round(p[face]).astype(np.int32)
        cv2.polylines(img, [poly], True, (160, 164, 168), 1,
                      cv2.LINE_AA)


def render(points, output):
    W, H = 1280, 720

    surface = build_ground_grid(points)

    if surface is None:
        raise RuntimeError(
            "Not enough road/ground LiDAR points to build a surface."
        )

    defects = detect_road_defects(surface)
    blocks = detect_obstacle_blocks(points, surface)

    img = np.zeros((H, W, 3), np.uint8)
    background(img)

    # Solid measured road.
    draw_road(img, surface, W, H)

    # Actual measured road defects.
    for d in sorted(defects, key=lambda q: q["xmin"], reverse=True):
        draw_defect(img, d, surface, W, H)

    # Generic solid obstacle blocks.
    for b in sorted(blocks, key=lambda q: q["xmin"], reverse=True):
        draw_block(img, b, W, H)

    # Center driving guide.
    guide = np.array([
        [W * 0.50, H * 0.98],
        [W * 0.50, H * 0.78],
        [W * 0.50, H * 0.62],
        [W * 0.50, H * 0.52],
    ], np.int32)

    cv2.polylines(
        img, [guide], False,
        (80, 225, 110), 3, cv2.LINE_AA
    )

    # Very subtle foreground darkening helps the road feel closer to
    # a camera view while leaving the LiDAR geometry unchanged.
    vignette = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    v = np.clip(1.0 - 0.10 * (nx * nx + ny * ny), 0.86, 1.0)
    img[:] = np.clip(img.astype(np.float32) * v[..., None], 0, 255).astype(np.uint8)

    # Only show generic warning symbol.
    near_defect = any(
        d["xmin"] < 20 for d in defects
    )
    if near_defect:
        cv2.circle(img, (W - 75, 70), 24,
                   (40, 40, 225), -1, cv2.LINE_AA)
        cv2.putText(
            img, "!",
            (W - 83, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 3,
            cv2.LINE_AA
        )

    cv2.imwrite(str(output), img)

    print(f"Road defects detected: {len(defects)}")
    for d in defects:
        print(
            f'{d["kind"]}: x={d["xmin"]:.1f}m, '
            f'y={d["ymin"]:.1f}..{d["ymax"]:.1f}m, '
            f'strength={d["depth"]:.3f}m'
        )

    print(f"Generic elevated blocks: {len(blocks)}")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="driver_view_pothole.png")
    args = parser.parse_args()

    points = load_bin(args.input)
    points = clean_points(points)

    if len(points) < 100:
        raise RuntimeError("Too few usable LiDAR points.")

    render(points, args.output)


if __name__ == "__main__":
    main()
