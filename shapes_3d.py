import numpy as np
import math


# ==========================================
# ALGORITMA BRESENHAM 3D (untuk wireframe edges)
# ==========================================
def bresenham_3d(x1, y1, z1, x2, y2, z2):
    """Algoritma Bresenham 3D untuk membentuk wireframe edges pada objek 3D."""
    SCALE = 50.0
    x1_i, y1_i, z1_i = int(x1 * SCALE), int(y1 * SCALE), int(z1 * SCALE)
    x2_i, y2_i, z2_i = int(x2 * SCALE), int(y2 * SCALE), int(z2 * SCALE)

    pts = []
    dx = abs(x2_i - x1_i)
    dy = abs(y2_i - y1_i)
    dz = abs(z2_i - z1_i)
    xs = 1 if x2_i > x1_i else -1
    ys = 1 if y2_i > y1_i else -1
    zs = 1 if z2_i > z1_i else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1_i != x2_i:
            pts.append((x1_i / SCALE, y1_i / SCALE, z1_i / SCALE))
            x1_i += xs
            if p1 >= 0: y1_i += ys; p1 -= 2 * dx
            if p2 >= 0: z1_i += zs; p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1_i != y2_i:
            pts.append((x1_i / SCALE, y1_i / SCALE, z1_i / SCALE))
            y1_i += ys
            if p1 >= 0: x1_i += xs; p1 -= 2 * dy
            if p2 >= 0: z1_i += zs; p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1_i != z2_i:
            pts.append((x1_i / SCALE, y1_i / SCALE, z1_i / SCALE))
            z1_i += zs
            if p1 >= 0: y1_i += ys; p1 -= 2 * dz
            if p2 >= 0: x1_i += xs; p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
    pts.append((x2_i / SCALE, y2_i / SCALE, z2_i / SCALE))
    return pts


def apply_wireframe(particles, vertices, edges, wireframe_ratio=0.15):

    all_edge_pts = []
    for v1_idx, v2_idx in edges:
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        pts = bresenham_3d(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2])
        all_edge_pts.extend(pts)

    if all_edge_pts:
        edge_arr = np.array(all_edge_pts)
        n = len(particles)
        limit = min(len(edge_arr), int(n * wireframe_ratio))
        if limit > 0:
            sel = np.random.choice(len(edge_arr), limit,
                                   replace=(limit > len(edge_arr)))
            edge_particles = edge_arr[sel].copy()
            edge_particles += np.random.normal(0, 0.01, edge_particles.shape)
            particles[-limit:] = edge_particles

    return particles


# ==========================================
# GENERATOR OBJEK 3D SOLID
# ==========================================

def generate_cube(num_particles, size=2.5):

    half = size / 2.0
    per_face = num_particles // 6
    particles = np.zeros((num_particles, 3))
    idx = 0

    for face in range(6):
        n = per_face if face < 5 else (num_particles - 5 * per_face)
        axis = face // 2   # 0=X, 1=Y, 2=Z
        sign = 1.0 if face % 2 == 0 else -1.0
        other = [a for a in range(3) if a != axis]

        pts = np.zeros((n, 3))
        pts[:, axis] = sign * half
        pts[:, other[0]] = np.random.uniform(-half, half, n)
        pts[:, other[1]] = np.random.uniform(-half, half, n)
        particles[idx:idx + n] = pts
        idx += n

    # Wireframe: 8 vertex, 12 edges
    vertices = [
        [-half, -half, -half], [half, -half, -half],
        [half, half, -half], [-half, half, -half],
        [-half, -half, half], [half, -half, half],
        [half, half, half], [-half, half, half],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # back face
        (4, 5), (5, 6), (6, 7), (7, 4),   # front face
        (0, 4), (1, 5), (2, 6), (3, 7),   # connecting
    ]
    particles = apply_wireframe(particles, vertices, edges)
    return particles


def generate_pyramid(num_particles, base_size=3.0, height=3.0):

    half = base_size / 2.0
    apex = np.array([0.0, height / 2.0, 0.0])
    base_y = -height / 2.0

    base_verts = [
        np.array([-half, base_y, -half]),
        np.array([half, base_y, -half]),
        np.array([half, base_y, half]),
        np.array([-half, base_y, half]),
    ]

    particles = np.zeros((num_particles, 3))

    # Alas persegi — 30% partikel
    n_base = int(num_particles * 0.3)
    particles[:n_base, 0] = np.random.uniform(-half, half, n_base)
    particles[:n_base, 1] = base_y
    particles[:n_base, 2] = np.random.uniform(-half, half, n_base)

    # 4 sisi segitiga — 70% partikel
    n_faces = num_particles - n_base
    n_per_face = n_faces // 4
    faces = [(0, 1), (1, 2), (2, 3), (3, 0)]

    idx = n_base
    for fi, (i, j) in enumerate(faces):
        n = n_per_face if fi < 3 else (num_particles - idx)
        v1 = base_verts[i]
        v2 = base_verts[j]

        # Barycentric Coordinates sampling pada segitiga (apex, v1, v2)
        r1 = np.random.random(n)
        r2 = np.random.random(n)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]

        pts = (apex[None, :] * (1 - r1 - r2)[:, None] +
               v1[None, :] * r1[:, None] +
               v2[None, :] * r2[:, None])
        particles[idx:idx + n] = pts
        idx += n

    # Wireframe: 5 vertex (4 base + 1 apex), 8 edges
    vertices = [
        [-half, base_y, -half], [half, base_y, -half],
        [half, base_y, half], [-half, base_y, half],
        [0.0, height / 2.0, 0.0],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # base
        (0, 4), (1, 4), (2, 4), (3, 4),   # apex edges
    ]
    particles = apply_wireframe(particles, vertices, edges)
    return particles


def generate_sphere(num_particles, radius=2.0):

    golden_ratio = (1 + math.sqrt(5)) / 2.0
    indices = np.arange(num_particles, dtype=np.float64)

    # Fibonacci Sphere: distribusi merata berdasarkan golden ratio
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / num_particles)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.cos(phi)
    z = radius * np.sin(phi) * np.sin(theta)

    particles = np.stack([x, y, z], axis=1).astype(np.float64)
    particles += np.random.normal(0, 0.02, particles.shape)
    return particles


def generate_torus(num_particles, R=2.0, r=0.8):

    u = np.random.uniform(0, 2 * np.pi, num_particles)
    v = np.random.uniform(0, 2 * np.pi, num_particles)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = r * np.sin(v)
    z = (R + r * np.cos(v)) * np.sin(u)

    particles = np.stack([x, y, z], axis=1)
    particles += np.random.normal(0, 0.02, particles.shape)
    return particles


def generate_cylinder(num_particles, radius=1.5, height=3.5):

    half_h = height / 2.0
    particles = np.zeros((num_particles, 3))

    # Permukaan samping — 60% partikel
    n_side = int(num_particles * 0.6)
    theta_side = np.random.uniform(0, 2 * np.pi, n_side)
    particles[:n_side, 0] = radius * np.cos(theta_side)
    particles[:n_side, 1] = np.random.uniform(-half_h, half_h, n_side)
    particles[:n_side, 2] = radius * np.sin(theta_side)

    # Tutup atas — 20% partikel
    n_top = int(num_particles * 0.2)
    r_top = radius * np.sqrt(np.random.random(n_top))
    theta_top = np.random.uniform(0, 2 * np.pi, n_top)
    particles[n_side:n_side + n_top, 0] = r_top * np.cos(theta_top)
    particles[n_side:n_side + n_top, 1] = half_h
    particles[n_side:n_side + n_top, 2] = r_top * np.sin(theta_top)

    # Tutup bawah — sisa partikel
    n_bottom = num_particles - n_side - n_top
    r_bot = radius * np.sqrt(np.random.random(n_bottom))
    theta_bot = np.random.uniform(0, 2 * np.pi, n_bottom)
    idx = n_side + n_top
    particles[idx:, 0] = r_bot * np.cos(theta_bot)
    particles[idx:, 1] = -half_h
    particles[idx:, 2] = r_bot * np.sin(theta_bot)

    particles += np.random.normal(0, 0.01, particles.shape)
    return particles


# ==========================================
# ALGORITMA PENCAHAYAAN PHONG (Ambient + Diffuse)
# ==========================================

def _compute_normals(particles, shape_type):

    normals = np.zeros_like(particles)

    if shape_type == 'cube':
        # Normal = arah komponen absolut terbesar (face normal)
        abs_pos = np.abs(particles)
        max_axis = np.argmax(abs_pos, axis=1)
        for axis in range(3):
            mask = max_axis == axis
            normals[mask, axis] = np.sign(particles[mask, axis])

    elif shape_type == 'pyramid':
        # Normal mengarah keluar dari pusat massa piramida
        center_offset = particles.copy()
        center_offset[:, 1] += 0.5  # Offset pusat piramida
        norms = np.linalg.norm(center_offset, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        normals = center_offset / norms

    elif shape_type == 'sphere':
        # Normal = posisi dinormalisasi (arah radial dari pusat)
        norms = np.linalg.norm(particles, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        normals = particles / norms

    elif shape_type == 'torus':
        # Normal = arah dari pusat tube ke permukaan
        R = 2.0  # Major radius (harus sama dengan generate_torus)
        theta = np.arctan2(particles[:, 2], particles[:, 0])
        ring_center = np.stack([R * np.cos(theta),
                                np.zeros(len(particles)),
                                R * np.sin(theta)], axis=1)
        diff = particles - ring_center
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        normals = diff / norms

    elif shape_type == 'cylinder':
        half_h = 1.75  # height/2 (harus sama dengan generate_cylinder)
        is_cap = np.abs(particles[:, 1]) > (half_h - 0.15)
        # Tutup atas/bawah: normal vertikal
        normals[is_cap, 1] = np.sign(particles[is_cap, 1])
        # Sisi samping: normal radial di bidang XZ
        side = ~is_cap
        r = np.sqrt(particles[side, 0] ** 2 + particles[side, 2] ** 2)
        r = np.maximum(r, 1e-6)
        normals[side, 0] = particles[side, 0] / r
        normals[side, 2] = particles[side, 2] / r

    return normals


def compute_lighting_colors(particles, shape_type, base_color,
                            light_pos=None, time_val=0.0):

    if light_pos is None:
        # Sumber cahaya berputar mengelilingi objek
        lx = 5.0 * math.cos(time_val * 0.5)
        ly = 4.0
        lz = 5.0 * math.sin(time_val * 0.5)
        light_pos = np.array([lx, ly, lz])

    normals = _compute_normals(particles, shape_type)

    # Arah cahaya: dari partikel ke sumber cahaya
    light_dir = light_pos - particles
    light_dist = np.linalg.norm(light_dir, axis=1, keepdims=True)
    light_dist = np.maximum(light_dist, 1e-6)
    light_dir = light_dir / light_dist

    # Komponen Diffuse: dot(normal, light_dir), clamp ke [0, 1]
    diffuse = np.sum(normals * light_dir, axis=1)
    diffuse = np.clip(diffuse, 0.0, 1.0)

    # Gabungkan: Ambient (25%) + Diffuse (75%)
    ambient = 0.25
    intensity = ambient + (1.0 - ambient) * diffuse

    # Terapkan intensitas ke warna dasar
    colors = np.zeros((len(particles), 4))
    colors[:, 0] = base_color[0] * intensity
    colors[:, 1] = base_color[1] * intensity
    colors[:, 2] = base_color[2] * intensity
    colors[:, 3] = 0.9  # Alpha

    return colors