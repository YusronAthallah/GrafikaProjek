import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import threading
import time
from sign_language import detect_finger_number
from sound_effects import SoundManager

WIDTH, HEIGHT = 900, 620
NUM_PARTICLES = 10000  

lock = threading.Lock()
shared_data = {
    "mode": 1,
    "target_x": 0.0,
    "target_y": 0.0,
    "target_z": -12.0,
    "frame": None,
    "running": True,
    "sign_mode": False,
    "detected_number": None,
    "accumulated_text": "",
    "trigger_add": False,
    "sound_mode": False
}

# ==========================================
# ALGORITMA GRAFIKA (Bresenham & Bezier & Fill Area)
# ==========================================
def bresenham_3d(x1, y1, z1, x2, y2, z2):
    """Algoritma Bresenham 3D untuk membentuk garis antar partikel."""
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
            pts.append((x1_i/SCALE, y1_i/SCALE, z1_i/SCALE))
            x1_i += xs
            if p1 >= 0: y1_i += ys; p1 -= 2 * dx
            if p2 >= 0: z1_i += zs; p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1_i != y2_i:
            pts.append((x1_i/SCALE, y1_i/SCALE, z1_i/SCALE))
            y1_i += ys
            if p1 >= 0: x1_i += xs; p1 -= 2 * dy
            if p2 >= 0: z1_i += zs; p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1_i != z2_i:
            pts.append((x1_i/SCALE, y1_i/SCALE, z1_i/SCALE))
            z1_i += zs
            if p1 >= 0: y1_i += ys; p1 -= 2 * dz
            if p2 >= 0: x1_i += xs; p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
    pts.append((x2_i/SCALE, y2_i/SCALE, z2_i/SCALE))
    return pts

def cubic_bezier(t, p0, p1, p2, p3):
    """Algoritma Kurva Bezier (Cubic)."""
    u = 1 - t
    return (u**3)*p0 + 3*(u**2)*t*p1 + 3*u*(t**2)*p2 + (t**3)*p3

def custom_fill_rect(img, x, y, w, h, color):
    """Algoritma Fill Area kustom menggunakan iterasi scanline (baris demi baris)."""
    y_start = max(0, y)
    y_end = min(img.shape[0], y + h)
    x_start = max(0, x)
    x_end = min(img.shape[1], x + w)
    # Loop baris (Scanline Fill)
    for row in range(y_start, y_end):
        for col in range(x_start, x_end):
            img[row, col] = color

pos_space = np.random.uniform(-4.0, 4.0, (NUM_PARTICLES, 3))
# Terapkan Bresenham 3D pada mode Kosmos (membentuk pola konstelasi berlian/silang)
bresenham_pts = []
bresenham_pts += bresenham_3d(-2, 0, 0, 0, 2, 0)
bresenham_pts += bresenham_3d(0, 2, 0, 2, 0, 0)
bresenham_pts += bresenham_3d(2, 0, 0, 0, -2, 0)
bresenham_pts += bresenham_3d(0, -2, 0, -2, 0, 0)
b_array = np.array(bresenham_pts)
if len(b_array) > 0:
    limit = min(len(b_array), NUM_PARTICLES // 3)
    pos_space[:limit] = b_array[:limit]

def _make_text_pos(text, img_w=800, font_scale=3.5, thickness=12):
    """Helper: buat partikel dari teks."""
    img = np.zeros((200, img_w), dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = max(0, (img_w - text_size[0]) // 2)
    text_y = (200 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
    yi, xi = np.where(img > 0)
    x_t = (xi - img_w // 2) / 70.0
    y_t = -(yi - 100) / 70.0
    z_t = np.random.uniform(-0.1, 0.1, len(x_t))
    pts = np.stack((x_t, y_t, z_t), axis=-1)
    idx = np.random.choice(len(pts), NUM_PARTICLES, replace=True)
    result = pts[idx]
    result[:, 0] += np.random.normal(0, 0.015, NUM_PARTICLES)
    result[:, 1] += np.random.normal(0, 0.015, NUM_PARTICLES)
    return result

# 1 jari
pos_upn = _make_text_pos("UPN", img_w=600, font_scale=5.0, thickness=14)

# 2 jari
pos_veteran = _make_text_pos("VETERAN", img_w=800, font_scale=3.5, thickness=12)

# 3 jari
pos_yogya = _make_text_pos("YOGYAKARTA", img_w=1000, font_scale=3.0, thickness=10)

# tangan kanan genggam
pos_jokowi = _make_text_pos("HIDUP JOKOWI", img_w=900, font_scale=3.0, thickness=10)

# tangan kiri genggam
pos_lawan = _make_text_pos("SAYA AKAN LAWAN", img_w=1100, font_scale=2.8, thickness=10)

# 4 jari
pos_upn_veteran = _make_text_pos("UPN VETERAN", img_w=900, font_scale=3.0, thickness=10)
_img_upn_v = np.zeros((300, 1100), dtype=np.uint8)
_t1 = "UPN VETERAN"
_t2 = "YOGYAKARTA"
_s1 = cv2.getTextSize(_t1, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 10)[0]
_s2 = cv2.getTextSize(_t2, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 10)[0]
_x1 = (1100 - _s1[0]) // 2
_x2 = (1100 - _s2[0]) // 2
cv2.putText(_img_upn_v, _t1, (_x1, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 10, cv2.LINE_AA)
cv2.putText(_img_upn_v, _t2, (_x2, 240), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 10, cv2.LINE_AA)
_yi7, _xi7 = np.where(_img_upn_v > 0)
_x7 = (_xi7 - 550) / 70.0
_y7 = -(_yi7 - 150) / 70.0
_z7 = np.random.uniform(-0.1, 0.1, len(_x7))
_pts7 = np.stack((_x7, _y7, _z7), axis=-1)
_idx7 = np.random.choice(len(_pts7), NUM_PARTICLES, replace=True)
pos_upn_veteran = _pts7[_idx7]
pos_upn_veteran[:, 0] += np.random.normal(0, 0.015, NUM_PARTICLES)
pos_upn_veteran[:, 1] += np.random.normal(0, 0.015, NUM_PARTICLES)

# angka
pos_sign_text = np.copy(pos_space)
current_sign_text = ""
cached_sign_pos = np.copy(pos_space)

def generate_text_particles(text):
    if not text:
        return np.copy(pos_space)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.5
    thickness = 12
    max_width = 750
    
    lines = []
    current_line = ""
    for char in text:
        if char == '\n':
            lines.append(current_line.strip())
            current_line = ""
            continue
            
        test_line = current_line + char
        size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        if size[0] > max_width and current_line != "":
            lines.append(current_line.strip())
            current_line = char.lstrip()
        else:
            current_line = test_line
    if current_line or text.endswith('\n'):
        lines.append(current_line.strip())
        
    img = np.zeros((800, 800), dtype=np.uint8)
    line_height = 110
    total_height = len(lines) * line_height
    start_y = (800 - total_height) // 2 + 80
    
    for i, line in enumerate(lines):
        if not line: continue
        text_size = cv2.getTextSize(line, font, scale, thickness)[0]
        text_x = max(0, (800 - text_size[0]) // 2)
        text_y = start_y + i * line_height
        cv2.putText(img, line, (text_x, text_y), font, scale, 255, thickness, cv2.LINE_AA)
        
    yi, xi = np.where(img > 0)
    if len(xi) == 0:
        return np.copy(pos_space)
    
    x_t = (xi - 400) / 70.0
    y_t = -(yi - 400) / 70.0 
    z_t = np.random.uniform(-0.1, 0.1, len(x_t))
    pts = np.stack((x_t, y_t, z_t), axis=-1)
    
    if len(pts) > 0:
        idx = np.random.choice(len(pts), NUM_PARTICLES, replace=True)
        pts_sampled = pts[idx]
        pts_sampled[:, 0] += np.random.normal(0, 0.015, NUM_PARTICLES)
        pts_sampled[:, 1] += np.random.normal(0, 0.015, NUM_PARTICLES)
        return pts_sampled
    return np.copy(pos_space)

current_pos = np.copy(pos_space)
target_pos = np.copy(pos_space)

# logika deteksi gestur
def hitung_mode_gestur(hand_landmarks, hand_label="Right"):
    tips = [8, 12, 16, 20]  
    pips = [6, 10, 14, 18]
    jari_berdiri = [hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y for t, p in zip(tips, pips)]
    
    if sum(jari_berdiri) == 0:
        # Genggam: bedakan tangan kiri dan kanan
        if hand_label == "Right":
            return 5  # Genggam tangan kanan -> HIDUP JOKOWI
        else:
            return 6  # Genggam tangan kiri -> SAYA AKAN LAWAN
    if jari_berdiri[0] and not any(jari_berdiri[1:]):
        return 2  # 1 jari (Telunjuk) -> UPN
    if jari_berdiri[0] and jari_berdiri[1] and not jari_berdiri[2] and not jari_berdiri[3]:
        return 3  # 2 jari (Peace) -> VETERAN
    if jari_berdiri[0] and jari_berdiri[1] and jari_berdiri[2] and not jari_berdiri[3]:
        return 4  # 3 jari -> YOGYAKARTA
    if all(jari_berdiri):
        return 7  # 4 jari -> UPN VETERAN YOGYAKARTA
    return 1  # Default

# kamera & AI
def camera_thread_func():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.waitKey(500)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while shared_data["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        local_mode = 1
        local_x, local_y, local_z = 0.0, 0.0, -12.0
        local_number = None

        with lock:
            is_sign_mode = shared_data["sign_mode"]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_sign_mode:
                # Deteksi angka berdasarkan jumlah jari dari semua tangan yang terdeteksi
                local_number = detect_finger_number(results.multi_hand_landmarks)
                local_mode = 8  # Custom mode untuk Deteksi Angka
            else:
                # Mode gestur menggunakan tangan pertama, dengan info handedness
                hand_label = "Right"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[0].classification[0].label
                local_mode = hitung_mode_gestur(results.multi_hand_landmarks[0], hand_label)

            # Posisi tracking menggunakan tangan pertama
            first_hand = results.multi_hand_landmarks[0]
            wrist = first_hand.landmark[0]
            local_x = (wrist.x - 0.5) * 10.0 
            local_y = -(wrist.y - 0.5) * 7.0 
            
            pinky_mcp = first_hand.landmark[17]
            distance = math.sqrt((wrist.x - pinky_mcp.x)**2 + (wrist.y - pinky_mcp.y)**2)
            local_z = -10.0 - (1.0 / (distance + 0.01)) * 0.2

        with lock:
            if not is_sign_mode:
                shared_data["mode"] = local_mode
            else:
                shared_data["mode"] = 8
            
            # Simple debounce / stabilization untuk angka
            if local_number is not None:
                if "debounce_number" not in shared_data:
                    shared_data["debounce_number"] = local_number
                    shared_data["debounce_count"] = 0
                
                if local_number == shared_data["debounce_number"]:
                    shared_data["debounce_count"] += 1
                else:
                    shared_data["debounce_number"] = local_number
                    shared_data["debounce_count"] = 1
                    
                if shared_data["debounce_count"] >= 10:
                    shared_data["detected_number"] = local_number
            else:
                shared_data["detected_number"] = None

            shared_data["target_x"] = local_x
            shared_data["target_y"] = local_y
            shared_data["target_z"] = local_z
            shared_data["frame"] = frame

    cap.release()

# Jalankan Thread Kamera terlebih dahulu
camera_thread = threading.Thread(target=camera_thread_func, daemon=True)
camera_thread.start()

time.sleep(1.0)

# render grafis 3d
pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Space Gesture Controller")

# Inisialisasi Sound Manager
sound_mgr = SoundManager()
sound_mgr.init()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glEnable(GL_DEPTH_TEST)

clock = pygame.time.Clock()
rotation_angle = 0.0
hand_x, hand_y, hand_z = 0.0, 0.0, -12.0

while shared_data["running"]:
    pygame.event.pump()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            shared_data["running"] = False
        elif event.type == KEYDOWN:
            with lock:
                if event.key == K_TAB:
                    shared_data["sign_mode"] = not shared_data["sign_mode"]
                    if not shared_data["sign_mode"]:
                        shared_data["mode"] = 1
                elif event.key in (K_LSHIFT, K_RSHIFT):
                    # Toggle Sound Mode
                    new_state = sound_mgr.toggle()
                    shared_data["sound_mode"] = new_state
                elif shared_data["sign_mode"]:
                    if event.key == K_RETURN or event.key == K_KP_ENTER:
                        shared_data["accumulated_text"] += "\n"
                        shared_data["trigger_add"] = True
                    elif event.key == K_BACKSPACE:
                        shared_data["accumulated_text"] = shared_data["accumulated_text"][:-1]
                        shared_data["trigger_add"] = True
                    elif event.key == K_SPACE:
                        shared_data["accumulated_text"] += " "
                        shared_data["trigger_add"] = True
                    elif event.key == K_DELETE:
                        shared_data["accumulated_text"] = ""
                        shared_data["trigger_add"] = True
                    elif hasattr(event, 'unicode') and event.unicode.isalnum():
                        shared_data["accumulated_text"] += event.unicode.upper()
                        shared_data["trigger_add"] = True

    with lock:
        current_mode = shared_data["mode"]
        target_hand_x = shared_data["target_x"]
        target_hand_y = shared_data["target_y"]
        target_hand_z = shared_data["target_z"]
        frame = shared_data["frame"]
        is_sign_mode = shared_data["sign_mode"]
        is_sound_mode = shared_data["sound_mode"]
        det_number = shared_data.get("detected_number", "")
        acc_text = shared_data.get("accumulated_text", "")
        
        if shared_data.get("trigger_add", False):
            shared_data["trigger_add"] = False

    # Update Sound Manager berdasarkan mode saat ini
    sound_mgr.update(current_mode)

    if frame is not None:
        display_frame = frame.copy()
        if is_sign_mode:
            cv2.putText(display_frame, "NUMBER MODE: ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Angka: {det_number if det_number else '-'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Text: {acc_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "NUMBER MODE: OFF (Press 'TAB')", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Tampilkan status Sound Mode dengan kotak fill custom
        sound_label = "SOUND: ON" if is_sound_mode else "SOUND: OFF (Press 'SHIFT')"
        sound_color = (0, 255, 128) if is_sound_mode else (128, 128, 128)
        
        # Algoritma Fill Area kustom untuk kotak latar belakang indikator suara
        box_y = display_frame.shape[0] - 40
        custom_fill_rect(display_frame, 5, box_y, 300, 35, (40, 40, 40)) # kotak abu gelap
        
        cv2.putText(display_frame, sound_label, (10, display_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sound_color, 2)
            
        frame_small = cv2.resize(display_frame, (360, 240))
        cv2.imshow("Hand Sensor Monitor", frame_small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared_data["running"] = False

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    hand_x += (target_hand_x - hand_x) * 0.25
    hand_y += (target_hand_y - hand_y) * 0.25
    hand_z += (target_hand_z - hand_z) * 0.25

    if current_mode == 1:
        target_pos = pos_space
        rotation_angle += 0.5 
    elif current_mode == 2:
        target_pos = pos_upn
        rotation_angle = 0.0 
    elif current_mode == 3:
        target_pos = pos_veteran
        rotation_angle = 0.0  
    elif current_mode == 4:
        target_pos = pos_yogya
        rotation_angle = 0.0  
    elif current_mode == 5:
        target_pos = pos_jokowi
        rotation_angle = 0.0  
    elif current_mode == 6:
        target_pos = pos_lawan
        rotation_angle = 0.0  
    elif current_mode == 7:
        target_pos = pos_upn_veteran
        rotation_angle = 0.0  
    elif current_mode == 8:
        display_str = acc_text if acc_text else (det_number if det_number else "")
        if display_str:
            if display_str != current_sign_text:
                cached_sign_pos = generate_text_particles(display_str)
                current_sign_text = display_str
            target_pos = cached_sign_pos
            rotation_angle = 0.0
        else:
            target_pos = pos_space
            rotation_angle += 0.2
            current_sign_text = ""

    current_pos += (target_pos - current_pos) * 0.15

    # Offset global Y agar semua objek tampil sedikit lebih naik di layar
    Y_OFFSET = 1.5
    
    if current_mode in [2, 3, 4, 5, 6, 7, 8]:
        glTranslatef(hand_x, hand_y + Y_OFFSET, hand_z)
    else:
        glTranslatef(0.0, Y_OFFSET, -12.0)
        
    # Transformasi Scaling dinamis untuk mode 5 dan 6
    if current_mode in [5, 6]:
        scale = 1.0 + 0.15 * math.sin(pygame.time.get_ticks() / 150.0)
        glScalef(scale, scale, scale)
    
    glRotatef(rotation_angle, 0.0, 1.0, 0.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(4.5)  
    
    glBegin(GL_POINTS)
    for i in range(NUM_PARTICLES):
        if current_mode == 2:
            glColor4f(1.0, 0.85, 0.0, 0.9)       # UPN - Kuning Emas
        elif current_mode == 3:
            glColor4f(0.0, 0.8, 1.0, 0.9)        # VETERAN - Cyan
        elif current_mode == 4:
            glColor4f(0.2, 1.0, 0.4, 0.9)        # YOGYAKARTA - Hijau
        elif current_mode == 5:
            glColor4f(1.0, 0.15, 0.15, 0.95)     # HIDUP JOKOWI - Merah
        elif current_mode == 6:
            glColor4f(0.8, 0.3, 1.0, 0.9)        # SAYA AKAN LAWAN - Ungu
        elif current_mode == 7:
            glColor4f(1.0, 0.6, 0.0, 0.9)        # UPN VETERAN YOGYAKARTA - Oranye
        elif current_mode == 8:
            glColor4f(0.0, 1.0, 0.53, 0.9)       # Number Mode - Hijau Neon
        else:
            glColor4f(0.1, 0.5, 1.0, 0.8)        # Kosmos - Biru  
            
        glVertex3f(current_pos[i, 0], current_pos[i, 1], current_pos[i, 2])
    glEnd()

    # Algoritma Kurva Bezier dinamis untuk Mode 7
    if current_mode == 7:
        glBegin(GL_POINTS)
        glColor4f(1.0, 1.0, 0.0, 1.0) # Kuning
        time_val = pygame.time.get_ticks() / 1000.0
        
        # Jari-jari cincin orbit utama
        a = 6.2  # Radius sumbu X (lebar)
        b = 3.5  # Radius untuk sumbu tegak lurus
        k = 1.3333  # Konstanta Bezier
        
        # Sudut kemiringan (berputar perlahan)
        theta = pygame.time.get_ticks() / 1500.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Titik jangkar di ujung kiri dan kanan (Simetris sempurna di Y=0, Z=0)
        p_left = np.array([-a, 0.0, 0.0])
        p_right = np.array([a, 0.0, 0.0])
        
        # Kurva 1 (Sisi Atas cincin yang dimiringkan dalam 3D)
        c1_1 = np.array([-a, b * k * cos_t, b * k * sin_t])
        c1_2 = np.array([a, b * k * cos_t, b * k * sin_t])
        
        # Kurva 2 (Sisi Bawah cincin yang dimiringkan dalam 3D)
        c2_1 = np.array([a, -b * k * cos_t, -b * k * sin_t])
        c2_2 = np.array([-a, -b * k * cos_t, -b * k * sin_t])
        
        # Gambar Kurva 1 (Depan)
        for i in range(150):
            t = i / 150.0
            pt = cubic_bezier(t, p_left, c1_1, c1_2, p_right)
            for _ in range(2): 
                glVertex3f(pt[0] + np.random.normal(0,0.04), pt[1] + np.random.normal(0,0.04), pt[2] + np.random.normal(0,0.04))
                
        # Gambar Kurva 2 (Belakang)
        for i in range(150):
            t = i / 150.0
            pt = cubic_bezier(t, p_right, c2_1, c2_2, p_left)
            for _ in range(2): 
                glVertex3f(pt[0] + np.random.normal(0,0.04), pt[1] + np.random.normal(0,0.04), pt[2] + np.random.normal(0,0.04))
        glEnd()

    pygame.display.flip()
    clock.tick(60)

sound_mgr.cleanup()
cv2.destroyAllWindows()
pygame.quit()