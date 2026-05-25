"""
SPECTRA - Sistema de Pintura e Educacao Com Tecnologia de Reconhecimento de maos
Totalmente controlado por gestos com a camera. Sem teclado, sem mouse.

Modos:
  Menu Principal    - navegacao por hover do indicador
  Pintura Livre     - combinacoes de dedos formam cores; desfazer, salvar
  Educacao: Cores   - quiz: gesticule a cor exibida
  Educacao: Contagem- levante exatamente N dedos
  Fisioterapia      - exercicios guiados de reabilitacao
"""

import os
import time
import urllib.request
import random
import math
import collections
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
import csv
try:
    import winsound
except Exception:
    winsound = None

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── MODELO ───────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"


def ensure_model(model_path: str) -> None:
    if os.path.exists(model_path):
        return
    print("Baixando modelo de landmarks...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Modelo baixado com sucesso.")


# ─── CONFIGURACOES GERAIS ──────────────────────────────────────
SMOOTH_ALPHA      = 0.35
HOVER_SELECT_SECS = 1.2
GESTURE_HOLD_SECS = 1.5
TRAIL_LENGTH      = 18
UNDO_HISTORY      = 15


# ─── MODOS DA APP ─────────────────────────────────────────────
class AppMode(Enum):
    MENU        = "Menu Principal"
    PAINT       = "Pintura Livre"
    EDU_COLORS  = "Educacao: Cores"
    EDU_COUNT   = "Educacao: Contagem"
    PHYSIO      = "Fisioterapia"


# ─── COMBINACOES DE DEDOS → COR ───────────────────────────────
# (polegar, indicador, medio, anelar, mindinho) True=levantado
COMBO_COLORS = [
    ((False, True,  False, False, False), "Vermelho",     (0,   0,   255)),
    ((False, False, True,  False, False), "Verde",        (0,   255, 0  )),
    ((False, False, False, True,  False), "Azul",         (255, 0,   0  )),
    ((False, False, False, False, True ), "Amarelo",      (0,   255, 255)),
    ((False, True,  True,  False, False), "Laranja",      (0,   165, 255)),
    ((False, True,  False, True,  False), "Roxo",         (128, 0,   128)),
    ((False, True,  False, False, True ), "Rosa",         (147, 20,  255)),
    ((False, False, True,  True,  False), "Ciano",        (255, 255, 0  )),
    ((False, False, True,  False, True ), "Marrom",       (42,  42,  165)),
    ((False, True,  True,  True,  False), "Branco/Apaga", (255, 255, 255)),
    ((False, False, False, True,  True ), "DESFAZER",     None           ),  # anelar+mindinho
    ((True,  False, False, False, False), "Borracha",     None           ),  # polegar
    ((True,  True,  True,  True,  True ), "LIMPAR",       None           ),  # mao aberta
]

ERASER_COLOR_BGR = (255, 255, 255)
ERASER_SIZE      = 50
DEFAULT_BRUSH    = 10


# ─── BOTAO UI ─────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label,
                 color_bg=(50, 50, 50), color_text=(255, 255, 255), value=None):
        self.rect = (x, y, w, h)
        self.label = label
        self.color_bg = color_bg
        self.color_text = color_text
        self.value = value
        self.hover_start = None
        self.progress = 0.0

    def contains(self, pt):
        x, y, w, h = self.rect
        return x <= pt[0] <= x + w and y <= pt[1] <= y + h

    def update_hover(self, pointing):
        if not pointing:
            self.hover_start = None
            self.progress = 0.0
            return False
        if self.hover_start is None:
            self.hover_start = time.time()
        elapsed = time.time() - self.hover_start
        self.progress = min(1.0, elapsed / HOVER_SELECT_SECS)
        if elapsed >= HOVER_SELECT_SECS:
            self.hover_start = None
            self.progress = 0.0
            return True
        return False

    def draw(self, frame):
        x, y, w, h = self.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color_bg, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)
        if self.progress > 0:
            cv2.rectangle(frame, (x, y + h - 5),
                          (x + int(w * self.progress), y + h), (0, 220, 255), -1)
        fs = 0.52
        (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(frame, self.label,
                    (x + (w - tw) // 2, y + (h + th) // 2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, self.color_text, 1, cv2.LINE_AA)


# ─── CANVAS COM UNDO ──────────────────────────────────────────
class DrawingCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self._history = collections.deque(maxlen=UNDO_HISTORY)

    def begin_stroke(self):
        """Salva estado antes de iniciar novo traco (para undo)."""
        self._history.append(self.canvas.copy())

    def undo(self):
        """Restaura estado anterior. Retorna True se havia historico."""
        if self._history:
            self.canvas = self._history.pop()
            return True
        return False

    def draw(self, pt1, pt2, color_bgr, size):
        if color_bgr is None:
            cv2.line(self.canvas, pt1, pt2, ERASER_COLOR_BGR, ERASER_SIZE)
            cv2.circle(self.canvas, pt2, ERASER_SIZE // 2, ERASER_COLOR_BGR, -1)
        else:
            cv2.line(self.canvas, pt1, pt2, color_bgr, size)
            cv2.circle(self.canvas, pt2, size // 2, color_bgr, -1)

    def clear(self):
        self._history.append(self.canvas.copy())
        self.canvas[:] = 255


# ─── RASTRO VISUAL (TRAIL) ────────────────────────────────────
class Trail:
    """Pontos com fade-out temporal para rastro do dedo."""
    def __init__(self, maxlen=TRAIL_LENGTH):
        self._pts = collections.deque(maxlen=maxlen)

    def add(self, pt, color):
        self._pts.append((pt, color, time.time()))

    def clear(self):
        self._pts.clear()

    def draw(self, frame):
        now = time.time()
        for pt, color, t in list(self._pts):
            age = now - t
            if age > 0.45:
                continue
            alpha = 1.0 - age / 0.45
            r = max(2, int(7 * alpha))
            c = tuple(int(ch * alpha) for ch in (color if color else (180, 180, 180)))
            overlay = frame.copy()
            cv2.circle(overlay, pt, r, c, -1)
            cv2.addWeighted(overlay, alpha * 0.65, frame, 1 - alpha * 0.65, 0, frame)


# ─── UTILITARIOS DE MAO ───────────────────────────────────────
def get_finger_states(lm, hand_label="Right"):
    """
    Retorna (polegar, indicador, medio, anelar, mindinho) True=levantado.
    Usa pequenos thresholds para tolerancia a deteccoes ruidosas.
    """
    # tolerancias em coordenadas normalizadas
    TY = 0.02
    TX = 0.02
    if hand_label == "Right":
        thumb = lm[4].x < lm[3].x - TX
    else:
        thumb = lm[4].x > lm[3].x + TX
    index  = lm[8].y  < lm[6].y - TY
    middle = lm[12].y < lm[10].y - TY
    ring   = lm[16].y < lm[14].y - TY
    pinky  = lm[20].y < lm[18].y - TY
    return (thumb, index, middle, ring, pinky)


def get_index_tip(lm, frame_shape):
    """Posicao da ponta do indicador em pixels, ou None."""
    h, w = frame_shape[:2]
    # add small tolerance so slightly bent finger still counts
    if lm[8].y < lm[6].y + 0.03:
        return (int(lm[8].x * w), int(lm[8].y * h))
    return None


def resolve_color(states):
    if states is None:
        return None
    s = tuple(bool(x) for x in states)
    for combo, name, bgr in COMBO_COLORS:
        if s == combo:
            return (name, bgr)
    return None


def get_hand_label(hd_list, idx=0):
    """Extrai rotulo de handedness de forma tolerante, retorna 'Right' por padrao."""
    try:
        entry = hd_list[idx]
        # hd_list pode ser lista de categorias ou entradas aninhadas
        if isinstance(entry, (list, tuple)) and len(entry) > 0:
            cat = entry[0]
        else:
            cat = entry
        return getattr(cat, "category_name", "Right")
    except Exception:
        return "Right"


def count_extended(states):
    return sum(states)


def smooth_pt(prev, pt, alpha=SMOOTH_ALPHA):
    if prev is None:
        return pt
    return (int((1 - alpha) * prev[0] + alpha * pt[0]),
            int((1 - alpha) * prev[1] + alpha * pt[1]))


def draw_finger_hud(frame, states, x0, y0):
    """Mini-HUD com indicadores dos 5 dedos (P I M A Mi)."""
    labels = ["P", "I", "M", "A", "Mi"]
    for i, (label, up) in enumerate(zip(labels, states)):
        cx = x0 + i * 28
        fill = (0, 200, 80) if up else (55, 55, 55)
        cv2.circle(frame, (cx, y0), 11, fill, -1)
        cv2.circle(frame, (cx, y0), 11, (180, 180, 180), 1)
        cv2.putText(frame, label, (cx - 6, y0 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)


def _beep_ok():
    if winsound:
        try:
            winsound.Beep(880, 120)
        except Exception:
            pass


def _beep_err():
    if winsound:
        try:
            winsound.Beep(440, 160)
        except Exception:
            pass


def make_mode_buttons(W, H, include_next=False, right_align=True):
    bw, bh = 138, 44
    y = H - bh - 10
    buttons = []
    # Right-aligned: [Next] [Voltar] [Sair]
    if include_next:
        # place next, voltar, sair from left to right starting at W-3*bw-30
        buttons.append(Button(W - 3*bw - 30, y, bw, bh, "Proximo", (60, 60, 15), value="next"))
        buttons.append(Button(W - 2*bw - 20, y, bw, bh, "Voltar", (20, 20, 80), value="menu"))
        buttons.append(Button(W - bw - 10, y, bw, bh, "Sair", (120, 15, 15), value="quit"))
    else:
        buttons.append(Button(W - 2*bw - 20, y, bw, bh, "Voltar", (20, 20, 80), value="menu"))
        buttons.append(Button(W - bw - 10, y, bw, bh, "Sair", (120, 15, 15), value="quit"))
    return buttons


# ─── MENU ─────────────────────────────────────────────────────
class MenuMode:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        bw, bh, gap = 320, 72, 22
        cx = (W - bw) // 2
        sy = (H - (5 * bh + 4 * gap)) // 2
        self.buttons = [
            Button(cx, sy + 0*(bh+gap), bw, bh, "Pintura Livre",       (20, 90, 20),  value=AppMode.PAINT),
            Button(cx, sy + 1*(bh+gap), bw, bh, "Educacao: Cores",     (20, 20, 120), value=AppMode.EDU_COLORS),
            Button(cx, sy + 2*(bh+gap), bw, bh, "Educacao: Contagem",  (80, 20, 120), value=AppMode.EDU_COUNT),
            Button(cx, sy + 3*(bh+gap), bw, bh, "Fisioterapia",        (120, 55, 15), value=AppMode.PHYSIO),
            Button(cx, sy + 4*(bh+gap), bw, bh, "Sair",                (120, 15, 15), value="quit"),
        ]
        self.smooth_tip = None

    def process(self, frame, lm_list, hd_list):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (self.W, self.H), (12, 12, 12), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "SPECTRA", (self.W//2 - 95, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 220, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "Aponte o indicador e segure sobre o botao (1.2s)",
                    (self.W//2 - 285, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

        tip = None
        states = None
        if lm_list:
            lm = lm_list[0]
            label = get_hand_label(hd_list)
            states = get_finger_states(lm, label)
            raw = get_index_tip(lm, frame.shape)
            if raw:
                self.smooth_tip = smooth_pt(self.smooth_tip, raw)
                tip = self.smooth_tip
                cv2.circle(frame, tip, 14, (0, 220, 255), -1)
                cv2.circle(frame, tip, 14, (255, 255, 255), 2)
        else:
            self.smooth_tip = None

        selected = None
        for btn in self.buttons:
            if btn.update_hover(tip is not None and btn.contains(tip)):
                selected = btn.value
            btn.draw(frame)

        if states is not None:
            draw_finger_hud(frame, states, self.W - 155, self.H - 22)

        return frame, selected


# ─── PINTURA LIVRE ────────────────────────────────────────────
class PaintMode:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.canvas = DrawingCanvas(W, H)
        self.smooth_tip = None
        self.prev_tip = None
        self.current_name = "Nenhum"
        self.current_bgr = None
        self.brush = DEFAULT_BRUSH
        self.trail = Trail()
        self.drawing = False
        self.feedback_msg = ""
        self.feedback_time = 0.0
        self.cmd_name = None
        self.cmd_start = None
        self.cmd_progress = 0.0
        bw, bh = 112, 42
        y = H - bh - 10
        self.buttons = [
            Button(10,       y, bw, bh, "Limpar",   (80, 20, 20),  value="clear"),
            Button(130,      y, bw, bh, "Salvar",   (20, 70, 20),  value="save"),
            Button(250,      y, bw, bh, "Desfazer", (80, 60, 10),  value="undo"),
        ]
        # adiciona botoes padrao (Voltar/Sair)
        self.buttons += make_mode_buttons(W, H, include_next=False)

    def _feedback(self, msg):
        self.feedback_msg = msg
        self.feedback_time = time.time()

    def process(self, frame, lm_list, hd_list):
        h, w = frame.shape[:2]
        composite = cv2.addWeighted(self.canvas.canvas, 0.65, frame, 0.35, 0)

        tip = None
        action = None
        states_hud = None

        if lm_list:
            lm = lm_list[0]
            label = get_hand_label(hd_list)
            states = get_finger_states(lm, label)
            states_hud = states
            result = resolve_color(states)

            if result is not None:
                cname, cbgr = result

                if cname == "LIMPAR":
                    self._update_cmd("clear")
                elif cname == "DESFAZER":
                    self._update_cmd("undo")
                else:
                    self.cmd_name = None
                    self.cmd_start = None
                    self.cmd_progress = 0.0
                    self.current_name = cname
                    self.current_bgr = cbgr

                    if states[1]:  # indicador levantado → desenha
                        raw = (int(lm[8].x * w), int(lm[8].y * h))
                        self.smooth_tip = smooth_pt(self.smooth_tip, raw)
                        tip = self.smooth_tip

                        if not self.drawing:
                            self.canvas.begin_stroke()
                            self.drawing = True

                        if self.prev_tip is not None:
                            self.canvas.draw(self.prev_tip, tip, cbgr, self.brush)
                        self.trail.add(tip, cbgr)
                        self.prev_tip = tip
                    else:
                        self.prev_tip = None
                        self.smooth_tip = None
                        self.drawing = False

            elif states == (True, False, False, False, False):  # polegar = borracha
                raw = (int(lm[4].x * w), int(lm[4].y * h))
                self.smooth_tip = smooth_pt(self.smooth_tip, raw)
                tip = self.smooth_tip
                self.current_name = "Borracha"
                self.current_bgr = None

                if not self.drawing:
                    self.canvas.begin_stroke()
                    self.drawing = True

                if self.prev_tip is not None:
                    self.canvas.draw(self.prev_tip, tip, None, ERASER_SIZE)
                self.prev_tip = tip
                self.cmd_name = None
                self.cmd_progress = 0.0
            else:
                self.prev_tip = None
                self.smooth_tip = None
                self.drawing = False
                self.cmd_name = None
                self.cmd_progress = 0.0
        else:
            self.prev_tip = None
            self.smooth_tip = None
            self.drawing = False
            self.cmd_name = None
            self.cmd_progress = 0.0

        self.trail.draw(composite)

        for btn in self.buttons:
            if btn.update_hover(tip is not None and btn.contains(tip)):
                action = btn.value
            btn.draw(composite)

        self._draw_hud(composite)

        if tip:
            cs = self.current_bgr if self.current_bgr else (200, 200, 200)
            cv2.circle(composite, tip, self.brush, cs, -1)
            cv2.circle(composite, tip, self.brush + 2, (0, 0, 0), 2)

        if self.feedback_msg and (time.time() - self.feedback_time) < 2.5:
            cv2.putText(composite, self.feedback_msg, (w//2 - 165, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3, cv2.LINE_AA)

        if self.cmd_progress > 0 and self.cmd_name is not None:
            label_map = {"clear": "Limpando...", "undo": "Desfazendo..."}
            lbl = label_map.get(self.cmd_name, self.cmd_name)
            cv2.putText(composite, f"{lbl} {self.cmd_progress*100:.0f}%",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.rectangle(composite, (10, h - 52), (310, h - 40), (50, 50, 50), -1)
            cv2.rectangle(composite, (10, h - 52),
                          (10 + int(300 * self.cmd_progress), h - 40), (0, 220, 255), -1)

        if states_hud is not None:
            draw_finger_hud(composite, states_hud, w - 155, h - 68)

        # Processar acoes dos botoes
        if action == "clear":
            self.canvas.clear()
            self.trail.clear()
            self._feedback("Canvas Limpo!")
            action = None
        elif action == "save":
            fname = f"pintura_{int(time.time())}.png"
            cv2.imwrite(fname, self.canvas.canvas)
            self._feedback(f"Salvo: {fname}")
            action = None
        elif action == "undo":
            ok = self.canvas.undo()
            self._feedback("Desfeito!" if ok else "Nada para desfazer")
            action = None

        return composite, action

    def _update_cmd(self, cmd):
        """Gerencia gesto de comando com confirmacao por tempo."""
        if self.cmd_name != cmd:
            self.cmd_name = cmd
            self.cmd_start = time.time()
            self.cmd_progress = 0.0
            return
        # BUGFIX: garante que cmd_start nunca e None neste ponto
        if self.cmd_start is None:
            self.cmd_start = time.time()
        elapsed = time.time() - self.cmd_start
        self.cmd_progress = min(1.0, elapsed / GESTURE_HOLD_SECS)
        if elapsed >= GESTURE_HOLD_SECS:
            done = self.cmd_name
            self.cmd_name = None
            self.cmd_start = None
            self.cmd_progress = 0.0
            if done == "clear":
                self.canvas.clear()
                self.trail.clear()
                self._feedback("Canvas Limpo!")
            elif done == "undo":
                ok = self.canvas.undo()
                self._feedback("Desfeito!" if ok else "Nada para desfazer")

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
        cv2.putText(frame, "PINTURA LIVRE", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        cs = self.current_bgr if self.current_bgr else (200, 200, 200)
        cv2.circle(frame, (230, 14), 11, cs, -1)
        cv2.circle(frame, (230, 14), 12, (255, 255, 255), 1)
        cv2.putText(frame, self.current_name, (248, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pincel:{self.brush}px", (420, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        legenda = [
            "I=Verm  M=Verd  A=Azul  Mi=Amar  I+M=Lara  I+A=Roxo",
            "M+A=Ciano  I+M+A=Branco(apaga)  A+Mi=Desfaz  Pol=Borr",
        ]
        for i, txt in enumerate(legenda):
            cv2.putText(frame, txt, (w - 490, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)


# ─── EDU: QUIZ DE CORES ───────────────────────────────────────
NAMED_COLORS = [
    ("Vermelho", (0,   0,   255), (False, True,  False, False, False)),
    ("Verde",    (0,   255, 0  ), (False, False, True,  False, False)),
    ("Azul",     (255, 0,   0  ), (False, False, False, True,  False)),
    ("Amarelo",  (0,   255, 255), (False, False, False, False, True )),
    ("Laranja",  (0,   165, 255), (False, True,  True,  False, False)),
    ("Roxo",     (128, 0,   128), (False, True,  False, True,  False)),
    ("Ciano",    (255, 255, 0  ), (False, False, True,  True,  False)),
]


class EduColorsMode:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.score = 0
        self.total = 0
        self.streak = 0
        self.max_streak = 0
        self.current = None
        self.result_msg = ""
        self.result_ok = True
        self.result_time = 0.0
        self.answer_cooldown = 0.0
        self._new_question()
        # botoes: Proxima, Voltar, Sair
        self.buttons = make_mode_buttons(W, H, include_next=True)
        # expõe referências usadas em outras partes do codigo
        # Proxima == buttons[0]
        self.btn_next = self.buttons[0]
        # Voltar == buttons[1], Sair == buttons[2]
        self.smooth_tip = None

    def _new_question(self):
        self.current = random.choice(NAMED_COLORS)
        self.answer_cooldown = time.time() + 2.5

    def process(self, frame, lm_list, hd_list):
        h, w = frame.shape[:2]

        # Garante que current nunca e None (dupla verificacao)
        if self.current is None:
            self._new_question()
        if self.current is None:  # NAMED_COLORS vazio — nao deve ocorrer
            return frame, None
        name, bgr, combo = self.current

        bg = np.zeros_like(frame)
        cv2.rectangle(bg, (0, 0), (w, h), (10, 10, 22), -1)
        cv2.addWeighted(bg, 0.72, frame, 0.28, 0, frame)

        # Circulo da cor com pulso animado
        cx, cy, r = w // 2, h // 2 - 52, 100
        pulse = int(5 * abs(math.sin(time.time() * 2.8)))
        cv2.circle(frame, (cx, cy), r + pulse, bgr, 7)
        cv2.circle(frame, (cx, cy), r, bgr, -1)
        cv2.circle(frame, (cx, cy), r + 3, (255, 255, 255), 3)

        cv2.putText(frame, "Gesticule a combinacao de dedos para esta cor:",
                    (w//2 - 310, cy - r - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, name, (cx - 52, cy + r + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, bgr, 3, cv2.LINE_AA)

        hint = self._combo_hint(combo)
        cv2.putText(frame, f"Dica: {hint}", (w//2 - 200, cy + r + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Placar: {self.score}/{self.total}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
        sc = (0, 255, 100) if self.streak >= 3 else (190, 190, 190)
        cv2.putText(frame, f"Seq: {self.streak}  (record: {self.max_streak})",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc, 1, cv2.LINE_AA)

        tip = None
        action = None
        states_hud = None

        if lm_list:
            lm = lm_list[0]
            label = get_hand_label(hd_list)
            states = get_finger_states(lm, label)
            states_hud = states

            raw = get_index_tip(lm, frame.shape)
            if raw:
                self.smooth_tip = smooth_pt(self.smooth_tip, raw)
                tip = self.smooth_tip
                cv2.circle(frame, tip, 10, (0, 220, 255), -1)

            if time.time() > self.answer_cooldown:
                if states == combo:
                    self.score += 1
                    self.total += 1
                    self.streak += 1
                    self.max_streak = max(self.max_streak, self.streak)
                    self.result_msg = f"CORRETO! +1  (seq:{self.streak})"
                    self.result_ok = True
                    self.result_time = time.time()
                    self._new_question()
                elif count_extended(states) > 0:
                    self.total += 1
                    self.streak = 0
                    self.result_msg = f"Errado! Era: {hint}"
                    self.result_ok = False
                    self.result_time = time.time()
                    self._new_question()
        else:
            self.smooth_tip = None

        if self.result_msg and (time.time() - self.result_time) < 2.6:
            col = (0, 255, 80) if self.result_ok else (0, 50, 255)
            cv2.putText(frame, self.result_msg, (w//2 - 185, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

        for btn in self.buttons:
            if btn.update_hover(tip is not None and btn.contains(tip)):
                action = btn.value
            btn.draw(frame)

        if action == "next":
            self._new_question()
            action = None

        if states_hud is not None:
            draw_finger_hud(frame, states_hud, w - 155, h - 68)

        return frame, action

    @staticmethod
    def _combo_hint(combo):
        names = ["Polegar", "Indicador", "Medio", "Anelar", "Mindinho"]
        parts = [n for n, s in zip(names, combo) if s]
        return " + ".join(parts) if parts else "Nenhum"


# ─── EDU: CONTAGEM DE DEDOS ───────────────────────────────────
class EduCountMode:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.target = 0
        self.score = 0
        self.total = 0
        self.streak = 0
        self.max_streak = 0
        self.result_msg = ""
        self.result_ok = True
        self.result_time = 0.0
        self.answer_cooldown = 0.0
        self._new_question()
        # botoes: Voltar, Sair
        self.buttons = make_mode_buttons(W, H, include_next=False)
        # Voltar == buttons[-2]
        self.smooth_tip = None
        self.hold_start = None
        self.last_n = -1

    def _new_question(self):
        self.target = random.randint(0, 5)
        self.answer_cooldown = time.time() + 2.5
        self.hold_start = None
        self.last_n = -1

    def process(self, frame, lm_list, hd_list):
        h, w = frame.shape[:2]

        bg = np.zeros_like(frame)
        cv2.rectangle(bg, (0, 0), (w, h), (10, 22, 10), -1)
        cv2.addWeighted(bg, 0.72, frame, 0.28, 0, frame)

        # Numero grande com sombra
        ns = str(self.target)
        cv2.putText(frame, ns, (w//2 - 52, h//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 6.5, (40, 40, 40), 18, cv2.LINE_AA)
        cv2.putText(frame, ns, (w//2 - 52, h//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 6.5, (255, 255, 255), 10, cv2.LINE_AA)

        cv2.putText(frame, "Levante esse numero de dedos!",
                    (w//2 - 285, h//2 + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "(mantenha por 0.8s para confirmar)",
                    (w//2 - 205, h//2 + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Placar: {self.score}/{self.total}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
        sc = (0, 255, 100) if self.streak >= 3 else (190, 190, 190)
        cv2.putText(frame, f"Seq: {self.streak}  (record: {self.max_streak})",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc, 1, cv2.LINE_AA)

        tip = None
        action = None
        states_hud = None

        if lm_list:
            lm = lm_list[0]
            label = get_hand_label(hd_list)
            states = get_finger_states(lm, label)
            states_hud = states
            n = count_extended(states)

            cv2.putText(frame, f"Seus dedos: {n}", (10, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)

            raw = get_index_tip(lm, frame.shape)
            if raw:
                self.smooth_tip = smooth_pt(self.smooth_tip, raw)
                tip = self.smooth_tip
                cv2.circle(frame, tip, 10, (0, 220, 255), -1)

            if time.time() > self.answer_cooldown:
                if n == self.last_n:
                    if self.hold_start is None:
                        self.hold_start = time.time()
                    held = time.time() - self.hold_start
                    prog = min(1.0, held / 0.8)
                    bx = w//2 - 100
                    cv2.rectangle(frame, (bx, h//2 + 154), (bx + 200, h//2 + 167),
                                  (50, 50, 50), -1)
                    cv2.rectangle(frame, (bx, h//2 + 154),
                                  (bx + int(200 * prog), h//2 + 167), (0, 220, 255), -1)
                    if held >= 0.8:
                        if n == self.target:
                            self.score += 1
                            self.total += 1
                            self.streak += 1
                            self.max_streak = max(self.max_streak, self.streak)
                            self.result_msg = f"CERTO! {n} dedo(s)! seq:{self.streak}"
                            self.result_ok = True
                        else:
                            self.total += 1
                            self.streak = 0
                            self.result_msg = f"Errado! Era {self.target}"
                            self.result_ok = False
                        self.result_time = time.time()
                        self._new_question()
                else:
                    self.hold_start = None
                self.last_n = n
        else:
            self.smooth_tip = None
            self.hold_start = None
            self.last_n = -1

        if self.result_msg and (time.time() - self.result_time) < 2.6:
            col = (0, 255, 80) if self.result_ok else (0, 50, 255)
            cv2.putText(frame, self.result_msg, (w//2 - 205, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

        for btn in self.buttons:
            if btn.update_hover(tip is not None and btn.contains(tip)):
                action = btn.value
            btn.draw(frame)

        if states_hud is not None:
            draw_finger_hud(frame, states_hud, w - 155, h - 68)

        return frame, action


# ─── FISIOTERAPIA ─────────────────────────────────────────────
class PhysioExercise(Enum):
    OPEN_CLOSE   = "Abrir e Fechar Mao"
    FINGER_TOUCH = "Toque de Dedos (Pinca)"
    FINGER_WAVE  = "Onda de Dedos"


class PhysioMode:
    EXERCISES   = [PhysioExercise.OPEN_CLOSE,
                   PhysioExercise.FINGER_TOUCH,
                   PhysioExercise.FINGER_WAVE]
    REPS_TARGET = 8

    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.exercise_idx = 0
        self.reps = 0
        self.phase = "abrir"
        self.wave_idx = 0
        self.wave_state = False
        self.last_state = None
        self.result_msg = ""
        self.result_time = 0.0
        self.cooldown = 0.0
        self.session_start = time.time()
        self.rep_history = []
        # botoes: Proximo, Voltar, Sair, Salvar Sessao
        self.buttons = make_mode_buttons(W, H, include_next=True)
        # adiciona botao de salvar sessao (valor: save_physio)
        bw, bh = 138, 44
        y = H - bh - 70
        self.btn_save = Button(10, y, bw, bh, "Salvar Sessao", (20, 80, 20), value="save_physio")
        self.buttons.insert(0, self.btn_save)
        # Proximo == buttons[1] now
        self.btn_next = [b for b in self.buttons if b.value == "next"][0]
        self.smooth_tip = None

    @property
    def current_exercise(self):
        return self.EXERCISES[self.exercise_idx]

    def _next_exercise(self):
        self.exercise_idx = (self.exercise_idx + 1) % len(self.EXERCISES)
        self.reps = 0
        self.phase = "abrir"
        self.wave_idx = 0
        self.wave_state = False
        self.last_state = None
        self.rep_history = []
        self.cooldown = time.time() + 1.0

    def _add_rep(self):
        now = time.time()
        self.rep_history = [t for t in self.rep_history if now - t < 10]
        self.rep_history.append(now)
        self.reps += 1
        self.result_msg = f"Rep {self.reps}!"
        self.result_time = now

    def process(self, frame, lm_list, hd_list):
        h, w = frame.shape[:2]

        bg = np.zeros_like(frame)
        cv2.rectangle(bg, (0, 0), (w, h), (22, 10, 10), -1)
        cv2.addWeighted(bg, 0.65, frame, 0.35, 0, frame)

        ex = self.current_exercise
        cv2.putText(frame, "FISIOTERAPIA", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, ex.value, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Barra de reps
        prog = min(1.0, self.reps / self.REPS_TARGET)
        bar_col = (0, 255, 80) if self.reps >= self.REPS_TARGET else (0, 200, 80)
        cv2.rectangle(frame, (10, 76), (w - 10, 96), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 76), (10 + int((w - 20) * prog), 96), bar_col, -1)
        cv2.putText(frame, f"{self.reps}/{self.REPS_TARGET} reps", (10, 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Ex {self.exercise_idx+1}/{len(self.EXERCISES)}",
                    (w - 115, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (155, 155, 155), 1, cv2.LINE_AA)

        # Timer de sessao
        m, s = divmod(int(time.time() - self.session_start), 60)
        cv2.putText(frame, f"Sessao: {m:02d}:{s:02d}", (10, h - 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 170, 170), 1, cv2.LINE_AA)

        for i, line in enumerate(self._get_instructions(ex)):
            cv2.putText(frame, line, (10, 140 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

        tip = None
        action = None
        states_hud = None

        if lm_list and time.time() > self.cooldown:
            lm = lm_list[0]
            label = get_hand_label(hd_list)
            states = get_finger_states(lm, label)
            states_hud = states
            n = count_extended(states)

            raw = get_index_tip(lm, frame.shape)
            if raw:
                self.smooth_tip = smooth_pt(self.smooth_tip, raw)
                tip = self.smooth_tip

            if ex == PhysioExercise.OPEN_CLOSE:
                self._detect_open_close(states, n)
            elif ex == PhysioExercise.FINGER_TOUCH:
                self._detect_finger_touch(lm, frame.shape)
            elif ex == PhysioExercise.FINGER_WAVE:
                self._detect_finger_wave(states)

            self._draw_finger_indicators(frame, states, w, h)

            if tip:
                cv2.circle(frame, tip, 10, (0, 220, 255), -1)
        else:
            self.smooth_tip = None

        # Ritmo
        if len(self.rep_history) >= 2:
            span = self.rep_history[-1] - self.rep_history[0]
            if span > 0:
                rpm = len(self.rep_history) / span * 60
                cv2.putText(frame, f"Ritmo: {rpm:.0f} rep/min", (w - 230, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 0), 1, cv2.LINE_AA)

        # Celebracao ao completar
        if self.reps >= self.REPS_TARGET:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, h), (0, 55, 0), -1)
            cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
            cv2.putText(frame, "EXERCICIO CONCLUIDO!", (w//2 - 245, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 80), 3, cv2.LINE_AA)
            cv2.putText(frame, "Avance para o proximo exercicio!",
                        (w//2 - 205, h//2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 255, 180), 2, cv2.LINE_AA)

        if self.result_msg and (time.time() - self.result_time) < 1.2:
            cv2.putText(frame, self.result_msg, (w//2 - 65, h - 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 150), 3, cv2.LINE_AA)

        for btn in self.buttons:
            if btn.update_hover(tip is not None and btn.contains(tip)):
                action = btn.value
            btn.draw(frame)

        if action == "next":
            self._next_exercise()
            action = None
        elif action == "save_physio":
            fname = self.save_session()
            self.result_msg = f"Sessao salva: {fname}"
            self.result_time = time.time()
            _beep_ok()
            action = None

        if states_hud is not None:
            draw_finger_hud(frame, states_hud, w - 155, h - 68)

        return frame, action

    def _detect_open_close(self, states, n):
        if self.phase == "abrir" and n >= 4:
            self.phase = "fechar"
            self.cooldown = time.time() + 0.4
        elif self.phase == "fechar" and n == 0:
            self.phase = "abrir"
            self._add_rep()
            self.cooldown = time.time() + 0.4

    def _detect_finger_touch(self, lm, frame_shape):
        h, w = frame_shape[:2]
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)
        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        touching = math.hypot(tx - ix, ty - iy) < 40
        if touching and self.last_state != "touching":
            self._add_rep()
            self.cooldown = time.time() + 0.5
        self.last_state = "touching" if touching else "open"

    def _detect_finger_wave(self, states):
        wave = [1, 2, 3, 4]
        exp = wave[self.wave_idx]
        if states[exp] and not self.wave_state:
            self.wave_state = True
        elif not states[exp] and self.wave_state:
            self.wave_state = False
            self.wave_idx += 1
            if self.wave_idx >= len(wave):
                self.wave_idx = 0
                self._add_rep()

    def _draw_finger_indicators(self, frame, states, w, h):
        labels = ["P", "I", "M", "A", "Mi"]
        by = h - 163
        for i, (label, up) in enumerate(zip(labels, states)):
            x = 18 + i * 52
            col = (0, 220, 100) if up else (58, 58, 58)
            cv2.rectangle(frame, (x, by - 32), (x + 44, by), col, -1)
            cv2.rectangle(frame, (x, by - 32), (x + 44, by), (145, 145, 145), 1)
            cv2.putText(frame, label, (x + 12, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Fase: {self.phase.upper()}", (310, h - 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _get_instructions(ex):
        if ex == PhysioExercise.OPEN_CLOSE:
            return [
                "1. Abra a mao completamente (4+ dedos levantados)",
                "2. Feche em punho (0 dedos levantados)",
                "3. Repita suavemente, sem forcar",
            ]
        if ex == PhysioExercise.FINGER_TOUCH:
            return [
                "1. Toque o polegar na ponta do indicador",
                "2. Forme uma pinca e solte",
                "3. Movimento lento e controlado",
            ]
        if ex == PhysioExercise.FINGER_WAVE:
            return [
                "1. Levante o indicador, abaixe",
                "2. Levante o medio, abaixe",
                "3. Repita com anelar e mindinho em sequencia",
            ]
        return []

    def save_session(self):
        """Salva um resumo da sessao fisioterapia em CSV e retorna o nome do arquivo."""
        now = int(time.time())
        fname = f"physio_session_{now}.csv"
        try:
            with open(fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "exercise_idx", "exercise", "reps", "rep_times"])
                for i, t in enumerate(self.rep_history):
                    writer.writerow([now, self.exercise_idx, self.current_exercise.value, self.reps, ";".join(str(x) for x in self.rep_history)])
        except Exception:
            return "erro_salvar"
        return fname


# ─── APP PRINCIPAL ────────────────────────────────────────────
class Spectra:
    def __init__(self):
        ensure_model(MODEL_PATH)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.app_mode = AppMode.MENU
        self.handlers: dict = {}
        self.fps = {"start": None, "n": 0, "val": 0.0}

    def _init_handlers(self, W, H):
        self.handlers = {
            AppMode.MENU:       MenuMode(W, H),
            AppMode.PAINT:      PaintMode(W, H),
            AppMode.EDU_COLORS: EduColorsMode(W, H),
            AppMode.EDU_COUNT:  EduCountMode(W, H),
            AppMode.PHYSIO:     PhysioMode(W, H),
        }

    def _detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)
        return (res.hand_landmarks or []), (res.handedness or [])

    def _draw_fps(self, frame):
        if self.fps["start"] is None:
            self.fps["start"] = time.time()
        self.fps["n"] += 1
        elapsed = time.time() - self.fps["start"]
        if elapsed > 1.0:
            self.fps["val"] = self.fps["n"] / elapsed
            self.fps["n"] = 0
            self.fps["start"] = time.time()
        cv2.putText(frame, f"FPS:{self.fps['val']:.0f}",
                    (frame.shape[1] - 78, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (110, 110, 110), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Nao foi possivel abrir a webcam.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Nao foi possivel ler o primeiro frame.")
        H, W = cv2.flip(first, 1).shape[:2]
        self._init_handlers(W, H)

        print("SPECTRA iniciado.")
        print("Aponte o indicador e segure sobre os botoes para navegar.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)

                lm_list, hd_list = self._detect(frame)
                handler = self.handlers[self.app_mode]
                frame_out, action = handler.process(frame, lm_list, hd_list)

                self._draw_fps(frame_out)
                cv2.imshow("SPECTRA", frame_out)
                cv2.waitKey(1)

                if action == "quit":
                    break
                elif action == "menu":
                    self.app_mode = AppMode.MENU
                    self.handlers[AppMode.MENU] = MenuMode(W, H)
                elif isinstance(action, AppMode):
                    self.app_mode = action

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print("SPECTRA encerrado.")


if __name__ == "__main__":
    Spectra().run()