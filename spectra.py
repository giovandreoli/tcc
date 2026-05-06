import argparse
import os
import time
import urllib.request
from pathlib import Path
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Tasks API (versão 0.10.35+)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"


class DrawMode(Enum):
    """Modos de desenho."""
    ERASER = "Borracha"
    COLOR1 = "Vermelho"
    COLOR2 = "Verde"
    COLOR3 = "Azul"
    COLOR4 = "Amarelo"


# Cores em BGR (OpenCV)
COLORS = {
    DrawMode.COLOR1: (0, 0, 255),      # Vermelho
    DrawMode.COLOR2: (0, 255, 0),      # Verde
    DrawMode.COLOR3: (255, 0, 0),      # Azul
    DrawMode.COLOR4: (0, 255, 255),    # Amarelo
}

# Mapeamento: dedo estendido -> modo/cor
FINGER_TO_MODE = {
    "index": DrawMode.COLOR1,
    "middle": DrawMode.COLOR2,
    "ring": DrawMode.COLOR3,
    "pinky": DrawMode.COLOR4,
    "thumb": DrawMode.ERASER,
}

UI_BAR_HEIGHT = 110
HOVER_SELECT_SECONDS = 0.8

GESTURE_HOLD_SECONDS = 1.2
SMOOTH_ALPHA = 0.35


def ensure_model(model_path: str) -> None:
    """Baixa o modelo se não existir."""
    if os.path.exists(model_path):
        print(f"✓ Modelo encontrado: {model_path}")
        return
    
    print(f"📥 Baixando modelo ({MODEL_URL})...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"✓ Modelo baixado com sucesso")
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar modelo: {e}")


def get_finger_position(hand_landmarks, landmark_index: int, frame_shape):
    """Obtém posição normalizada de um landmark."""
    landmark = hand_landmarks[landmark_index]
    h, w = frame_shape[:2]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return (x, y)


def is_finger_extended(hand_landmarks, finger_tip: int, finger_pip: int) -> bool:
    """Verifica se um dedo está estendido."""
    return hand_landmarks[finger_tip].y < hand_landmarks[finger_pip].y


def count_fingers(hand_landmarks) -> int:
    """Conta quantos dedos estão estendidos."""
    # Landmarks: 0=pulso, 4=polegar, 8=indicador, 12=médio, 16=anelar, 20=mindinho
    fingers = [
        is_finger_extended(hand_landmarks, 4, 3),   # Polegar
        is_finger_extended(hand_landmarks, 8, 6),   # Indicador
        is_finger_extended(hand_landmarks, 12, 10), # Médio
        is_finger_extended(hand_landmarks, 16, 14), # Anelar
        is_finger_extended(hand_landmarks, 20, 18), # Mindinho
    ]
    return sum(fingers)


class DrawingCanvas:
    """Canvas para desenho."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.mode = DrawMode.COLOR1
        self.brush_size = 8
        self.eraser_size = 40

    def draw_point(self, pt: tuple, size: int, color: tuple):
        """Desenha um ponto no canvas."""
        if self.mode == DrawMode.ERASER:
            cv2.circle(self.canvas, pt, self.eraser_size, (255, 255, 255), -1)
        else:
            cv2.circle(self.canvas, pt, size, color, -1)

    def draw_line(self, pt1: tuple, pt2: tuple, size: int, color: tuple):
        """Desenha uma linha no canvas."""
        if self.mode == DrawMode.ERASER:
            cv2.line(self.canvas, pt1, pt2, (255, 255, 255), self.eraser_size)
        else:
            cv2.line(self.canvas, pt1, pt2, color, size)

    def clear(self):
        """Limpa o canvas."""
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

    def set_mode(self, mode: DrawMode):
        """Muda o modo de desenho."""
        self.mode = mode

    def get_mode_display(self) -> str:
        """Retorna texto para exibir modo atual."""
        return f"Modo atual: {self.mode.value}"

    @staticmethod
    def get_mode_hint() -> str:
        """Retorna instrucoes de uso sem teclado."""
        return "Use um dedo por vez: indicador=vermelho, medio=verde, anelar=azul, mindinho=amarelo, polegar=borracha"


class HandLandmarkDetector:
    """Detector de landmarks de mão com MediaPipe Tasks."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_num_hands: int = 2,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        output_video: str | None = None,
        drawing_mode: bool = False,
    ):
        """Inicializa o detector."""
        self.model_path = model_path
        self.max_num_hands = max_num_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.output_video = output_video
        self.drawing_mode = drawing_mode
        self.fps_clock = {"start": None, "frames": 0, "fps": 0.0}
        self.prev_tip_pos: tuple[int, int] | None = None
        self.smooth_tip_pos: tuple[int, int] | None = None
        self.active_tip_index: int | None = None
        self.command_name: str | None = None
        self.command_start_time: float | None = None
        self.command_progress: float = 0.0

        # Baixa modelo se necessário
        ensure_model(model_path)

        # Inicializa detector
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_tracking_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Canvas de desenho (será inicializado depois com tamanho correto)
        self.canvas = None

    def get_single_extended_finger(self, hand_landmarks) -> tuple[str | None, int | None]:
        """Retorna qual dedo unico esta estendido (ou None se nenhum/mais de um)."""
        extended = self.get_finger_states(hand_landmarks)
        up = [name for name, is_up in extended.items() if is_up]
        tip_idx = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        if len(up) == 1:
            return up[0], tip_idx[up[0]]
        return None, None

    def get_finger_states(self, hand_landmarks) -> dict[str, bool]:
        """Retorna estado estendido/recolhido dos dedos."""
        return {
            "thumb": is_finger_extended(hand_landmarks, 4, 3),
            "index": is_finger_extended(hand_landmarks, 8, 6),
            "middle": is_finger_extended(hand_landmarks, 12, 10),
            "ring": is_finger_extended(hand_landmarks, 16, 14),
            "pinky": is_finger_extended(hand_landmarks, 20, 18),
        }

    def detect_command_gesture(self, states: dict[str, bool]) -> str | None:
        """Detecta gestos de comando (sem teclado)."""
        # Salvar: indicador + medio
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            return "save"
        # Limpar: mao aberta (4 dedos principais)
        if states["index"] and states["middle"] and states["ring"] and states["pinky"]:
            return "clear"
        # Sair: punho fechado (nenhum dedo)
        if not any(states.values()):
            return "exit"
        return None

    def apply_command(self, command: str) -> None:
        """Aplica comando por gesto."""
        if not self.canvas:
            return
        if command == "save":
            filename = f"drawing_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.canvas.canvas)
            print(f"Desenho salvo: {filename}")
        elif command == "clear":
            self.canvas.clear()
            print("Canvas limpo")
        elif command == "exit":
            raise KeyboardInterrupt()

    def update_command_hold(self, command: str | None) -> None:
        """Controla confirmacao por tempo para evitar falsos positivos."""
        if command is None:
            self.command_name = None
            self.command_start_time = None
            self.command_progress = 0.0
            return

        if self.command_name != command:
            self.command_name = command
            self.command_start_time = time.time()
            self.command_progress = 0.0
            return

        if self.command_start_time is None:
            self.command_start_time = time.time()

        elapsed = time.time() - self.command_start_time
        self.command_progress = min(1.0, elapsed / GESTURE_HOLD_SECONDS)
        if elapsed >= GESTURE_HOLD_SECONDS:
            cmd = self.command_name
            self.command_name = None
            self.command_start_time = None
            self.command_progress = 0.0
            if cmd:
                self.apply_command(cmd)

    def smooth_point(self, pt: tuple[int, int]) -> tuple[int, int]:
        """Aplica suavizacao exponencial para reduzir tremor do cursor."""
        if self.smooth_tip_pos is None:
            self.smooth_tip_pos = pt
            return pt
        sx, sy = self.smooth_tip_pos
        nx = int((1.0 - SMOOTH_ALPHA) * sx + SMOOTH_ALPHA * pt[0])
        ny = int((1.0 - SMOOTH_ALPHA) * sy + SMOOTH_ALPHA * pt[1])
        self.smooth_tip_pos = (nx, ny)
        return self.smooth_tip_pos

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, object]:
        """Processa um frame e detecta landmarks."""
        # Inicializa canvas na primeira vez
        if self.canvas is None and self.drawing_mode:
            h, w = frame.shape[:2]
            self.canvas = DrawingCanvas(w, h)

        # Converte para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cria imagem MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detecta landmarks
        result = self.detector.detect(mp_image)
        
        frame_annotated = frame.copy()
        h, w = frame.shape[:2]
        detected_command = None

        # Desenha landmarks e lógica de desenho
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                # Lista de conexões das mãos
                HAND_CONNECTIONS = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
                    (5, 6), (6, 7), (7, 8),          # Indicador
                    (9, 10), (10, 11), (11, 12),    # Dedo médio
                    (13, 14), (14, 15), (15, 16),   # Dedo anelar
                    (17, 18), (18, 19), (19, 20),   # Mindinho
                    (0, 5), (5, 9), (9, 13), (13, 17),  # Palma
                ]
                
                # Desenha conexões
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)
                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)
                    cv2.line(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Desenha pontos
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame_annotated, (x, y), 4, (255, 0, 0), -1)
                
                # Lógica de desenho (modo pincel)
                if self.drawing_mode and self.canvas:
                    states = self.get_finger_states(hand_landmarks)
                    detected_command = self.detect_command_gesture(states)

                    finger_name, tip_idx = self.get_single_extended_finger(hand_landmarks)

                    if finger_name and tip_idx is not None:
                        raw_tip = get_finger_position(hand_landmarks, tip_idx, frame.shape)
                        tip_pos = self.smooth_point(raw_tip)

                        self.canvas.set_mode(FINGER_TO_MODE[finger_name])
                        color = COLORS.get(self.canvas.mode, (0, 0, 255))

                        if self.active_tip_index != tip_idx:
                            self.prev_tip_pos = None

                        if self.prev_tip_pos is not None:
                            self.canvas.draw_line(
                                self.prev_tip_pos,
                                tip_pos,
                                self.canvas.brush_size,
                                color,
                            )
                        self.canvas.draw_point(tip_pos, self.canvas.brush_size, color)
                        self.prev_tip_pos = tip_pos
                        self.active_tip_index = tip_idx
                    else:
                        self.prev_tip_pos = None
                        self.smooth_tip_pos = None
                        self.active_tip_index = None
                
                # Label da mão
                hand_label = handedness[0].category_name
                conf = handedness[0].score
                
                wrist = hand_landmarks[0]
                x, y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(
                    frame_annotated,
                    f"{hand_label} ({conf:.2f})",
                    (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        else:
            self.prev_tip_pos = None
            self.smooth_tip_pos = None
            self.active_tip_index = None

        if self.drawing_mode:
            self.update_command_hold(detected_command)

        return frame_annotated, result

    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Desenha FPS no frame."""
        if self.fps_clock["start"] is None:
            self.fps_clock["start"] = time.time()

        self.fps_clock["frames"] += 1
        elapsed = time.time() - self.fps_clock["start"]

        if elapsed > 1.0:
            self.fps_clock["fps"] = self.fps_clock["frames"] / elapsed
            self.fps_clock["frames"] = 0
            self.fps_clock["start"] = time.time()

        cv2.putText(
            frame,
            f"FPS: {self.fps_clock['fps']:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return frame

    def run_drawing_mode(self) -> None:
        """Executa modo de pintura com as mãos."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError(
                "❌ Não foi possível abrir a webcam. Verifique se ela está conectada."
            )

        print("✓ Modo de Pintura Ativado!")
        print("Controles apenas pela camera:")
        print("- Cor por dedo: indicador=vermelho, medio=verde, anelar=azul, mindinho=amarelo, polegar=borracha")
        print("- Salvar: indicador + medio por 1.2s")
        print("- Limpar: mao aberta por 1.2s")
        print("- Sair: punho fechado por 1.2s")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("❌ Falha ao ler frame da webcam")
                    break

                # Espelha a imagem horizontalmente para melhor experiência
                frame = cv2.flip(frame, 1)

                frame_annotated, result = self.process_frame(frame)
                frame_annotated = self.draw_fps(frame_annotated)

                # Composição: mistura o canvas com a câmera
                if self.canvas:
                    alpha = 0.7
                    frame_annotated = cv2.addWeighted(
                        self.canvas.canvas, alpha,
                        frame_annotated, 1 - alpha,
                        0
                    )

                    cv2.putText(
                        frame_annotated,
                        self.canvas.get_mode_display(),
                        (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame_annotated,
                        self.canvas.get_mode_hint(),
                        (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    if self.command_name is not None:
                        msg = f"Gesto: {self.command_name.upper()} ({self.command_progress * 100:.0f}%)"
                        cv2.putText(
                            frame_annotated,
                            msg,
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 220, 255),
                            2,
                        )
                        cv2.rectangle(frame_annotated, (10, 130), (310, 142), (50, 50, 50), -1)
                        cv2.rectangle(
                            frame_annotated,
                            (10, 130),
                            (10 + int(300 * self.command_progress), 142),
                            (0, 220, 255),
                            -1,
                        )

                cv2.imshow("🎨 Pintor de Mão MediaPipe", frame_annotated)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            print("Encerrando modo de pintura (botao SAIR)")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

    def run_webcam(self) -> None:
        """Executa detecção em tempo real da webcam."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError(
                "❌ Não foi possível abrir a webcam. Verifique se ela está conectada."
            )

        print("✓ Webcam aberta com sucesso")
        print("📷 Pressione 'q' para sair, 's' para salvar frame")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        writer = None
        if self.output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))
            print(f"💾 Gravando em: {self.output_video}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("❌ Falha ao ler frame da webcam")
                    break

                frame_annotated, result = self.process_frame(frame)
                frame_annotated = self.draw_fps(frame_annotated)

                if writer:
                    writer.write(frame_annotated)

                cv2.imshow("MediaPipe Hand Landmarks Detector", frame_annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("👋 Encerrando...")
                    break
                elif key == ord("s"):
                    filename = f"hand_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame_annotated)
                    print(f"💾 Frame salvo: {filename}")

        finally:
            if writer:
                writer.release()
                print(f"✓ Vídeo salvo: {self.output_video}")
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

    def process_image(self, image_path: str) -> None:
        """Processa uma imagem estática."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Erro ao ler imagem: {image_path}")

        print(f"📷 Processando imagem: {image_path}")

        frame_annotated, result = self.process_frame(frame)

        if result.hand_landmarks:  # type: ignore
            print(f"✓ {len(result.hand_landmarks)} mão(s) detectada(s)")  # type: ignore
        else:
            print("⚠ Nenhuma mão detectada")

        cv2.imshow("MediaPipe Hand Detection", frame_annotated)
        print("Pressione qualquer tecla para fechar")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, frame_annotated)
        print(f"💾 Resultado salvo: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎯 Detector de Hand Landmarks com MediaPipe Tasks - Pintura, Detecção e Imagens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  Webcam padrão (detecção):
    python spectra.py

  Modo de Pintura com as mãos:
    python spectra.py --draw

  Webcam com mais confiança:
    python spectra.py --min-detection-conf 0.7

  Gravando vídeo:
    python spectra.py --output video_saida.mp4

  Processando imagem:
    python spectra.py --image imagem.jpg

  Modo Pintura com cores personalizadas:
    python spectra.py --draw --min-detection-conf 0.6
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--image",
        type=str,
        help="Caminho para imagem a processar",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Caminho para o modelo hand_landmarker.task",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Número máximo de mãos a detectar (padrão: 2)",
    )
    parser.add_argument(
        "--min-detection-conf",
        type=float,
        default=0.5,
        help="Confiança mínima para detecção (0-1, padrão: 0.5)",
    )
    parser.add_argument(
        "--min-tracking-conf",
        type=float,
        default=0.5,
        help="Confiança mínima para rastreamento (0-1, padrão: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Caminho para salvar vídeo de saída",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Ativa modo de pintura com as mãos",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        detector = HandLandmarkDetector(
            model_path=args.model,
            max_num_hands=args.max_hands,
            min_detection_conf=args.min_detection_conf,
            min_tracking_conf=args.min_tracking_conf,
            output_video=args.output,
            drawing_mode=args.draw,
        )

        if args.draw:
            detector.run_drawing_mode()
        elif args.image:
            detector.process_image(args.image)
        else:
            detector.run_webcam()

    except Exception as e:
        print(f"❌ Erro: {e}")
        exit(1)
