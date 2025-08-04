import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class PlayerHeatmapGenerator:
    """
    Collects per-frame tactical positions and produces a per-player heat-map tensor.
    It can also draw the heat-map as a semi-transparent overlay on the tactical court.

    Uses Turbo colormap and smooth blending (Option #2 + Option #4).
    """
    def __init__(self,
                 court_w: int = 300,
                 court_h: int = 161,
                 grid_x: int = 60,
                 grid_y: int = 32,
                 blur_sigma: float = 1.2,
                 court_image_path: str = r'D:\basketball ml - Copy\images\basketball_court.png'):
        self.court_w = court_w
        self.court_h = court_h
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.blur_sigma = blur_sigma
        self.accumulators = {}  # player_id → 2-D histogram
        self.decay = 0.985
        self.base_court = cv2.imread(court_image_path)
        if self.base_court is None:
            raise FileNotFoundError(f"Could not load court image at {court_image_path}")
        self.base_court = cv2.resize(self.base_court, (self.court_w, self.court_h))

    def add_frame_positions(self, tactical_positions: dict[int, list[float]]):
        """
        tactical_positions : {player_id: [x, y], ...}  (tactical coords)
        Applies temporal decay to create smooth heatmap accumulation over time.
        """
        for pid in set(self.accumulators) | set(tactical_positions):
            if pid not in self.accumulators:
                self.accumulators[pid] = np.zeros((self.grid_y, self.grid_x), dtype=np.float32)
            else:
                self.accumulators[pid] *= self.decay  # Apply decay

        for pid, (x, y) in tactical_positions.items():
            ix = int(np.clip(x / self.court_w * self.grid_x, 0, self.grid_x - 1))
            iy = int(np.clip(y / self.court_h * self.grid_y, 0, self.grid_y - 1))
            self.accumulators[pid][iy, ix] += 1


    def get_heatmaps(self) -> dict[int, np.ndarray]:
        """
        Returns {player_id: (H, W) float32 heatmap, already Gaussian-blurred}
        Heat-maps are normalised 0-1.
        """
        output = {}
        for pid, hist in self.accumulators.items():
            blurred = gaussian_filter(hist, self.blur_sigma)
            if blurred.max() > 0:
                blurred /= blurred.max()
            output[pid] = blurred
        return output

    def heatmap_to_bgr(self, heat: np.ndarray,
                       color_map: int = cv2.COLORMAP_TURBO,
                       alpha: float = 0.7) -> np.ndarray:
        """
        Returns BGR uint8 image the same size as tactical court (H, W, 3),
        with transparency already premultiplied by `alpha`.
        Uses smoother colour blending and Turbo colormap.
        """
        heat_resized = cv2.resize(heat, (self.court_w, self.court_h), interpolation=cv2.INTER_LINEAR)
        heat_uint8 = np.uint8(np.clip(heat_resized * 255, 0, 255))
        colour = cv2.applyColorMap(heat_uint8, color_map)
        overlay = cv2.addWeighted(colour, alpha, self.base_court.copy(), 1 - alpha, 0)
        return overlay

    def draw_on_frame(self, frame: np.ndarray, heat_bgr: np.ndarray,
                      top_left: tuple[int, int] = (20, 40)) -> np.ndarray:
        """
        Blends heat-map onto `frame` at `top_left` (x, y) for mini-court overlay.
        Smoothly blends overlay using addWeighted.
        """
        x0, y0 = top_left
        h, w = heat_bgr.shape[:2]
        x1, y1 = x0 + w, y0 + h

        if 0 <= x0 and 0 <= y0 and x1 <= frame.shape[1] and y1 <= frame.shape[0]:
            roi = frame[y0:y1, x0:x1]
            blended = cv2.addWeighted(roi, 1.0, heat_bgr, 1.0, 0)
            frame[y0:y1, x0:x1] = blended
        return frame
