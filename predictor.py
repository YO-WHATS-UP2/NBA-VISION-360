import cv2
import numpy as np
import pandas as pd
from collections import deque, Counter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    EASY_OCR_AVAILABLE = False
    easyocr_reader = None

# ========== Utility Functions ==========

def scale_bbox(bbox, scale_x, scale_y):
    x1, y1, x2, y2 = bbox
    return (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

def scale_all_bboxes(bbox_dict, scale_x, scale_y):
    return {k: scale_bbox(v, scale_x, scale_y) for k, v in bbox_dict.items()}

def most_common_text(deque_values):
    return Counter(deque_values).most_common(1)[0][0]

def fix_clock(clock):
    return clock.replace("O", "0").replace("o", "0")

def fix_quarter(qtr):
    qtr = qtr.upper().replace("ATH", "4TH").replace("OT", "OT").strip()
    if "4TI" in qtr: return "4TH"
    return qtr

def parse_clock_to_seconds(clk_str):
    clk_str = clk_str.strip().replace("O", "0").replace("o", "0")

    if ':' in clk_str:
        try:
            mins, secs = map(float, clk_str.split(":"))
            return mins * 60 + secs
        except:
            return None
    if '.' in clk_str:
        try:
            return float(clk_str)
        except:
            return None
    if clk_str.isdigit():
        if len(clk_str) == 3:
            return float(clk_str[:2] + '.' + clk_str[2])
        elif len(clk_str) == 2:
            return float(clk_str[0] + '.' + clk_str[1])
    return None

# ========== OCR Extractor ==========

class ScoreTimeExtractor:
    def __init__(self, score_bbox_1, score_bbox_2, clock_bbox, quarter_bbox, use_easyocr=False):
        self.score_bbox_1 = score_bbox_1
        self.score_bbox_2 = score_bbox_2
        self.clock_bbox = clock_bbox
        self.quarter_bbox = quarter_bbox
        self.use_easyocr = use_easyocr and EASY_OCR_AVAILABLE

    def enhance_and_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            raise ValueError(f"ROI is empty for bbox: {bbox}")
        roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return roi, thresh

    def extract_text(self, frame, bbox, config):
        roi, thresh = self.enhance_and_crop(frame, bbox)
        if self.use_easyocr:
            result = easyocr_reader.readtext(roi)
            return result[0][1].strip() if result else ""
        else:
            return pytesseract.image_to_string(thresh, config=config).strip()

    def extract(self, frame):
        s1 = self.extract_text(frame, self.score_bbox_1, "--psm 7 -c tessedit_char_whitelist=0123456789")
        s2 = self.extract_text(frame, self.score_bbox_2, "--psm 7 -c tessedit_char_whitelist=0123456789")
        clk = self.extract_text(frame, self.clock_bbox, "--psm 7 -c tessedit_char_whitelist=0123456789:.")
        qtr = self.extract_text(frame, self.quarter_bbox, "--psm 7 -c tessedit_char_whitelist=0123456789QOTHRST")
        return s1, s2, clk, qtr

# ========== Win Probability Model (with interval logic) ==========

class LogisticWinProbabilityModel:
    def __init__(self, coef_csv_path):
        self.df = pd.read_csv(coef_csv_path)

    def compute_win_probability(self, score1, score2, time_remaining_sec, favored_by=0.5):
        try:
            t = int(np.round(time_remaining_sec))
            interval_df = self.df[(self.df['min_time'] <= t) & (self.df['max_time'] >= t)]
            if len(interval_df) < 2:
                raise ValueError(f"No coefficients for time {t}")
            pts_diff_coef = interval_df[interval_df['coefficient'] == 'pts_diff']['estimate'].values[0]
            favored_by_coef = interval_df[interval_df['coefficient'] == 'favored_by']['estimate'].values[0]
            intercept = 0  # No intercept in CSV, assume 0
            score_diff = int(score1) - int(score2)
            logit = intercept + pts_diff_coef * score_diff + favored_by_coef * favored_by
            prob = 1 / (1 + np.exp(-logit))
            return min(max(prob, 0), 1)
        except Exception as e:
            print(f"Model error: {e}")
            return None

# ========== Main Overlay Function ==========

def overlay_win_probability_on_frames(video_frames, coef_csv_path, team1_abbr="LAL", team2_abbr="LAC"):
    if not video_frames:
        return []

    ref_width, ref_height = 1920, 1080
    h, w = video_frames[0].shape[:2]
    scale_x = w / ref_width
    scale_y = h / ref_height

    raw_bboxes = {
        "score1": (1518, 846, 1596, 896),
        "score2": (1706, 849, 1784, 899),
        "clock": (1568, 908, 1646, 958),
        "quarter": (1441, 918, 1517, 963),
    }

    scaled_bboxes = scale_all_bboxes(raw_bboxes, scale_x, scale_y)
    extractor = ScoreTimeExtractor(
        scaled_bboxes["score1"],
        scaled_bboxes["score2"],
        scaled_bboxes["clock"],
        scaled_bboxes["quarter"],
        use_easyocr=True
    )
    model = LogisticWinProbabilityModel(coef_csv_path)

    buffer_len = 5
    s1_deque, s2_deque, clk_deque, qtr_deque = deque(maxlen=buffer_len), deque(maxlen=buffer_len), deque(maxlen=buffer_len), deque(maxlen=buffer_len)

    for i, frame in enumerate(video_frames):
        try:
            s1, s2, clock, qtr = extractor.extract(frame)
            s1_deque.append(s1)
            s2_deque.append(s2)
            clk_deque.append(clock)
            qtr_deque.append(qtr)

            s1_m = most_common_text(s1_deque)
            s2_m = most_common_text(s2_deque)
            clk_m = fix_clock(most_common_text(clk_deque))
            qtr_m = fix_quarter(most_common_text(qtr_deque))

            print(f"[Frame {i}] s1={s1_m}, s2={s2_m}, clock={clk_m}, quarter={qtr_m}")

            if not (s1_m.isdigit() and s2_m.isdigit()) or not clk_m:
                raise ValueError("Invalid OCR result")

            time_left = parse_clock_to_seconds(clk_m)
            if time_left is None:
                raise ValueError("Unrecognized clock format")

            quarter_map = {
                "1ST": 36 * 60, "2ND": 24 * 60, "3RD": 12 * 60, "4TH": 0,
            }
            qtr_offset = quarter_map.get(qtr_m.upper(), 0)
            time_left_sec = qtr_offset + time_left

            wp = model.compute_win_probability(s1_m, s2_m, time_left_sec)
            if wp is None:
                raise ValueError("No coefficients for this time")

            text = f"{team1_abbr} WP: {wp:.2%}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        except Exception as e:
            print(f"Frame {i} OCR error: {e}")
            cv2.putText(frame, "OCR FAIL", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    return video_frames
# Add this at the end of predictor.py
def extract_game_context_from_frame(frame, coeff_path):
    raw_bboxes = {
        "score1": (1518, 846, 1596, 896),
        "score2": (1706, 849, 1784, 899),
        "clock": (1568, 908, 1646, 958),
        "quarter": (1441, 918, 1517, 963),
    }
    ref_w, ref_h = 1920, 1080
    frame_h, frame_w = frame.shape[:2]
    scale_x, scale_y = frame_w / ref_w, frame_h / ref_h

    scaled_bboxes = {
        k: (
            int(v[0] * scale_x), int(v[1] * scale_y),
            int(v[2] * scale_x), int(v[3] * scale_y)
        ) for k, v in raw_bboxes.items()
    }

    extractor = ScoreTimeExtractor(
        scaled_bboxes["score1"],
        scaled_bboxes["score2"],
        scaled_bboxes["clock"],
        scaled_bboxes["quarter"],
        use_easyocr=True
    )
    s1, s2, clk, qtr = extractor.extract(frame)
    clk = fix_clock(clk)
    qtr = fix_quarter(qtr)

    time_left = parse_clock_to_seconds(clk)
    quarter_map = {"1ST": 36*60, "2ND": 24*60, "3RD": 12*60, "4TH": 0}
    time_left += quarter_map.get(qtr.upper(), 0)

    try:
        score_diff = float(s1) - float(s2)
    except ValueError:
        print(f"[Frame {frame_idx}] Invalid score values: s1={s1}, s2={s2}")
        return frame  # or skip overlay

    model = LogisticWinProbabilityModel(coeff_path)
    wp = model.compute_win_probability(s1, s2, time_left)

    return s1, s2, time_left, score_diff, wp

