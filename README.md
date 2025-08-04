# 🏀 NBA Vision360 – Real-Time Basketball Video Intelligence

NBA Vision360 is a real-time AI-based video analytics system that turns raw basketball footage into ESPN-style stat overlays, predictive modeling, and AI-generated commentary. Designed for analysts, fans, and coaches, it combines computer vision, machine learning, and natural language processing into one streamlined pipeline.

---

## 🎯 Features

### 👁️ Player Detection & Stat Overlay
- Uses YOLOv5 + OpenCV to detect players and overlay circular stat rings
- Highlights player metrics like **PPG, PER, 3P%**, and more

### 📊 Win Probability Prediction
- Computes win probabilities dynamically using logistic regression
- Inputs include: **quarter, time left, and score**, parsed from live OCR

### 🧮 Player Performance Rating
- Calculates game-specific ratings using historical stats, live performance, and spread-based modeling
- Includes **stamina and fatigue estimation** based on pace and minutes

### 🗣️ AI Commentary System
- Uses Edge-TTS to generate real-time play-by-play commentary
- Triggered by high-leverage moments, clutch plays, or rating spikes

### 🗺️ Tactical View Conversion
- Converts broadcast camera view to tactical court positioning using homography
- Useful for **coaching-level breakdowns**

---

## 🛠️ Tech Stack

| Layer             | Tools & Libraries                                  |
|------------------|----------------------------------------------------|
| Language         | Python                                             |
| CV/OCR           | OpenCV, YOLOv5, pytesseract                        |
| ML/Stats         | scikit-learn, pandas, NumPy                        |
| Audio            | Edge-TTS                                           |
| Visualization    | OpenCV overlays, circular stat rings               |
| Data             | Custom JSON stat feeds, CSV coefficients           |

---
## 📂 Folder Structure

NBA-Vision360/
├── main.py # Entry point
├── predictor.py # Win probability engine
├── commentary/ # Edge-TTS commentary system
├── Rating/ # Player rating logic
├── court_keypoint_detector/ # Homography & tactical conversion
├── score_detector/ # Scoreboard OCR
├── team_assigner/, trackers/ # Player tracking logic
├── heat_map_players/, speed_and_distance_calculator/
├── pass_and_interception_detector/, ball_acquisition/
├── input_videos/ # (sample game clips)
├── requirements.txt
├── README.md

---
▶️ Running the Project
bash
Copy
Edit
python main.py --input input_videos/sample_game.mp4
What you'll see:

✅ Overlaid player stats

📉 Real-time win probability

🎤 AI-generated voice commentary
---
📌 Roadmap
 Win probability + rating model

 TTS commentary engine

 Scoreboard OCR + quarter tracking

 Assist prediction engine (in progress)

 Real-time Web UI or Streamlit demo

 Support for multiple camera angles

📄 License
This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgements
ESPN for inspiring the analytics experience

Roboflow + YOLO community

NBA stat datasets (cleaned and compiled)
---
