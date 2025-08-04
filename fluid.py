from collections import deque
import cv2
import numpy as np

class MomentumFluidCalculator:
    def __init__(self, max_history=20):
        self.events = deque(maxlen=max_history)
        self.momentum = {"team1": 0.0, "team2": 0.0}
        self.prev_pos = 0.5

    def add_event(self, team, event_type, value=1.0, game_minute=12):
        """
        team: 'team1' or 'team2'
        event_type: one of ['3pt', '2pt', 'block', 'steal', 'turnover', 'miss', 'foul']
        value: custom weight (default 1.0)
        game_minute: used to boost clutch events (last 2 mins)
        """
        clutch_multiplier = 1.5 if game_minute <= 2 else 1.0
        weight = self._event_weight(event_type, value) * clutch_multiplier
        self.events.append({"team": team, "weight": weight})

    def _event_weight(self, event_type, value):
        base_weights = {
            "3pt": 3.0,
            "2pt": 2.0,
            "block": 1.5,
            "steal": 1.5,
            "turnover": -2.0,
            "miss": -1.0,
            "foul": -0.5
        }
        return base_weights.get(event_type, 0.0) * value

    def compute_momentum(self, decay_factor=0.9):
        self.momentum = {"team1": 0.0, "team2": 0.0}

        for i, event in enumerate(reversed(self.events)):
            decay = decay_factor ** i
            self.momentum[event["team"]] += event["weight"] * decay

        total = self.momentum["team1"] + self.momentum["team2"]
        if total == 0:
            return 0.5  # Neutral
        return self.momentum["team1"] / total

    def get_momentum_fluid_label(self):
        pos = self.compute_momentum()
        if pos > 0.7:
            return "🔥 All momentum on Team 1!"
        elif pos < 0.3:
            return "🔥 All momentum on Team 2!"
        else:
            return "⚖️ Even battle!"

    def get_momentum_bar(self, length=30):
        pos = self.compute_momentum()
        team1_len = round(pos * length)
        team2_len = length - team1_len
        return "T1 [" + "#" * team1_len + "-" * team2_len + "] T2"

    def momentum_swing_detected(self, threshold=0.25):
        new_pos = self.compute_momentum()
        swing = abs(new_pos - self.prev_pos)
        self.prev_pos = new_pos
        return swing >= threshold

    def reset(self):
        self.events.clear()
        self.momentum = {"team1": 0.0, "team2": 0.0}
        self.prev_pos = 0.5

    def draw_on_frame(self, frame):
        pos = self.compute_momentum()
        bar_width = 400
        bar_height = 20
        x, y = 50, 50
        t1_width = int(pos * bar_width)
        t2_width = bar_width - t1_width

        # Draw outer bar
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 2)

        # Fill bars
        cv2.rectangle(frame, (x, y), (x + t1_width, y + bar_height), (0, 102, 255), -1)  # Team1: Orange
        cv2.rectangle(frame, (x + t1_width, y), (x + bar_width, y + bar_height), (255, 50, 50), -1)  # Team2: Red

        # Add text
        label = self.get_momentum_fluid_label()
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
