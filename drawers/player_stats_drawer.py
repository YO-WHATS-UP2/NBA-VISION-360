import cv2
import math
import numpy as np
from player_name_mapper import PlayerNameMapper
class PlayerStatsDrawer:
    def __init__(self, player_mapper, yolo_id_to_slug=None):
        self.player_mapper = player_mapper
        self.yolo_id_to_slug = yolo_id_to_slug or {}
        self.frame_count = 0

    def update_and_draw(self, frame, player, ball_xy, team_in_possession, all_stats):
        self.frame_count += 1
        player_id = player["id"]
        bbox = player["bbox"]

        # Get player slug (e.g., "lebron_james")
        slug = self.yolo_id_to_slug.get(player_id) or self.player_mapper.get_slug_by_id(player_id)
        if not slug or slug not in all_stats:
            return frame

        stats = all_stats[slug]
        try:
            ppg = float(stats.get("ppg", 0))
            apg = float(stats.get("apg", 0))
            per = float(stats.get("per", 0))
            tp_pct = float(stats.get("3p_pct", 0))
        except:
            return frame  # Skip if data is malformed

        # Where to draw the ring – bottom center of the bbox
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)

        # Radius and thickness
        base_radius = 35
        thickness = 3

        # Ring colors for each stat
        colors = {
            "ppg": (255, 255, 0),     # Yellow
            "apg": (0, 255, 255),     # Cyan
            "per": (0, 255, 0),       # Green
            "3p_pct": (255, 0, 255)   # Magenta
        }

        # Normalize values for angle drawing (set reasonable caps)
        def stat_to_angle(value, max_val):
            return int(360 * min(value / max_val, 1.0))

        stats_angles = {
            "ppg": stat_to_angle(ppg, 40),
            "apg": stat_to_angle(apg, 15),
            "per": stat_to_angle(per, 35),
            "3p_pct": stat_to_angle(tp_pct, 60)
        }

        # Draw 4 stat arcs as concentric circles
        for i, (stat, angle) in enumerate(stats_angles.items()):
            radius = base_radius + i * (thickness + 3)
            start_angle = (self.frame_count * 3) % 360
            end_angle = (start_angle + angle) % 360

            cv2.ellipse(
                frame,
                (center_x, center_y),
                (radius, radius),
                0,
                start_angle,
                end_angle,
                colors[stat],
                thickness,
                lineType=cv2.LINE_AA
            )

        # Optional: display player name/slug
        # cv2.putText(frame, slug.replace("_", " ").title(), (center_x - 50, center_y - 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
