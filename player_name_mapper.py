from pathlib import Path
import json
import cv2
from collections import defaultdict
import os


class PlayerNameMapper:
    # ------------------------------------------------------------------ #
    # Initialise paths and load JSON
    # ------------------------------------------------------------------ #
    def __init__(self, player_data_path, player_stats_path):
        self.player_data_path = Path(player_data_path)
        self.player_stats_path = Path(player_stats_path)

        # --- Load raw JSON ------------------------------------------------
        self.player_data = self._load_json(self.player_data_path)["bios"]
        self.raw_player_stats = self._load_json(self.player_stats_path)["stats"]

        # --- Build mappings ----------------------------------------------
        self.slug_to_name = self._build_slug_to_name()
        self.slug_to_stats = self._build_slug_to_stats()

        # YOLO‑ID ↔ slug map that you fill at runtime
        self.yolo_id_to_slug: dict[int, str] = {}

        # --- Debug summary ------------------------------------------------
        print(f"\n✅ Loaded {len(self.slug_to_name):,} players")
        print("   Examples:",
              list(self.slug_to_name.items())[:3] or "(!) none found")
        print(f"✅ Loaded stats for {len(self.slug_to_stats):,} players")
        print("----------------------------------------------------\n")

    # ------------------------------------------------------------------ #
    # JSON helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found → {path}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        print(f"📂 {path.name}: type={type(data).__name__}, "
              f"top‑level items={len(data) if hasattr(data, '__len__') else '?'}")
        return data

    # ------------------------------------------------------------------ #
    # Build slug → name mapping
    # ------------------------------------------------------------------ #
    def _build_slug_to_stats(self):
        """Handle stats format: either dict keyed by slug OR dict of dicts with slug inside"""
        stats_map: dict[str, dict] = {}
        data = self.raw_player_stats

        if isinstance(data, dict):
            for _, stat_dict in data.items():
                if isinstance(stat_dict, dict):
                    slug = stat_dict.get("slug")
                    if slug:
                        stats_map[slug] = stat_dict

        elif isinstance(data, list):
            for stat in data:
                if isinstance(stat, dict):
                    slug = stat.get("slug")
                    if slug:
                        stats_map[slug] = stat

        return stats_map

    def _build_slug_to_name(self):
        slug_to_name: dict[str, str] = {}
        if isinstance(self.player_data, dict):
            for slug, info in self.player_data.items():      # slug is the key!
                name = info.get("name")
                if name:
                    slug_to_name[slug] = name
        return slug_to_name


    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def assign_player_to_yolo_id(self, yolo_id: int, slug: str):
        """Manually map a YOLO‑tracking ID to a player slug."""
        self.yolo_id_to_slug[yolo_id] = slug
        if slug not in self.slug_to_name:
            print(f"⚠️  slug '{slug}' not found in name map")

    def get_player_name_from_yolo_id(self, yolo_id: int) -> str:
        slug = self.yolo_id_to_slug.get(yolo_id)
        if slug:
            return self.slug_to_name.get(slug, f"Unknown ({slug})")
        return f"Unassigned (ID {yolo_id})"

    def get_player_stats_from_yolo_id(self, yolo_id: int) -> dict | None:
        slug = self.yolo_id_to_slug.get(yolo_id)
        if not slug:
            return None

        stat = self.slug_to_stats.get(slug)
        if not stat:
            return None

        # ----- Compute per‑game metrics ---------------------------------
        gp = stat.get("gp", 1) or 1          # guard /0
        pts = stat.get("pts", 0)
        ast = stat.get("ast", 0)
        per = stat.get("per", 0)
        tp, tpa = stat.get("tp", 0), stat.get("tpa", 1) or 1

        return {
            "PPG": round(pts / gp, 1),
            "APG": round(ast / gp, 1),
            "PER": round(per, 1),
            "3P%": round(tp / tpa * 100, 1)
        }

    # ------------------------------------------------------------------ #
    # Convenience for one‑off demos
    # ------------------------------------------------------------------ #
    def demo_known_players(self):
        """Hard‑codes a few YOLO IDs → slugs for quick tests."""
        hard_coded = {9: "jamesle01"}   # add more here
        for yolo_id, slug in hard_coded.items():
            self.assign_player_to_yolo_id(yolo_id, slug)
            print(f"ID {yolo_id:2d} → {self.get_player_name_from_yolo_id(yolo_id)}")
            print("     stats:", self.get_player_stats_from_yolo_id(yolo_id))


# ---------------------------------------------------------------------- #
# Example standalone usage                                               #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Adjust these paths if your JSON lives elsewhere
    
    player_data_json = "D:/basketball ml - Copy/real-player-data.basketball.json"
    player_stats_json = "D:/basketball ml - Copy/real-player-stats.basketball.json"

    mapper = PlayerNameMapper(player_data_json, player_stats_json)

    # Quick sanity check
    mapper.demo_known_players()

    # ------------------------------------------------------------------ #
    # Now you can integrate `mapper` in your YOLO loop like:
    #
    #   for det in results:
    #       yid = int(det.id)
    #       name = mapper.get_player_name_from_yolo_id(yid)
    #       stats = mapper.get_player_stats_from_yolo_id(yid)
    #       ...
    # ------------------------------------------------------------------ #
