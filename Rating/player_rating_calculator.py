import json
from statistics import mean
from pathlib import Path
from datetime import datetime

from fitness import bornYear

def compute_5_season_average(slug, stats):
    player_stats = [s for s in stats if s.get("slug") == slug and isinstance(s.get("season"), int)]
    player_stats = sorted(player_stats, key=lambda x: x.get("season"), reverse=True)[:5]

    if not player_stats:
        return None

    ppg, apg, rpg, per, tp_pct, mpg = [], [], [], [], [], []
    season_wise = []

    for s in player_stats:
        gp = s.get("gp", 1) or 1
        tp = s.get("tp", 0)
        tpa = s.get("tpa", 1) or 1
        min_total = s.get("min", 0)
        mpg_val = min_total / gp

        ppg_val = s.get("pts", 0) / gp
        apg_val = s.get("ast", 0) / gp
        rpg_val = (s.get("orb", 0) + s.get("drb", 0)) / gp
        per_val = s.get("per", 0)
        tp_pct_val = (tp / tpa) * 100 if tpa else 0.0

        ppg.append(ppg_val)
        apg.append(apg_val)
        rpg.append(rpg_val)
        per.append(per_val)
        tp_pct.append(tp_pct_val)
        mpg.append(mpg_val)

        season_wise.append({
            "season": s["season"],
            "PPG": round(ppg_val, 1),
            "APG": round(apg_val, 1),
            "RPG": round(rpg_val, 1),
            "PER": round(per_val, 1),
            "3P%": round(tp_pct_val, 1),
            "MPG": round(mpg_val, 1)
        })

    return {
        "slug": slug,
        "seasons": season_wise,
        "avg": {
            "PPG": round(mean(ppg), 1),
            "APG": round(mean(apg), 1),
            "RPG": round(mean(rpg), 1),
            "PER": round(mean(per), 1),
            "3P%": round(mean(tp_pct), 1),
            "MPG": round(mean(mpg), 1)
        }
    }

def calculate_rating(avg_stats, current_stats):
    weights = {
        "PPG": 0.3,
        "APG": 0.2,
        "RPG": 0.15,
        "PER": 0.2,
        "3P%": 0.1,
        "MPG": 0.05
    }

    total_weight = sum(weights.values())
    score = 0.0

    for stat, weight in weights.items():
        avg_val = avg_stats.get(stat)
        current_val = current_stats.get(stat)

        if avg_val is not None and current_val is not None and avg_val != 0:
            ratio = current_val / avg_val
            ratio = max(0.5, min(1.5, ratio))  # clamp
            score += weight * ratio

    rating = score * (10 / total_weight)
    return round(rating, 1)

def get_player_age(slug, bios):
    for player in bios:
        if player.get("slug") == slug:
            born = player.get("born", "")
            try:
                born_year = int(born.split("-")[0])
                return datetime.now().year - born_year
            except:
                return None
    return None

if __name__ == "__main__":
    slug = "jamesle01"

    # Load bios
    bios_path = Path("D:/basketball ml - Copy - Copy/real-player-data.basketball.json")
    with bios_path.open("r", encoding="utf-8") as f:
        bios_data = json.load(f).get("players", [])

    age = get_player_age(slug, bios_data)

    # Load historical stats
    stats_path = Path("D:/basketball ml - Copy - Copy/real-player-stats.basketball.json")
    with stats_path.open("r", encoding="utf-8") as f:
        stats_data = json.load(f).get("stats", [])

    result = compute_5_season_average(slug, stats_data)

    print(f"\n--- PLAYER: {slug} ---")
    print(f"Age: {age if age is not None else 'Unknown'}")

    if result is None:
        print("No last 5 season data.")
        exit()

    for season in result["seasons"]:
        print(f"{season['season']}: {season}")

    print("\nAverages over last 5 seasons:")
    for k, v in result["avg"].items():
        print(f"{k}: {v}")

    # Load current game stats from txt
    txt_path = Path("D:/basketball ml - Copy - Copy/rating/current_game.txt")
    with txt_path.open("r", encoding="utf-8") as f:
        current_stats = json.loads(f.read())

    print("\n➤ 2025 (custom game data):")
    print(json.dumps(current_stats, indent=4))

    # Rating
    rating = calculate_rating(result["avg"], current_stats)
    print(f"\n★ Player Rating (out of 10): {rating}")

    # Injury Risk Estimation
    mpg = result["avg"]["MPG"]
    injury_risk = "Low"
    print(f"\n★ Player Age: {age if age is not None else 'Unknown'}")
    if isinstance(age, int):
        if age >= 35 and mpg >= 36:
            injury_risk = "High"
        elif age >= 30 and mpg >= 34:
            injury_risk = "Medium"
        elif mpg > 38:
            injury_risk = "High"

    print(f"⚠️ Injury Risk: {injury_risk}")
