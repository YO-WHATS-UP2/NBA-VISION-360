import json
import random
from pathlib import Path
import asyncio
import edge_tts


class StatLineGenerator:
    def __init__(self, bios_path, stats_path, templates_path):
        self.bios = self._load_json(Path(bios_path)).get("bios", {})
        self.raw_stats = self._load_json(Path(stats_path)).get("stats", [])
        self.templates = self._load_json(Path(templates_path))
        self.slug_to_name = {slug: bio.get("name", slug) for slug, bio in self.bios.items()}

    def _load_json(self, path):
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found → {path}")
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def get_enhanced_stats(self, slug, season=None):
        for stat in self.raw_stats:
            if stat.get("slug") != slug:
                continue
            if season and str(stat.get("season")) not in str(season):
                continue

            gp = stat.get("gp", 1) or 1
            tp = stat.get("tp", 0)
            tpa = stat.get("tpa", 1) or 1

            return {
                "player": self.slug_to_name.get(slug, slug),
                "season": stat.get("season"),
                "PPG": round(stat.get("pts", 0) / gp, 1),
                "APG": round(stat.get("ast", 0) / gp, 1),
                "TRBPG": round((stat.get("orb", 0) + stat.get("drb", 0)) / gp, 1),
                "PER": round(stat.get("per", 0), 1),
                "3P%": round(tp / tpa * 100, 1) if tpa else 0.0,
                "pts": stat.get("pts", 0),
                "ast": stat.get("ast", 0),
                "drb": stat.get("drb", 0),
                "blk": stat.get("blk", 0),
                "stl": stat.get("stl", 0),
                "vorp": stat.get("vorp", 0),
                "gp": gp
            }
        return None

    def generate_commentary(self, slug, category=None, season=None):
        stats = self.get_enhanced_stats(slug, season)
        if not stats:
            return f"Stats not found for {slug} in season {season}"

        templates = self.templates.get(category) if category else random.choice(list(self.templates.values()))
        if not templates:
            return f"No templates found for category: {category}"

        template = random.choice(templates)
        try:
            return template.format(**stats)
        except KeyError as e:
            return f"Missing stat key: {e.args[0]} in template"


async def generate_voice_files(text, output_basename="commentary"):
    styles = ["cheerful", "excited", "angry", "sad", "serious"]
    voice = "en-US-GuyNeural"

    for style in styles:
        ssml = f"""

    
            {text}
    

"""
        output_path = f"{output_basename}_{style}.mp3"
        print(f"🔊 Saving: {output_path}")
        communicate = edge_tts.Communicate(ssml)
        await communicate.save(output_path)


if __name__ == "__main__":
    bios_path = "D:/basketball ml - Copy/real-player-data.basketball.json"
    stats_path = "D:/basketball ml - Copy/real-player-stats.basketball.json"
    templates_path = "D:/basketball ml - Copy/commentary/filler_templates.json"

    generator = StatLineGenerator(bios_path, stats_path, templates_path)
    commentary = generator.generate_commentary("jamesle01", category="Offense", season="2004")
    print("📣 Commentary Text:", commentary)

    asyncio.run(generate_voice_files(commentary))
