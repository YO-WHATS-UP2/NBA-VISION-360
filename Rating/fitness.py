# fitness.py
import json

def bornYear(slug, bios_path='D:/basketball ml - Copy - Copy/real-player-data.basketball.json'):
    with open(bios_path, 'r') as f:
        bios = json.load(f)

    player_bio = bios.get(slug)
    if player_bio and 'born' in player_bio and isinstance(player_bio['born'], dict):
        return player_bio['born'].get('year')
    return None
