"""
Профиль игрока (L2): winrate по героям, любимые роли, опыт (кол-во игр).
Строится из player_{id}_matches.json (см. fetch_matches.py).

Запуск:
    python -m src.features.player_profile
"""
import json
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_player_profile(matches: list[dict]) -> pd.DataFrame:
    """
    Каждый матч в ответе /players/{id}/matches содержит hero_id и player_slot
    + radiant_win, по которым можно определить win/loss для этого игрока.
    """
    rows = []
    for m in matches:
        is_radiant = m["player_slot"] < 128
        won = m["radiant_win"] == is_radiant
        rows.append({"hero_id": m["hero_id"], "win": int(won)})

    df = pd.DataFrame(rows)
    profile = df.groupby("hero_id").agg(games=("win", "count"), wins=("win", "sum"))
    profile["winrate"] = profile["wins"] / profile["games"]
    return profile.reset_index()


def main():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    account_id = cfg["opendota"].get("player_account_id")

    if not account_id:
        print("player_account_id не задан в config.yaml — нечего строить")
        return

    matches_path = raw_dir / f"player_{account_id}_matches.json"
    matches = json.loads(matches_path.read_text(encoding="utf-8"))

    profile = build_player_profile(matches)
    out_path = processed_dir / f"player_{account_id}_profile.parquet"
    profile.to_parquet(out_path)
    print(f"Профиль игрока сохранён -> {out_path} ({len(profile)} героев)")


if __name__ == "__main__":
    main()
