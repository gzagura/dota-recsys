"""Отдельная база по матчам самого игрока (SQLite).

Зачем отдельно от остального пайплайна:
- личные данные не смешиваются с публичной выборкой, на которой учится
  модель: у них разное происхождение, разный жизненный цикл и разные
  причины для перевыгрузки;
- по матчам удобно спрашивать произвольные срезы (по герою, по режиму, по
  периоду), чего агрегированный parquet-профиль не позволяет: он хранит
  только сумму игр и побед;
- профиль для персонализации становится производной величиной, а не
  первоисточником, и его можно пересобрать под любой фильтр.

Запуск:
    python -m src.data_pipeline.build_player_db
"""
import json
import sqlite3
from pathlib import Path

import yaml

CONFIG_PATH = Path("configs/config.yaml")

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id     INTEGER PRIMARY KEY,
    start_time   INTEGER,
    hero_id      INTEGER NOT NULL,
    win          INTEGER NOT NULL,   -- 1 = победа этого игрока
    is_radiant   INTEGER NOT NULL,
    radiant_win  INTEGER,
    player_slot  INTEGER,
    game_mode    INTEGER,
    lobby_type   INTEGER,
    duration     INTEGER,
    kills        INTEGER,
    deaths       INTEGER,
    assists      INTEGER,
    party_size   INTEGER,
    average_rank INTEGER,
    lane_role    INTEGER            -- есть только у распарсенных матчей
);

CREATE INDEX IF NOT EXISTS idx_matches_hero ON matches(hero_id);
CREATE INDEX IF NOT EXISTS idx_matches_time ON matches(start_time);

-- Профиль по героям: то же, что попадает в персонализацию модели.
CREATE VIEW IF NOT EXISTS hero_profile AS
SELECT hero_id,
       COUNT(*)                          AS games,
       SUM(win)                          AS wins,
       CAST(SUM(win) AS REAL) / COUNT(*)  AS winrate
FROM matches
GROUP BY hero_id;
"""


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def row_from_match(match: dict) -> tuple:
    """Запись /players/{id}/matches -> строка таблицы.

    Победа определяется сопоставлением стороны игрока с исходом: слоты
    0-127 — Radiant, 128+ — Dire.
    """
    is_radiant = int(match["player_slot"]) < 128
    win = int(bool(match["radiant_win"]) == is_radiant)
    return (
        int(match["match_id"]),
        match.get("start_time"),
        int(match["hero_id"]),
        win,
        int(is_radiant),
        int(bool(match["radiant_win"])),
        match.get("player_slot"),
        match.get("game_mode"),
        match.get("lobby_type"),
        match.get("duration"),
        match.get("kills"),
        match.get("deaths"),
        match.get("assists"),
        match.get("party_size"),
        match.get("average_rank"),
        match.get("lane_role"),
    )


def build_db(matches: list[dict], db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        # INSERT OR REPLACE: докачка истории не плодит дубли, а обновляет
        # уже известные матчи (например, когда матч успели распарсить и у
        # него появился lane_role).
        conn.executemany(
            "INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [row_from_match(m) for m in matches],
        )
        conn.commit()
        return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finally:
        conn.close()


def main():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    account_id = cfg["opendota"].get("player_account_id")
    if not account_id:
        print("player_account_id не задан в config.yaml — нечего собирать")
        return

    src = raw_dir / f"player_{account_id}_matches.json"
    if not src.exists():
        raise SystemExit(f"Нет {src}. Сначала: python -m src.data_pipeline.fetch_matches")

    matches = json.loads(src.read_text(encoding="utf-8"))
    db_path = Path(cfg["data"].get("player_db", f"data/player_{account_id}.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    total = build_db(matches, db_path)

    print(f"База игрока {account_id}: {total} матчей -> {db_path}")


if __name__ == "__main__":
    main()
