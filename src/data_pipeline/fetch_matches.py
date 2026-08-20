"""
Шаг 2 пайплайна: выгрузка матчей для расчёта синергии/контрпиков (L1)
и, опционально, истории конкретного игрока для персонализации (L2).

Важно про источники данных:
- Для L1 (синергия по всей популяции) OpenDota отдаёт готовый эндпоинт
  /heroes/{hero_id}/matchups — win rate героя В ПАРЕ и ПРОТИВ каждого
  другого героя. Это НАМНОГО дешевле по запросам, чем скачивать сырые
  матчи и парсить их самому — этим стоит и воспользоваться в первую
  очередь.
- Сырые матчи (/matches/{id} или /players/{id}/matches) нужны только
  для L2 (персонализация под конкретного игрока).

Запуск:
    python -m src.data_pipeline.fetch_matches
"""
import json
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_matchups_for_hero(base_url: str, hero_id: int) -> list[dict]:
    """Win rate hero_id в паре / против каждого другого героя."""
    resp = requests.get(f"{base_url}/heroes/{hero_id}/matchups", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_matchups(base_url: str, raw_dir: Path):
    heroes = json.loads((raw_dir / "heroes.json").read_text(encoding="utf-8"))
    all_matchups = {}
    for hero in tqdm(heroes, desc="Скачиваю matchups по героям"):
        hid = hero["id"]
        all_matchups[hid] = fetch_matchups_for_hero(base_url, hid)
        time.sleep(1)  # уважаем rate limit бесплатного тарифа
    (raw_dir / "matchups.json").write_text(
        json.dumps(all_matchups, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Сохранено matchups для {len(all_matchups)} героев -> {raw_dir / 'matchups.json'}")


def fetch_player_matches(
    base_url: str,
    account_id: int,
    raw_dir: Path,
    limit: int = 20000,
    days: int | None = None,
):
    """История матчей конкретного игрока — для L2 (персонализация).

    Один запрос отдаёт всю историю сразу, поэтому лимит держим заведомо
    большим: 500 матчей давали медиану 2-3 игры на героя, чего для
    персонализации не хватает. days ограничивает глубину истории (в днях):
    патчи пятилетней давности — про другую игру, и в профиле их вес
    стоит осознавать.
    """
    params: dict[str, object] = {"limit": limit}
    if days:
        params["date"] = days
    resp = requests.get(
        f"{base_url}/players/{account_id}/matches",
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    matches = resp.json()
    (raw_dir / f"player_{account_id}_matches.json").write_text(
        json.dumps(matches, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Сохранено {len(matches)} матчей игрока {account_id}")


def main():
    cfg = load_config()
    base_url = cfg["opendota"]["base_url"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    fetch_all_matchups(base_url, raw_dir)

    account_id = cfg["opendota"].get("player_account_id")
    if account_id:
        pm_cfg = cfg["opendota"].get("player_matches", {})
        fetch_player_matches(
            base_url,
            account_id,
            raw_dir,
            limit=pm_cfg.get("limit", 20000),
            days=pm_cfg.get("days"),
        )
    else:
        print("player_account_id не указан в config.yaml — пропускаю сбор персональной истории (нужен для L2)")


if __name__ == "__main__":
    main()
