"""
Шаг 1 пайплайна: справочник героев (id, имя, роли, атрибут).
Источник: OpenDota /heroes и /heroStats (доп. агрегаты по pick/win rate).

Запуск:
    python -m src.data_pipeline.fetch_heroes
"""
import json
from pathlib import Path

import requests
import yaml

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_heroes(base_url: str) -> list[dict]:
    """Базовый справочник: id, localized_name, roles, primary_attr."""
    resp = requests.get(f"{base_url}/heroes", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_hero_stats(base_url: str) -> list[dict]:
    """Агрегированная статистика: pick/win по бракетам (herostats)."""
    resp = requests.get(f"{base_url}/heroStats", timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    cfg = load_config()
    base_url = cfg["opendota"]["base_url"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    heroes = fetch_heroes(base_url)
    hero_stats = fetch_hero_stats(base_url)

    (raw_dir / "heroes.json").write_text(
        json.dumps(heroes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / "hero_stats.json").write_text(
        json.dumps(hero_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сохранено: {len(heroes)} героев, {len(hero_stats)} записей hero_stats -> {raw_dir}")


if __name__ == "__main__":
    main()
