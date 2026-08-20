"""
Шаг 2b пайплайна: выгрузка публичных матчей с ПОЛНЫМИ драфтами — данные
для обучения L3 (и для честного расчёта матрицы синергии в L1).

Почему отдельный эндпоинт:
- /heroes/{id}/matchups (шаг 2) отдаёт только попарные агрегаты и не
  позволяет восстановить, какие 10 героев стояли в одном матче;
- /publicMatches отдаёт radiant_team / dire_team / radiant_win пачками по
  100 матчей за запрос — это единственный дешёвый способ получить драфты
  целиком, без скачивания каждого матча по одному.

Пагинация идёт назад по времени через less_than_match_id, результат
пишется в JSONL (append-friendly: докачивать можно частями, не теряя
уже скачанное).

Запуск:
    python -m src.data_pipeline.fetch_public_matches
"""
import json
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from src.data_pipeline.rank_tiers import describe_window, sample_window

CONFIG_PATH = Path("configs/config.yaml")

PAGE_SIZE = 100  # столько матчей отдаёт /publicMatches за один запрос


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_team(value) -> list[int]:
    """radiant_team приходит либо списком int, либо строкой "1,2,3,4,5"."""
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return []


def is_usable(
    match: dict,
    min_rank_tier: int | None,
    game_modes: list[int] | None,
    max_rank_tier: int | None = None,
) -> bool:
    """Матч годится, если драфт полон и он попадает в заявленную выборку.

    Бракет и режим фиксируются здесь осознанно: в работе нужно указать,
    на какой популяции обучалась модель (см. раздел про смещение данных
    в README) — фильтр и есть это определение популяции.
    """
    radiant, dire = parse_team(match.get("radiant_team")), parse_team(match.get("dire_team"))
    if len(radiant) != 5 or len(dire) != 5:
        return False
    if len(set(radiant) | set(dire)) != 10:  # битые записи с дублями героев
        return False
    if match.get("radiant_win") is None:
        return False
    if game_modes and match.get("game_mode") not in game_modes:
        return False
    if min_rank_tier is not None or max_rank_tier is not None:
        tier = match.get("avg_rank_tier")
        if tier is None:
            return False
        if min_rank_tier is not None and tier < min_rank_tier:
            return False
        if max_rank_tier is not None and tier > max_rank_tier:
            return False
    return True


def normalize(match: dict) -> dict:
    """Оставляем только то, что нужно для обучения — файл и так будет крупным."""
    return {
        "match_id": int(match["match_id"]),
        "start_time": match.get("start_time"),
        "radiant_win": bool(match["radiant_win"]),
        "avg_rank_tier": match.get("avg_rank_tier"),
        "game_mode": match.get("game_mode"),
        "duration": match.get("duration"),
        "radiant_team": parse_team(match.get("radiant_team")),
        "dire_team": parse_team(match.get("dire_team")),
    }


def fetch_public_matches(
    base_url: str,
    out_path: Path,
    target_matches: int,
    min_rank_tier: int | None,
    game_modes: list[int] | None,
    max_rank_tier: int | None = None,
    api_key: str | None = None,
    sleep_sec: float = 1.0,
) -> int:
    """Качает матчи назад по времени, пока не наберётся target_matches годных.

    Уже скачанные match_id читаются из существующего файла и пропускаются —
    так докачка не создаёт дублей, а обучающая выборка не «поедет» из-за
    повторов одного и того же матча.
    """
    seen: set[int] = set()
    less_than: int | None = None
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["match_id"])
        if seen:
            less_than = min(seen)
            print(f"Найдено {len(seen)} уже скачанных матчей, продолжаю с match_id < {less_than}")

    session = requests.Session()
    added = 0
    empty_pages = 0
    with out_path.open("a", encoding="utf-8") as out, tqdm(
        total=target_matches, desc="Качаю publicMatches"
    ) as bar:
        while added < target_matches:
            params: dict[str, object] = {}
            if less_than is not None:
                params["less_than_match_id"] = less_than
            # Бракет фильтруем на стороне API: иначе страница из 100 матчей
            # даёт 5-15 подходящих, и дневной лимит запросов уходит впустую.
            # Клиентская проверка в is_usable остаётся страховкой.
            if min_rank_tier is not None:
                params["min_rank"] = min_rank_tier
            if max_rank_tier is not None:
                params["max_rank"] = max_rank_tier
            if api_key:
                params["api_key"] = api_key

            resp = session.get(f"{base_url}/publicMatches", params=params, timeout=30)
            resp.raise_for_status()
            page = resp.json()
            if not page:
                print("API вернул пустую страницу — останавливаюсь")
                break

            page_ids = [int(m["match_id"]) for m in page if m.get("match_id")]
            less_than = min(page_ids) if page_ids else less_than

            kept = 0
            for match in page:
                mid = int(match["match_id"])
                if mid in seen or not is_usable(match, min_rank_tier, game_modes, max_rank_tier):
                    continue
                seen.add(mid)
                out.write(json.dumps(normalize(match), ensure_ascii=False) + "\n")
                kept += 1
                added += 1
                bar.update(1)
                if added >= target_matches:
                    break

            # Жёсткий фильтр по бракету может выдавать длинные пустые серии;
            # это не ошибка, но и крутиться вечно смысла нет.
            empty_pages = empty_pages + 1 if kept == 0 else 0
            if empty_pages >= 50:
                print("50 страниц подряд без подходящих матчей — проверь min_rank_tier/game_modes")
                break

            out.flush()
            time.sleep(sleep_sec)  # rate limit бесплатного тарифа: 60 запросов/мин

    return added


def main():
    cfg = load_config()
    base_url = cfg["opendota"]["base_url"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    pm_cfg = cfg["opendota"].get("public_matches", {})
    out_path = raw_dir / cfg["data"].get("public_matches_file", "public_matches.jsonl")

    # Если задан MMR игрока, бракет выборки считается из него: модель должна
    # учиться на популяции, в которой игрок реально играет.
    min_rank = pm_cfg.get("min_rank_tier")
    max_rank = pm_cfg.get("max_rank_tier")
    player_mmr = pm_cfg.get("player_mmr")
    if player_mmr:
        spread = pm_cfg.get("rank_spread_medals", 1)
        min_rank, max_rank = sample_window(int(player_mmr), spread)
        print(describe_window(int(player_mmr), spread))

    added = fetch_public_matches(
        base_url=base_url,
        out_path=out_path,
        target_matches=pm_cfg.get("target_matches", 50_000),
        min_rank_tier=min_rank,
        max_rank_tier=max_rank,
        game_modes=pm_cfg.get("game_modes"),
        api_key=cfg["opendota"].get("api_key") or None,
        sleep_sec=pm_cfg.get("sleep_sec", 1.0),
    )
    print(f"Добавлено {added} матчей -> {out_path}")


if __name__ == "__main__":
    main()
