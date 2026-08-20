"""
Шаг 3 пайплайна: из сырых матчей строим две матрицы NxN (N = число героев)
и справочник базовой статистики героев.

- synergy[i][j]   = дельта winrate героя i, когда i играет В ОДНОЙ команде с j
                     относительно базового winrate героя i
- counter[i][j]   = дельта winrate героя i, когда i играет ПРОТИВ j
                     относительно базового winrate героя i
- hero_stats      = hero_id, games, wins, winrate, pick_rate

Всё это — «feature store» для L1/L2/L3.

Источник данных: data/raw/public_matches.jsonl (см. fetch_public_matches.py).
Там есть полные драфты, поэтому синергия считается честно, из совместных
игр. Если файла нет, остаётся деградированный режим на matchups.json:
контрпики считаются, синергия остаётся нулевой (по matchups её восстановить
нельзя — эндпоинт не отдаёт составы команд).

Сглаживание (важно для методологии): доля побед пары оценивается не как
wins/games, а со сжатием к базовому winrate героя с весом prior_weight
(эмпирический байес). Пара, сыгранная 3 раза, не получает дельту +50%,
но и не выбрасывается жёстким порогом — её вклад просто затухает.

Запуск:
    python -m src.features.build_synergy_matrix
"""
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path("configs/config.yaml")

TEAM_SIZE = 5


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hero_ids(raw_dir: Path) -> list[int]:
    heroes = json.loads((raw_dir / "heroes.json").read_text(encoding="utf-8"))
    return sorted(h["id"] for h in heroes)


def load_heroes_frame(raw_dir: Path) -> pd.DataFrame:
    """Справочник героев с hero_id (а не id) — в таком виде его ждут модели."""
    heroes = json.loads((raw_dir / "heroes.json").read_text(encoding="utf-8"))
    return pd.DataFrame(heroes).rename(columns={"id": "hero_id"})


@dataclass
class MatchArrays:
    """Матчи в виде numpy-массивов ВНУТРЕННИХ индексов героев (0..N-1)."""

    radiant: np.ndarray  # (M, 5)
    dire: np.ndarray  # (M, 5)
    radiant_win: np.ndarray  # (M,) 0/1
    match_id: np.ndarray  # (M,)
    hero_ids: np.ndarray  # индекс -> hero_id

    def __len__(self) -> int:
        return len(self.match_id)

    def slice(self, sel: slice | np.ndarray) -> "MatchArrays":
        return MatchArrays(
            radiant=self.radiant[sel],
            dire=self.dire[sel],
            radiant_win=self.radiant_win[sel],
            match_id=self.match_id[sel],
            hero_ids=self.hero_ids,
        )


def load_public_matches(path: Path, hero_ids: list[int]) -> MatchArrays:
    """JSONL -> MatchArrays, отсортированные по времени (по возрастанию match_id).

    Сортировка нужна для честного временного сплита в обучении L3:
    валидация должна лежать строго «в будущем» относительно обучения,
    иначе модель подглядывает в ту же мету.
    """
    index = {h: i for i, h in enumerate(hero_ids)}
    radiant, dire, wins, mids = [], [], [], []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            r = [index[h] for h in m["radiant_team"] if h in index]
            d = [index[h] for h in m["dire_team"] if h in index]
            if len(r) != TEAM_SIZE or len(d) != TEAM_SIZE:
                continue
            radiant.append(r)
            dire.append(d)
            wins.append(int(bool(m["radiant_win"])))
            mids.append(int(m["match_id"]))

    if not mids:
        raise ValueError(f"В {path} нет пригодных матчей — проверь шаг fetch_public_matches")

    order = np.argsort(np.asarray(mids, dtype=np.int64))
    return MatchArrays(
        radiant=np.asarray(radiant, dtype=np.int64)[order],
        dire=np.asarray(dire, dtype=np.int64)[order],
        radiant_win=np.asarray(wins, dtype=np.int8)[order],
        match_id=np.asarray(mids, dtype=np.int64)[order],
        hero_ids=np.asarray(hero_ids, dtype=np.int64),
    )


def build_matrices_from_matches(
    matches: MatchArrays,
    prior_weight: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Считает synergy/counter/hero_stats по набору матчей.

    Вызывается дважды с разными данными:
      - на всех матчах — артефакты для инференса бота;
      - только на train-части — фичи для обучения L3 без утечки.
    """
    n = len(matches.hero_ids)
    n_matches = len(matches)
    win = matches.radiant_win.astype(np.float64)

    games = np.zeros(n, dtype=np.float64)
    wins = np.zeros(n, dtype=np.float64)
    pair_games = np.zeros((n, n), dtype=np.float64)
    pair_wins = np.zeros((n, n), dtype=np.float64)
    vs_games = np.zeros((n, n), dtype=np.float64)
    vs_wins = np.zeros((n, n), dtype=np.float64)

    for team, team_win in ((matches.radiant, win), (matches.dire, 1.0 - win)):
        for slot in range(TEAM_SIZE):
            np.add.at(games, team[:, slot], 1.0)
            np.add.at(wins, team[:, slot], team_win)

        # Синергия: все пары внутри одной команды, симметрично.
        for a, b in combinations(range(TEAM_SIZE), 2):
            ia, ib = team[:, a], team[:, b]
            np.add.at(pair_games, (ia, ib), 1.0)
            np.add.at(pair_games, (ib, ia), 1.0)
            np.add.at(pair_wins, (ia, ib), team_win)
            np.add.at(pair_wins, (ib, ia), team_win)

    # Контрпики: каждая пара radiant x dire, с зеркальной записью для dire.
    for a in range(TEAM_SIZE):
        for b in range(TEAM_SIZE):
            ir, idr = matches.radiant[:, a], matches.dire[:, b]
            np.add.at(vs_games, (ir, idr), 1.0)
            np.add.at(vs_wins, (ir, idr), win)
            np.add.at(vs_games, (idr, ir), 1.0)
            np.add.at(vs_wins, (idr, ir), 1.0 - win)

    base_wr = (wins + prior_weight * 0.5) / (games + prior_weight)

    # Сжатие к базовому winrate героя: при нуле совместных игр дельта = 0.
    syn = (pair_wins + prior_weight * base_wr[:, None]) / (
        pair_games + prior_weight
    ) - base_wr[:, None]
    ctr = (vs_wins + prior_weight * base_wr[:, None]) / (vs_games + prior_weight) - base_wr[:, None]
    np.fill_diagonal(syn, 0.0)
    np.fill_diagonal(ctr, 0.0)

    ids = matches.hero_ids
    synergy = pd.DataFrame(syn, index=ids, columns=ids)
    counter = pd.DataFrame(ctr, index=ids, columns=ids)
    hero_stats = pd.DataFrame(
        {
            "hero_id": ids,
            "games": games.astype(np.int64),
            "wins": wins.astype(np.int64),
            "winrate": base_wr,
            "pick_rate": games / max(n_matches, 1),
        }
    )
    return synergy, counter, hero_stats


def build_counter_from_matchups(
    raw_dir: Path, hero_ids: list[int], min_games: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Деградированный режим: контрпики из /heroes/{id}/matchups.

    Синергию этот эндпоинт восстановить не позволяет (нет составов команд),
    поэтому она остаётся нулевой — намеренно не подделываю данные.
    """
    matchups = json.loads((raw_dir / "matchups.json").read_text(encoding="utf-8"))
    counter = pd.DataFrame(0.0, index=hero_ids, columns=hero_ids)

    for hid_str, opponents in matchups.items():
        hid = int(hid_str)
        if hid not in counter.index:
            continue
        for opp in opponents:
            oid = opp["hero_id"]
            games = opp.get("games_played", 0)
            wins = opp.get("wins", 0)
            if games >= min_games and oid in counter.columns:
                counter.loc[hid, oid] = wins / games - 0.5

    synergy = pd.DataFrame(0.0, index=hero_ids, columns=hero_ids)
    return synergy, counter


def main():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    min_games = cfg["model"]["synergy_min_games"]
    prior_weight = cfg["model"].get("prior_weight", min_games)

    hero_ids = load_hero_ids(raw_dir)
    pm_path = raw_dir / cfg["data"].get("public_matches_file", "public_matches.jsonl")

    if pm_path.exists():
        matches = load_public_matches(pm_path, hero_ids)
        synergy, counter, hero_stats = build_matrices_from_matches(matches, prior_weight)
        hero_stats.to_parquet(processed_dir / "hero_stats.parquet")
        print(f"Матрицы построены по {len(matches)} матчам из {pm_path}")
        print(f"hero_stats -> {processed_dir / 'hero_stats.parquet'}")
    else:
        print(f"{pm_path} не найден — режим без синергии (только matchups.json).")
        print("Для L1 с синергией и для L3 запусти: python -m src.data_pipeline.fetch_public_matches")
        synergy, counter = build_counter_from_matchups(raw_dir, hero_ids, min_games)

    synergy.to_parquet(processed_dir / "synergy_matrix.parquet")
    counter.to_parquet(processed_dir / "counter_matrix.parquet")
    print(f"Матрицы сохранены в {processed_dir}")


if __name__ == "__main__":
    main()
