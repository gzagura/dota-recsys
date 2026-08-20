"""
Шаг 4 пайплайна: обучающая выборка для L3.

Как из матча получается обучающий пример
----------------------------------------
Один матч (10 героев + исход) даёт 10 примеров: каждый герой по очереди
считается «кандидатом», остальные четверо его команды — уже выбранными
союзниками, пятёрка соперника — врагами, метка = победила ли его команда.

Но в бою бот работает с НЕПОЛНЫМ драфтом (например, 2 союзника и 3 врага
уже взяты). Если учить модель только на полных составах, распределение
фич на обучении и в инференсе разойдётся, и метрики не воспроизведутся.
Поэтому для каждого примера случайно «отматывается» стадия драфта: из
союзников и врагов берётся случайное подмножество, а его размеры
(n_allies / n_enemies) подаются в модель как признаки.

Число врагов сэмплируется рядом с числом союзников (±1) — так драфт
имитирует поочерёдные пики команд, а не независимые случайные размеры.

Ограничение выборки: /publicMatches не отдаёт порядок пиков, поэтому
подмножество берётся случайно, а не «первые k по порядку выбора».
Настоящий порядок доступен только в распарсенных про-матчах — это
вынесено в направления развития.

Защита от утечки
----------------
Матрицы синергии/контрпиков и базовые winrate считаются ТОЛЬКО по
train-части (матчи, более ранние по времени). Валидация — строго более
поздние матчи. Иначе фичи примера содержали бы исход самого этого матча,
и AUC был бы завышен.

Запуск:
    python -m src.features.build_training_dataset
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.build_synergy_matrix import (
    MatchArrays,
    build_matrices_from_matches,
    load_heroes_frame,
    load_hero_ids,
    load_public_matches,
)
from src.features.draft_features import DraftFeatureBuilder

CONFIG_PATH = Path("configs/config.yaml")

TEAM_SIZE = 5
MAX_ALLIES = TEAM_SIZE - 1
MAX_ENEMIES = TEAM_SIZE

ALLY_COLUMNS = [f"ally_{i}" for i in range(MAX_ALLIES)]
ENEMY_COLUMNS = [f"enemy_{i}" for i in range(MAX_ENEMIES)]


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _random_subsets(
    full: np.ndarray, sizes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Случайное подмножество заданного размера в каждой строке.

    full:  (n, k) индексы героев
    sizes: (n,)   сколько из них оставить
    -> (n, k), лишние позиции забиты -1
    """
    n, k = full.shape
    if k == 0:
        return full
    order = np.argsort(rng.random((n, k)), axis=1)
    shuffled = np.take_along_axis(full, order, axis=1)
    keep = np.arange(k)[None, :] < sizes[:, None]
    return np.where(keep, shuffled, -1)


def _sample_stage(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Размеры видимой части драфта: союзники 0..4, враги — рядом (±1)."""
    n_allies = rng.integers(0, MAX_ALLIES + 1, size=n)
    n_enemies = np.clip(n_allies + rng.integers(-1, 2, size=n), 0, MAX_ENEMIES)
    return n_allies, n_enemies


def build_samples(
    matches: MatchArrays,
    builder: DraftFeatureBuilder,
    rng: np.random.Generator,
    stages_per_pick: int = 1,
) -> pd.DataFrame:
    """10 примеров на матч (x stages_per_pick вариантов стадии драфта)."""
    blocks = []
    n = len(matches)

    for is_radiant, (team, opponents) in enumerate(
        ((matches.dire, matches.radiant), (matches.radiant, matches.dire))
    ):
        team_won = matches.radiant_win if is_radiant else 1 - matches.radiant_win

        for slot in range(TEAM_SIZE):
            candidate = team[:, slot]
            mates = np.delete(team, slot, axis=1)  # (n, 4)

            for _ in range(stages_per_pick):
                n_allies, n_enemies = _sample_stage(n, rng)
                ally_idx = _random_subsets(mates, n_allies, rng)
                enemy_idx = _random_subsets(opponents, n_enemies, rng)

                features = builder.build_batch(
                    candidates=candidate,
                    allies=ally_idx,
                    enemies=enemy_idx,
                    is_radiant=np.full(n, is_radiant, dtype=np.int8),
                )
                features["label"] = team_won.astype(np.int8)
                features["match_id"] = matches.match_id

                # Сохраняем и сам контекст в hero_id: он нужен скрипту
                # ablation, чтобы прогнать L0/L1/L2 по тем же ситуациям.
                ids = np.append(matches.hero_ids, -1)  # -1 остаётся -1
                for j, col in enumerate(ALLY_COLUMNS):
                    features[col] = ids[ally_idx[:, j]]
                for j, col in enumerate(ENEMY_COLUMNS):
                    features[col] = ids[enemy_idx[:, j]]

                blocks.append(features)

    return pd.concat(blocks, ignore_index=True)


def build_dataset(
    matches: MatchArrays,
    heroes: pd.DataFrame,
    val_fraction: float,
    prior_weight: float,
    seed: int,
    stages_per_pick: int = 1,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Временной сплит + фичи по train-статистике. Возвращает (датасет, train-артефакты)."""
    n_train = int(len(matches) * (1.0 - val_fraction))
    if n_train < 1 or n_train >= len(matches):
        raise ValueError(f"val_fraction={val_fraction} не оставляет данных на train/val")

    train_matches = matches.slice(slice(None, n_train))
    val_matches = matches.slice(slice(n_train, None))

    synergy, counter, hero_stats = build_matrices_from_matches(train_matches, prior_weight)
    builder = DraftFeatureBuilder.from_frames(hero_stats, synergy, counter, heroes)

    rng = np.random.default_rng(seed)
    train_df = build_samples(train_matches, builder, rng, stages_per_pick)
    train_df["split"] = "train"
    val_df = build_samples(val_matches, builder, rng, stages_per_pick)
    val_df["split"] = "val"

    dataset = pd.concat([train_df, val_df], ignore_index=True)
    return dataset, (synergy, counter, hero_stats)


def main():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    l3_cfg = cfg["model"].get("l3", {})

    hero_ids = load_hero_ids(raw_dir)
    heroes = load_heroes_frame(raw_dir)
    pm_path = raw_dir / cfg["data"].get("public_matches_file", "public_matches.jsonl")
    if not pm_path.exists():
        raise SystemExit(
            f"Нет {pm_path}. Сначала: python -m src.data_pipeline.fetch_public_matches"
        )

    matches = load_public_matches(pm_path, hero_ids)
    max_matches = l3_cfg.get("max_matches")
    if max_matches and len(matches) > max_matches:
        # Берём самые свежие: мета Dota меняется от патча к патчу.
        matches = matches.slice(slice(len(matches) - int(max_matches), None))

    dataset, (synergy, counter, hero_stats) = build_dataset(
        matches=matches,
        heroes=heroes,
        val_fraction=l3_cfg.get("val_fraction", 0.2),
        prior_weight=cfg["model"].get("prior_weight", cfg["model"]["synergy_min_games"]),
        seed=l3_cfg.get("random_seed", 42),
        stages_per_pick=l3_cfg.get("stages_per_pick", 1),
    )

    out_path = processed_dir / "l3_dataset.parquet"
    dataset.to_parquet(out_path, index=False)

    # Артефакты, посчитанные только по train: на них же должны работать
    # L0/L1/L2 в ablation, иначе сравнение уровней будет нечестным.
    synergy.to_parquet(processed_dir / "train_synergy_matrix.parquet")
    counter.to_parquet(processed_dir / "train_counter_matrix.parquet")
    hero_stats.to_parquet(processed_dir / "train_hero_stats.parquet")

    meta = {
        "n_matches": int(len(matches)),
        "match_id_min": int(matches.match_id.min()),
        "match_id_max": int(matches.match_id.max()),
        "n_samples": int(len(dataset)),
        "val_fraction": l3_cfg.get("val_fraction", 0.2),
        "prior_weight": cfg["model"].get("prior_weight", cfg["model"]["synergy_min_games"]),
        "seed": l3_cfg.get("random_seed", 42),
    }
    (processed_dir / "l3_dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts = dataset["split"].value_counts()
    print(f"Датасет L3 -> {out_path}")
    print(f"  матчей: {len(matches)}, примеров: {len(dataset)}")
    print(f"  train: {counts.get('train', 0)}, val: {counts.get('val', 0)}")
    print(f"  доля побед (train): {dataset.loc[dataset['split'] == 'train', 'label'].mean():.3f}")


if __name__ == "__main__":
    main()
