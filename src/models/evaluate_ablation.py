"""
Ablation: L0 vs L1 vs L3 на одних и тех же ситуациях драфта.

Метрика ранжирования
--------------------
Берём валидационные примеры (матчи, более поздние по времени), где кандидат
принадлежит ПОБЕДИВШЕЙ команде. Считаем такой пик «правильным ответом»,
ранжируем всех свободных героев в этом контексте и смотрим, на каком месте
оказался реально выбранный герой.

  Hit Rate@K — доля ситуаций, где реальный пик попал в топ-K
  MRR        — средний обратный ранг реального пика

Что метрика НЕ измеряет (важно для раздела «Выводы»): реальный пик — не
эталон оптимальности, а лишь пик, который в итоге привёл к победе. Поэтому
метрика вознаграждает в том числе популярность героя.

Две контрольные строки нужны, чтобы это было видно в цифрах:
  random     — уровень случайного угадывания (аналитический, k / кандидаты);
  popularity — ранжирование по одному pick_rate, без всякой модели.
Если popularity бьёт L0/L1/L3 — метрика измеряет в основном популярность,
а не качество рекомендации, и выводы надо строить по AUC. Ранжирование по
winrate анти-коррелирует с популярностью (высокий winrate обычно у нишевых
героев), поэтому уровни могут оказаться даже ниже random — это свойство
метрики, а не поломка модели.

L2 здесь не оценивается: /publicMatches не содержит account_id, профиля
игрока для чужих матчей не существует. Персонализацию корректно мерить
только на истории самого автора (см. data/manual_eval_log.csv в README).

Запуск:
    python -m src.models.evaluate_ablation
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.features.build_synergy_matrix import load_heroes_frame
from src.features.build_training_dataset import ALLY_COLUMNS, ENEMY_COLUMNS
from src.models.baseline import BaselineRecommender
from src.models.l3_lgbm import L3Recommender
from src.models.synergy_model import SynergyRecommender

CONFIG_PATH = Path("configs/config.yaml")

K_VALUES = (1, 3, 5, 10)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_contexts(dataset: pd.DataFrame, n_contexts: int, seed: int) -> pd.DataFrame:
    """Валидационные пики победившей команды — «правильные ответы»."""
    val = dataset[(dataset["split"] == "val") & (dataset["label"] == 1)]
    if val.empty:
        raise ValueError("В валидации нет примеров с label==1")
    if len(val) > n_contexts:
        val = val.sample(n=n_contexts, random_state=seed)
    return val.reset_index(drop=True)


def context_lists(row: pd.Series) -> tuple[list[int], list[int]]:
    allies = [int(row[c]) for c in ALLY_COLUMNS if int(row[c]) >= 0]
    enemies = [int(row[c]) for c in ENEMY_COLUMNS if int(row[c]) >= 0]
    return allies, enemies


def rank_of(ranked_ids: list[int], true_hero: int) -> int | None:
    """Позиция реального пика в отранжированном списке (1-based)."""
    try:
        return ranked_ids.index(true_hero) + 1
    except ValueError:
        return None


def summarize(ranks: list[int | None], n_candidates: float) -> dict:
    valid = [r for r in ranks if r is not None]
    out: dict[str, float] = {"n": len(valid), "mean_candidates": n_candidates}
    if not valid:
        return out | {f"hit@{k}": float("nan") for k in K_VALUES} | {"mrr": float("nan")}
    arr = np.asarray(valid, dtype=float)
    out |= {f"hit@{k}": float((arr <= k).mean()) for k in K_VALUES}
    out["mrr"] = float((1.0 / arr).mean())
    return out


def evaluate_level(
    name: str,
    contexts: pd.DataFrame,
    rank_fn,
) -> dict:
    ranks: list[int | None] = []
    n_cand = []
    for _, row in tqdm(
        contexts.iterrows(), total=len(contexts), desc=f"Оцениваю {name}", leave=False
    ):
        allies, enemies = context_lists(row)
        true_hero = int(row["hero_id"])
        ranked = rank_fn(allies, enemies, set(allies) | set(enemies))
        n_cand.append(len(ranked))
        ranks.append(rank_of(ranked, true_hero))
    return {"level": name} | summarize(ranks, float(np.mean(n_cand)) if n_cand else 0.0)


def main():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    l3_cfg = cfg["model"].get("l3", {})
    eval_cfg = cfg["model"].get("eval", {})

    dataset_path = processed_dir / "l3_dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit(
            f"Нет {dataset_path}. Сначала: python -m src.features.build_training_dataset"
        )

    dataset = pd.read_parquet(dataset_path)
    contexts = sample_contexts(
        dataset,
        n_contexts=eval_cfg.get("n_contexts", 1000),
        seed=l3_cfg.get("random_seed", 42),
    )

    heroes = load_heroes_frame(raw_dir)
    # Все уровни используют статистику, посчитанную ТОЛЬКО по train-части —
    # иначе L0/L1 подглядывали бы в валидацию, а сравнение было бы нечестным.
    hero_stats = pd.read_parquet(processed_dir / "train_hero_stats.parquet")
    hero_stats = hero_stats.merge(heroes[["hero_id", "localized_name"]], on="hero_id", how="left")
    synergy = pd.read_parquet(processed_dir / "train_synergy_matrix.parquet")
    counter = pd.read_parquet(processed_dir / "train_counter_matrix.parquet")

    all_ids = list(hero_stats["hero_id"].astype(int))
    full_k = len(all_ids)

    l0 = BaselineRecommender(hero_stats=hero_stats)
    l1 = SynergyRecommender(
        hero_stats=hero_stats,
        synergy_matrix=synergy,
        counter_matrix=counter,
        w_base=cfg["model"].get("w_base", 1.0),
        w_synergy=cfg["model"].get("w_synergy", 1.0),
        w_counter=cfg["model"].get("w_counter", 1.0),
        w_player=0.0,
    )

    # Контрольная строка: ранжирование по одной популярности героя, без
    # модели. Показывает, какую часть Hit Rate даёт просто «берут часто».
    popularity_order = list(
        hero_stats.sort_values("pick_rate", ascending=False)["hero_id"].astype(int)
    )

    rows = [
        evaluate_level(
            "popularity (pick_rate)",
            contexts,
            lambda a, e, excl: [h for h in popularity_order if h not in excl],
        ),
        evaluate_level(
            "L0 (winrate)",
            contexts,
            lambda a, e, excl: list(
                l0.recommend(exclude_ids=excl, top_k=full_k)["hero_id"].astype(int)
            ),
        ),
        evaluate_level(
            "L1 (synergy+counter)",
            contexts,
            lambda a, e, excl: list(
                l1.recommend(allies=a, enemies=e, exclude_ids=excl, top_k=full_k)[
                    "hero_id"
                ].astype(int)
            ),
        ),
    ]

    model_path = processed_dir / l3_cfg.get("model_file", "l3_lgbm.txt")
    if model_path.exists():
        l3 = L3Recommender.from_artifacts(
            processed_dir=processed_dir,
            heroes=heroes,
            model_file=l3_cfg.get("model_file", "l3_lgbm.txt"),
            synergy_file="train_synergy_matrix.parquet",
            counter_file="train_counter_matrix.parquet",
            hero_stats_file="train_hero_stats.parquet",
        )
        rows.append(
            evaluate_level(
                "L3 (LightGBM)",
                contexts,
                lambda a, e, excl: list(
                    l3.recommend(allies=a, enemies=e, exclude_ids=excl, top_k=full_k)[
                        "hero_id"
                    ].astype(int)
                ),
            )
        )
    else:
        print(f"{model_path} не найден — L3 пропущен. Обучи: python -m src.models.l3_lgbm")

    # Аналитический ориентир: случайный выбор из свободных героев.
    mean_candidates = rows[0]["mean_candidates"]
    random_row = {"level": "random", "n": rows[0]["n"], "mean_candidates": mean_candidates}
    for k in K_VALUES:
        random_row[f"hit@{k}"] = k / mean_candidates
    random_row["mrr"] = float("nan")
    rows.append(random_row)

    report = pd.DataFrame(rows)[
        ["level", "n"] + [f"hit@{k}" for k in K_VALUES] + ["mrr", "mean_candidates"]
    ]
    print("\nAblation на валидации (реальный пик победившей команды как эталон):")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_path = processed_dir / "ablation_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nОтчёт сохранён -> {out_path}")


if __name__ == "__main__":
    main()
