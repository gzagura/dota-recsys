"""
L3: LightGBM на агрегированных фичах драфта.

Постановка задачи
-----------------
Модель решает задачу бинарной классификации: по кандидату и агрегированному
контексту драфта предсказать вероятность победы его команды.
Рекомендация получается ранжированием: все свободные герои прогоняются через
модель в текущем контексте, наверх идут те, у кого выше P(win).

Почему именно так, а не «регрессия на score»:
- метка (победа/поражение) есть в данных как есть, без разметки экспертом;
- вероятность победы интерпретируема и сравнима между контекстами;
- градиентный бустинг на ~40 плотных агрегатах учится на десятках тысяч
  матчей, тогда как one-hot драфта (2 x 124 признака) на этих объёмах
  переобучается.

Чем L3 отличается от L1/L2
--------------------------
L1/L2 складывают синергию и контрпик с ФИКСИРОВАННЫМИ весами (w_synergy,
w_counter). L3 обучает нелинейную функцию от тех же величин плюс стадия
драфта, ролевой состав команды и атрибут героя — то есть сам находит,
когда синергия важнее контрпика, и как это зависит от момента драфта.

Персонализация (L2) в L3 не входит: /publicMatches не содержит account_id,
поэтому по этим данным профиль игрока обучить нельзя. Она навешивается
поверх предсказания как аддитивная поправка (w_player), см. L3Recommender.

Запуск обучения:
    python -m src.models.l3_lgbm
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.features.draft_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    DraftFeatureBuilder,
)

CONFIG_PATH = Path("configs/config.yaml")

# Используется низкоуровневый API lgb.train, а не обёртка LGBMClassifier:
# сигнатура fit() у обёртки менялась внутри ветки 4.x (eval_set -> eval_X/eval_y),
# и код, написанный под одну версию, ломается или сыплет warning'ами на другой.
# Ранняя остановка идёт по binary_logloss, а не по AUC. AUC зависит только
# от порядка предсказаний и не замечает, что вероятности разъезжаются: на
# первом прогоне модель останавливалась по AUC с val logloss 0.751, то есть
# хуже константного прогноза 0.5 (ln2 = 0.693). Для бота важен именно
# масштаб P(win), поэтому logloss идёт первым, а AUC остаётся для отчёта.
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_iterations": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    # hero_id — категориальный признак на ~124 уровня: LightGBM легко
    # выучивает по нему шум отдельного героя. Сглаживание категорий и
    # минимальный размер группы держат эти сплиты в узде.
    "cat_smooth": 50.0,
    "min_data_per_group": 200,
    "seed": 42,
    "num_threads": 0,
    "verbosity": -1,
}

# Имена гиперпараметров в config.yaml -> имена параметров LightGBM.
CONFIG_PARAM_ALIASES = {
    "n_estimators": "num_iterations",
    "learning_rate": "learning_rate",
    "num_leaves": "num_leaves",
    "min_child_samples": "min_data_in_leaf",
    "random_state": "seed",
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_l3(
    dataset: pd.DataFrame,
    params: dict | None = None,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, dict]:
    """Обучает модель на split=='train', валидирует на split=='val'."""
    # sklearn нужен только для метрик обучения. Импорт держим здесь, чтобы
    # инференс на сервере не тянул его в память: сайту он не нужен.
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    train = dataset[dataset["split"] == "train"]
    val = dataset[dataset["split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("В датасете должны быть обе части: split=='train' и split=='val'")

    x_train, y_train = train[FEATURE_COLUMNS], train["label"]
    x_val, y_val = val[FEATURE_COLUMNS], val["label"]

    model_params = {**DEFAULT_PARAMS, **(params or {})}
    num_rounds = int(model_params.pop("num_iterations"))

    train_set = lgb.Dataset(
        x_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=False)

    booster = lgb.train(
        model_params,
        train_set,
        num_boost_round=num_rounds,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            # first_metric_only: останавливаемся по binary_logloss (первая
            # метрика в DEFAULT_PARAMS), иначе LightGBM ждёт, пока перестанут
            # улучшаться ВСЕ метрики, и AUC тянет обучение в переобучение.
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    pred_val = booster.predict(x_val)
    pred_train = booster.predict(x_train)
    metrics = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "best_iteration": int(booster.best_iteration or num_rounds),
        "train_auc": float(roc_auc_score(y_train, pred_train)),
        "val_auc": float(roc_auc_score(y_val, pred_val)),
        "val_logloss": float(log_loss(y_val, pred_val)),
        "val_accuracy": float(accuracy_score(y_val, (pred_val >= 0.5).astype(int))),
        # Точка отсчёта: предсказывать победу по одному базовому winrate
        # кандидата. Если L3 не бьёт её по AUC — контекст драфта не выучен.
        "val_auc_base_wr_only": float(roc_auc_score(y_val, x_val["base_wr"])),
        # Вторая точка отсчёта, уже для калибровки: logloss константы 0.5.
        # Выборка сбалансирована по построению, поэтому это ln2 = 0.6931.
        # Модель, проигравшая константе, выдаёт вредные вероятности, даже
        # если её AUC выше 0.5.
        "val_logloss_constant": float(log_loss(y_val, np.full(len(y_val), 0.5))),
    }
    return booster, metrics


def feature_importance(booster: lgb.Booster, top_n: int = 15) -> pd.DataFrame:
    imp = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }
    )
    imp["gain_share"] = imp["gain"] / max(imp["gain"].sum(), 1e-9)
    return imp.sort_values("gain", ascending=False).head(top_n).reset_index(drop=True)


def save_model(
    booster: lgb.Booster,
    processed_dir: Path,
    model_file: str,
    metrics: dict,
    params: dict | None = None,
    extra_meta: dict | None = None,
) -> Path:
    model_path = processed_dir / model_file
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "params": params or {},
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra_meta or {}),
    }
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return model_path


def player_delta(profile: pd.DataFrame, prior_games: float = 10.0) -> dict[int, float]:
    """Отклонение личного winrate игрока от 50% по каждому герою.

    Сырой winrate по трём играм — шум, поэтому он сжимается к 0.5 приором в
    `prior_games` игр: чем меньше сыграно, тем ближе поправка к нулю и тем
    меньше личная статистика двигает выдачу модели.
    """
    shrunk = (profile["wins"] + prior_games * 0.5) / (profile["games"] + prior_games)
    return dict(zip(profile["hero_id"].astype(int), shrunk - 0.5))


@dataclass
class L3Recommender:
    """Инференс L3: ранжирование свободных героев по P(win) в текущем драфте.

    Интерфейс recommend() совпадает с BaselineRecommender/SynergyRecommender,
    поэтому уровни взаимозаменяемы в боте и в ablation-скрипте.
    """

    booster: lgb.Booster
    feature_builder: DraftFeatureBuilder
    hero_names: pd.DataFrame  # hero_id, localized_name

    # Персонализация поверх модели (L2-надстройка над L3):
    # score = P(win) + w_player * (сглаженный winrate игрока - 0.5)
    player_profile: pd.DataFrame | None = None
    w_player: float = 0.0
    player_prior_games: float = 10.0

    _player_delta: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.player_profile is not None and self.w_player:
            self._player_delta = player_delta(self.player_profile, self.player_prior_games)

    @classmethod
    def from_artifacts(
        cls,
        processed_dir: Path,
        heroes: pd.DataFrame,
        model_file: str = "l3_lgbm.txt",
        synergy_file: str = "synergy_matrix.parquet",
        counter_file: str = "counter_matrix.parquet",
        hero_stats_file: str = "hero_stats.parquet",
        **kwargs,
    ) -> "L3Recommender":
        """Собирает рекомендатель из артефактов пайплайна."""
        booster = lgb.Booster(model_file=str(processed_dir / model_file))
        builder = DraftFeatureBuilder.from_frames(
            hero_stats=pd.read_parquet(processed_dir / hero_stats_file),
            synergy=pd.read_parquet(processed_dir / synergy_file),
            counter=pd.read_parquet(processed_dir / counter_file),
            heroes=heroes,
        )

        # Имена моделью не используются — они нужны только для вывода в бот.
        names = heroes[["hero_id", "localized_name"]].copy()

        expected = booster.feature_name()
        if expected and list(expected) != list(FEATURE_COLUMNS):
            raise ValueError(
                "Набор фич модели не совпадает с текущим FEATURE_COLUMNS — "
                "модель обучена на другой версии кода, переобучи L3"
            )
        return cls(booster=booster, feature_builder=builder, hero_names=names, **kwargs)

    def score_components(
        self,
        candidate_ids: list[int],
        allies: list[int],
        enemies: list[int],
        is_radiant: bool = True,
        delta: dict[int, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(P(win) модели, итоговый скор с персонализацией).

        Величины возвращаются раздельно, потому что смешивать их в одном
        числе нельзя: P(win) — вероятность, а скор после прибавления
        w_player * (winrate игрока - 0.5) вероятностью быть перестаёт.
        Показывать его как «шанс победы» было бы враньём.

        `delta` позволяет подставить профиль конкретного пользователя на
        время одного запроса: на сайте у каждого свой steam-аккаунт, а
        модель и матрицы общие, поэтому персонализация не может жить в
        состоянии рекомендателя. None — взять профиль из конфига (если он
        вообще задан), пустой словарь — считать без персонализации.
        """
        features = self.feature_builder.build_context(
            candidate_ids=candidate_ids, allies=allies, enemies=enemies, is_radiant=is_radiant
        )
        p_win = np.asarray(self.booster.predict(features), dtype=float)
        adjust = self._player_delta if delta is None else delta
        scores = p_win
        if adjust:
            adj = np.array([adjust.get(int(h), 0.0) for h in candidate_ids])
            scores = p_win + self.w_player * adj
        return p_win, np.asarray(scores, dtype=float)

    def score_candidates(
        self,
        candidate_ids: list[int],
        allies: list[int],
        enemies: list[int],
        is_radiant: bool = True,
        delta: dict[int, float] | None = None,
    ) -> np.ndarray:
        _, scores = self.score_components(candidate_ids, allies, enemies, is_radiant, delta)
        return scores

    def recommend(
        self,
        allies: list[int],
        enemies: list[int],
        exclude_ids: set[int],
        top_k: int = 3,
        is_radiant: bool = True,
        delta: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        known = self.feature_builder.hero_ids
        candidates = [int(h) for h in known if int(h) not in exclude_ids]
        if not candidates:
            return pd.DataFrame(
                columns=["hero_id", "localized_name", "score", "p_win", "player_bonus"]
            )

        p_win, scores = self.score_components(candidates, allies, enemies, is_radiant, delta)
        result = pd.DataFrame(
            {
                "hero_id": candidates,
                "score": scores,
                "p_win": p_win,
                "player_bonus": scores - p_win,
            }
        )
        result = result.merge(self.hero_names, on="hero_id", how="left")
        return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def main():
    cfg = load_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    l3_cfg = cfg["model"].get("l3", {})

    dataset_path = processed_dir / "l3_dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit(
            f"Нет {dataset_path}. Сначала: python -m src.features.build_training_dataset"
        )
    dataset = pd.read_parquet(dataset_path)

    params = {
        lgb_name: l3_cfg[cfg_name]
        for cfg_name, lgb_name in CONFIG_PARAM_ALIASES.items()
        if cfg_name in l3_cfg
    }
    booster, metrics = train_l3(
        dataset, params=params, early_stopping_rounds=l3_cfg.get("early_stopping_rounds", 50)
    )

    dataset_meta_path = processed_dir / "l3_dataset_meta.json"
    dataset_meta = (
        json.loads(dataset_meta_path.read_text(encoding="utf-8"))
        if dataset_meta_path.exists()
        else {}
    )
    model_path = save_model(
        booster,
        processed_dir,
        l3_cfg.get("model_file", "l3_lgbm.txt"),
        metrics,
        params={**DEFAULT_PARAMS, **params},
        extra_meta={"dataset": dataset_meta},
    )

    print(f"Модель L3 сохранена -> {model_path}")
    print("\nМетрики (валидация — матчи, более поздние по времени):")
    print(f"  AUC   train / val : {metrics['train_auc']:.4f} / {metrics['val_auc']:.4f}")
    print(f"  logloss val       : {metrics['val_logloss']:.4f}")
    print(f"  logloss константы : {metrics['val_logloss_constant']:.4f}  <- точка отсчёта")
    print(f"  accuracy val      : {metrics['val_accuracy']:.4f}")
    print(f"  AUC только base_wr: {metrics['val_auc_base_wr_only']:.4f}  <- точка отсчёта")
    print(f"  лучшая итерация   : {metrics['best_iteration']}")
    print("\nТоп признаков по gain:")
    print(feature_importance(booster).to_string(index=False))


if __name__ == "__main__":
    main()
