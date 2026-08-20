"""
L3: агрегированные фичи драфта — ОДИН источник правды для обучения и инференса.

Ключевая идея уровня L3: не подавать в модель разреженный one-hot драфта
(2 x 124 бинарных признака на пример — переобучение на объёмах, доступных
через OpenDota), а свернуть контекст в ~40 агрегатов: статистики синергии
кандидата с уже выбранными союзниками, статистики контрпика против врагов,
ролевой состав своей команды и стадия драфта.

Один и тот же билдер используется:
  - в build_training_dataset.py  (батч из миллионов строк, обучение);
  - в models/l3_lgbm.py          (124 кандидата в одном контексте, инференс).

Так исключается train/serve skew: расхождение формул на обучении и в бою —
самая частая причина того, что офлайн-метрики не воспроизводятся в проде.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

# Роли из справочника OpenDota (/heroes -> поле roles).
ROLES = [
    "Carry",
    "Support",
    "Nuker",
    "Disabler",
    "Durable",
    "Escape",
    "Initiator",
    "Pusher",
    "Jungler",
]

PRIMARY_ATTRS = ["str", "agi", "int", "all"]

# Признаки, которые модель видит на входе. Порядок фиксирован: он же
# сохраняется в метаданных модели и проверяется при инференсе.
FEATURE_COLUMNS = (
    [
        "hero_id",
        "is_radiant",
        "n_allies",
        "n_enemies",
        "base_wr",
        "pick_rate",
        "syn_sum",
        "syn_mean",
        "syn_max",
        "syn_min",
        "ctr_sum",
        "ctr_mean",
        "ctr_max",
        "ctr_min",
        "ally_base_wr_mean",
        "enemy_base_wr_mean",
        "primary_attr_code",
    ]
    + [f"role_{r.lower()}" for r in ROLES]
    + [f"ally_role_{r.lower()}" for r in ROLES]
)

# Признаки, которые LightGBM должен трактовать как категориальные.
CATEGORICAL_FEATURES = ["hero_id", "primary_attr_code"]


def _agg_masked(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """sum/mean/max/min по замаскированной части строки; пустая строка -> 0."""
    n = values.shape[0]
    if values.shape[1] == 0:
        z = np.zeros(n, dtype=np.float32)
        return z, z.copy(), z.copy(), z.copy()

    count = mask.sum(axis=1)
    nonempty = count > 0
    total = np.where(mask, values, 0.0).sum(axis=1)
    mean = np.where(nonempty, total / np.maximum(count, 1), 0.0)
    mx = np.where(nonempty, np.where(mask, values, -np.inf).max(axis=1), 0.0)
    mn = np.where(nonempty, np.where(mask, values, np.inf).min(axis=1), 0.0)
    return (
        total.astype(np.float32),
        mean.astype(np.float32),
        mx.astype(np.float32),
        mn.astype(np.float32),
    )


@dataclass
class DraftFeatureBuilder:
    """Матрицы синергии/контрпиков + справочник героев -> матрица фич.

    Все матрицы хранятся в виде numpy-массивов, проиндексированных ВНУТРЕННИМ
    индексом героя (0..N-1), а не hero_id: hero_id в Dota разрежены (есть дыры),
    и работа по позиционному индексу заметно быстрее на батчах в сотни тысяч строк.
    """

    hero_ids: np.ndarray  # внутренний индекс -> hero_id
    synergy: np.ndarray  # (N, N), дельта winrate в паре
    counter: np.ndarray  # (N, N), дельта winrate против
    base_wr: np.ndarray  # (N,)
    pick_rate: np.ndarray  # (N,)
    attr_code: np.ndarray  # (N,)
    role_flags: np.ndarray  # (N, len(ROLES))

    @property
    def id_to_index(self) -> dict[int, int]:
        if not hasattr(self, "_id_to_index"):
            self._id_to_index = {int(h): i for i, h in enumerate(self.hero_ids)}
        return self._id_to_index

    @classmethod
    def from_frames(
        cls,
        hero_stats: pd.DataFrame,
        synergy: pd.DataFrame,
        counter: pd.DataFrame,
        heroes: pd.DataFrame,
    ) -> "DraftFeatureBuilder":
        """Собирает билдер из артефактов пайплайна.

        hero_stats: hero_id, winrate, pick_rate (из build_synergy_matrix)
        synergy/counter: квадратные DataFrame с hero_id в index/columns
        heroes: справочник /heroes (hero_id, roles, primary_attr)
        """
        hero_ids = np.asarray(synergy.index, dtype=np.int64)

        stats = hero_stats.set_index("hero_id").reindex(hero_ids)
        base_wr = stats["winrate"].fillna(0.5).to_numpy(dtype=np.float32)
        pick_col = "pick_rate" if "pick_rate" in stats.columns else None
        pick_rate = (
            stats[pick_col].fillna(0.0).to_numpy(dtype=np.float32)
            if pick_col
            else np.zeros(len(hero_ids), dtype=np.float32)
        )

        meta = heroes.set_index("hero_id").reindex(hero_ids)
        attr_code = np.array(
            [
                PRIMARY_ATTRS.index(a) if a in PRIMARY_ATTRS else len(PRIMARY_ATTRS)
                for a in meta.get("primary_attr", pd.Series(index=meta.index, dtype=object))
            ],
            dtype=np.int64,
        )

        role_flags = np.zeros((len(hero_ids), len(ROLES)), dtype=np.float32)
        if "roles" in meta.columns:
            for i, roles in enumerate(meta["roles"]):
                if isinstance(roles, (list, tuple, np.ndarray)):
                    for r in roles:
                        if r in ROLES:
                            role_flags[i, ROLES.index(r)] = 1.0

        return cls(
            hero_ids=hero_ids,
            synergy=np.ascontiguousarray(synergy.to_numpy(dtype=np.float32)),
            counter=np.ascontiguousarray(counter.to_numpy(dtype=np.float32)),
            base_wr=base_wr,
            pick_rate=pick_rate,
            attr_code=attr_code,
            role_flags=role_flags,
        )

    def to_index(self, hero_ids: Sequence[int]) -> np.ndarray:
        """hero_id -> внутренний индекс.

        Герои, которых нет в матрицах (например, добавленные патчем уже
        после сбора данных), молча отбрасываются: лучше выдать рекомендацию
        по неполному контексту, чем уронить бота посреди драфта.
        """
        mapping = self.id_to_index
        return np.array([mapping[int(h)] for h in hero_ids if int(h) in mapping], dtype=np.int64)

    def build_batch(
        self,
        candidates: np.ndarray,
        allies: np.ndarray,
        enemies: np.ndarray,
        is_radiant: np.ndarray,
    ) -> pd.DataFrame:
        """Векторизованная сборка фич.

        candidates: (n,) внутренние индексы кандидатов
        allies:     (n, k_a) внутренние индексы, -1 = пусто
        enemies:    (n, k_e) внутренние индексы, -1 = пусто
        is_radiant: (n,) 0/1
        """
        candidates = np.asarray(candidates, dtype=np.int64)
        n = candidates.shape[0]
        ally_mask = allies >= 0
        enemy_mask = enemies >= 0
        ally_safe = np.where(ally_mask, allies, 0)
        enemy_safe = np.where(enemy_mask, enemies, 0)

        syn_vals = self.synergy[candidates[:, None], ally_safe]
        ctr_vals = self.counter[candidates[:, None], enemy_safe]
        syn_sum, syn_mean, syn_max, syn_min = _agg_masked(syn_vals, ally_mask)
        ctr_sum, ctr_mean, ctr_max, ctr_min = _agg_masked(ctr_vals, enemy_mask)

        _, ally_wr_mean, _, _ = _agg_masked(self.base_wr[ally_safe], ally_mask)
        _, enemy_wr_mean, _, _ = _agg_masked(self.base_wr[enemy_safe], enemy_mask)

        # Ролевой состав уже собранной части своей команды: по одному
        # сложению на слот (максимум 4) вместо материализации (n, k, |ROLES|).
        ally_roles = np.zeros((n, len(ROLES)), dtype=np.float32)
        for slot in range(allies.shape[1]):
            ally_roles += self.role_flags[ally_safe[:, slot]] * ally_mask[:, slot, None]

        data = {
            "hero_id": self.hero_ids[candidates],
            "is_radiant": np.asarray(is_radiant, dtype=np.int8),
            "n_allies": ally_mask.sum(axis=1).astype(np.int8),
            "n_enemies": enemy_mask.sum(axis=1).astype(np.int8),
            "base_wr": self.base_wr[candidates],
            "pick_rate": self.pick_rate[candidates],
            "syn_sum": syn_sum,
            "syn_mean": syn_mean,
            "syn_max": syn_max,
            "syn_min": syn_min,
            "ctr_sum": ctr_sum,
            "ctr_mean": ctr_mean,
            "ctr_max": ctr_max,
            "ctr_min": ctr_min,
            "ally_base_wr_mean": ally_wr_mean,
            "enemy_base_wr_mean": enemy_wr_mean,
            "primary_attr_code": self.attr_code[candidates],
        }
        for j, role in enumerate(ROLES):
            data[f"role_{role.lower()}"] = self.role_flags[candidates, j]
        for j, role in enumerate(ROLES):
            data[f"ally_role_{role.lower()}"] = ally_roles[:, j]

        return pd.DataFrame(data, columns=FEATURE_COLUMNS)

    def build_context(
        self,
        candidate_ids: Sequence[int],
        allies: Sequence[int],
        enemies: Sequence[int],
        is_radiant: bool = True,
    ) -> pd.DataFrame:
        """Инференс: один контекст драфта, много кандидатов.

        Возвращает фичи в том же виде, что и build_batch — то есть модель
        в бою видит ровно то, на чём училась.
        """
        cand_idx = self.to_index(candidate_ids)
        ally_idx = self.to_index(allies)
        enemy_idx = self.to_index(enemies)
        n = len(cand_idx)

        ally_mat = np.tile(ally_idx, (n, 1)) if len(ally_idx) else np.empty((n, 0), dtype=np.int64)
        enemy_mat = (
            np.tile(enemy_idx, (n, 1)) if len(enemy_idx) else np.empty((n, 0), dtype=np.int64)
        )
        return self.build_batch(
            candidates=cand_idx,
            allies=ally_mat,
            enemies=enemy_mat,
            is_radiant=np.full(n, int(is_radiant), dtype=np.int8),
        )
