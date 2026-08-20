"""
L0 baseline: рекомендация по общему winrate героя, без учёта контекста
драфта и профиля игрока. Точка отсчёта для сравнения с L1/L2/L3.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class BaselineRecommender:
    hero_stats: pd.DataFrame  # ожидает колонки: hero_id, name, pro_win, pro_pick (или аналог)

    def recommend(self, exclude_ids: set[int], top_k: int = 3) -> pd.DataFrame:
        candidates = self.hero_stats[~self.hero_stats["hero_id"].isin(exclude_ids)]
        return candidates.sort_values("winrate", ascending=False).head(top_k)
