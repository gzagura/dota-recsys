"""
L1/L2: рекомендация с учётом контекста текущего драфта.

score(candidate) =
      w_base   * base_winrate(candidate)
    + w_syn    * mean(synergy[candidate][ally] for ally in current_allies)
    + w_ctr    * mean(counter[candidate][enemy] for enemy in current_enemies)
    + w_player * (сглаженный winrate игрока по герою - 0.5)   # только L2

Веса (w_*) — гиперпараметры, которые стоит подбирать через ablation
на исторических данных (см. метрики Hit Rate@K в README).
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SynergyRecommender:
    hero_stats: pd.DataFrame
    synergy_matrix: pd.DataFrame
    counter_matrix: pd.DataFrame
    player_profile: pd.DataFrame | None = None

    w_base: float = 1.0
    w_synergy: float = 1.0
    w_counter: float = 1.0
    w_player: float = 0.5
    # Приор персонализации в «играх»: герой, сыгранный prior_games раз,
    # получает половину своего отклонения от 0.5, остальное съедает
    # сжатие. Без него 1 игра / 1 победа давала бы максимальный вклад.
    player_prior_games: float = 10.0

    _player_delta: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.player_profile is not None:
            prof = self.player_profile
            shrunk = (prof["wins"] + self.player_prior_games * 0.5) / (
                prof["games"] + self.player_prior_games
            )
            self._player_delta = dict(zip(prof["hero_id"].astype(int), shrunk - 0.5))

    def _base_score(self, hero_id: int) -> float:
        row = self.hero_stats.loc[self.hero_stats["hero_id"] == hero_id]
        return float(row["winrate"].iloc[0]) if not row.empty else 0.5

    def _synergy_score(self, hero_id: int, allies: list[int]) -> float:
        if not allies:
            return 0.0
        return float(self.synergy_matrix.loc[hero_id, allies].mean())

    def _counter_score(self, hero_id: int, enemies: list[int]) -> float:
        if not enemies:
            return 0.0
        return float(self.counter_matrix.loc[hero_id, enemies].mean())

    def _player_score(self, hero_id: int) -> float:
        """Отклонение сглаженного winrate игрока от 0.5, а не сырой winrate.

        Ноль здесь означает «нейтрально»: столько получает и несыгранный
        герой, и герой, на котором игрок держит ровно 50%. Раньше возвращался
        сырой winrate, из-за чего несыгранный герой (0.0) оказывался хуже
        любого сыгранного, а одна победа на одной игре (1.0) перебивала
        весь остальной скор.
        """
        return self._player_delta.get(int(hero_id), 0.0)

    def recommend(
        self,
        allies: list[int],
        enemies: list[int],
        exclude_ids: set[int],
        top_k: int = 3,
    ) -> pd.DataFrame:
        candidates = self.hero_stats[~self.hero_stats["hero_id"].isin(exclude_ids)].copy()

        candidates["score"] = candidates["hero_id"].apply(
            lambda hid: (
                self.w_base * self._base_score(hid)
                + self.w_synergy * self._synergy_score(hid, allies)
                + self.w_counter * self._counter_score(hid, enemies)
                + self.w_player * self._player_score(hid)
            )
        )
        return candidates.sort_values("score", ascending=False).head(top_k)
