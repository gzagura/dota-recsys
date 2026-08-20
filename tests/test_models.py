import pandas as pd
import pytest

from src.models.baseline import BaselineRecommender
from src.models.synergy_model import SynergyRecommender


@pytest.fixture
def hero_stats():
    return pd.DataFrame(
        {
            "hero_id": [1, 2, 3, 4],
            "localized_name": ["Anti-Mage", "Axe", "Bane", "Crystal Maiden"],
            "winrate": [0.55, 0.50, 0.48, 0.52],
        }
    )


def test_baseline_excludes_taken_heroes(hero_stats):
    rec = BaselineRecommender(hero_stats=hero_stats)
    result = rec.recommend(exclude_ids={1}, top_k=3)
    assert 1 not in result["hero_id"].values
    assert len(result) == 3


def test_baseline_sorts_by_winrate_desc(hero_stats):
    rec = BaselineRecommender(hero_stats=hero_stats)
    result = rec.recommend(exclude_ids=set(), top_k=4)
    assert list(result["winrate"]) == sorted(result["winrate"], reverse=True)


@pytest.fixture
def matrices():
    ids = [1, 2, 3, 4]
    synergy = pd.DataFrame(0.0, index=ids, columns=ids)
    counter = pd.DataFrame(0.0, index=ids, columns=ids)
    # герой 2 хорошо синергирует с героем 1
    synergy.loc[2, 1] = 0.1
    return synergy, counter


def test_synergy_boosts_ally_synergized_hero(hero_stats, matrices):
    synergy, counter = matrices
    rec = SynergyRecommender(
        hero_stats=hero_stats,
        synergy_matrix=synergy,
        counter_matrix=counter,
    )
    result = rec.recommend(allies=[1], enemies=[], exclude_ids={1}, top_k=1)
    assert result.iloc[0]["hero_id"] == 2


@pytest.fixture
def player_profile():
    """Герой 3 — основной и успешный, герой 4 — одна игра, одна победа."""
    return pd.DataFrame(
        {
            "hero_id": [3, 4],
            "games": [100, 1],
            "wins": [70, 1],
            "winrate": [0.7, 1.0],
        }
    )


def test_player_score_is_smoothed_not_raw(hero_stats, matrices, player_profile):
    """Одна победа на одной игре не должна перебивать 70% на сотне игр."""
    synergy, counter = matrices
    rec = SynergyRecommender(
        hero_stats=hero_stats,
        synergy_matrix=synergy,
        counter_matrix=counter,
        player_profile=player_profile,
        w_player=1.0,
    )
    assert rec._player_score(3) > rec._player_score(4)
    # 1 игра при приоре 10 -> вклад заметно меньше половины отклонения
    assert rec._player_score(4) < 0.05


def test_unplayed_hero_is_neutral_not_worst(hero_stats, matrices, player_profile):
    """Несыгранный герой = 0 (нейтрально), а не хуже проигрышного."""
    losing = pd.DataFrame({"hero_id": [3], "games": [100], "wins": [20], "winrate": [0.2]})
    synergy, counter = matrices
    rec = SynergyRecommender(
        hero_stats=hero_stats,
        synergy_matrix=synergy,
        counter_matrix=counter,
        player_profile=losing,
        w_player=1.0,
    )
    assert rec._player_score(1) == 0.0        # герой без игр
    assert rec._player_score(3) < 0.0         # герой со стабильным минусом


def test_tactics_silent_on_empty_draft():
    """На пустом драфте подсказок быть не должно.

    Правила «саппортов мало» (<=1) и «нет инициации» (==0) формально верны
    для пустого состава, но говорят не о драфте, а о том, что герои ещё не
    введены.
    """
    from src.service import tactics

    assert tactics({}, {}, 0, 0) == []
    assert tactics({}, {}, 1, 1) == []


def test_tactics_fire_on_assembled_draft():
    from src.service import tactics

    my = {"Carry": 3, "Support": 1}
    enemy = {"Disabler": 3, "Nuker": 3}
    hints = tactics(my, enemy, 5, 5)
    assert hints and any("контроля" in h for h in hints)


def test_hero_aliases_have_no_duplicates():
    """Один алиас на двух героев — ошибка словаря, а не мелочь.

    Молча выбранный «первый попавшийся» герой давал бы неверный пик без
    всякого сигнала, поэтому build_alias_lookup обязан падать.
    """
    import json

    from src.hero_aliases import ALIASES, build_alias_lookup

    heroes = json.load(open("data/raw/heroes.json", encoding="utf-8"))
    name_to_id = {h["localized_name"]: h["id"] for h in heroes}
    lookup = build_alias_lookup(name_to_id)

    assert lookup["квопа"] == name_to_id["Queen of Pain"]
    assert lookup["кристалка"] == name_to_id["Crystal Maiden"]
    assert lookup["сф"] == name_to_id["Shadow Fiend"]
    # Словарь покрывает весь справочник: новый герой без алиасов —
    # это забытая строчка, а не осознанное решение.
    missing = set(name_to_id) - set(ALIASES)
    assert not missing, f"нет алиасов для: {sorted(missing)}"


def test_build_alias_lookup_rejects_conflict():
    from src.hero_aliases import build_alias_lookup

    import src.hero_aliases as mod

    original = mod.ALIASES
    mod.ALIASES = {"Axe": ["акс"], "Bane": ["акс"]}
    try:
        with pytest.raises(ValueError, match="двум героям"):
            build_alias_lookup({"Axe": 2, "Bane": 3})
    finally:
        mod.ALIASES = original


def test_engagement_hints_do_not_contradict():
    """«Навязывайте драку первыми» и «не входите в драку» — про одно решение.

    Обе строки в одном списке читались бы как шум, поэтому при своей
    инициативе советы «не входите» подавляются.
    """
    from src.service import tactics

    mine = {"Initiator": 3, "Disabler": 2, "Support": 2}
    enemy = {"Initiator": 0, "Disabler": 4}
    hints = tactics(mine, enemy, 5, 5)

    assert any("навязывайте драку первыми" in h for h in hints)
    assert not any("не входите" in h for h in hints)


def test_second_number_hint_when_enemy_initiates():
    from src.service import tactics

    hints = tactics({"Initiator": 0, "Support": 2}, {"Initiator": 3}, 5, 5)
    assert any("вторым номером" in h for h in hints)
