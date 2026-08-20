import pytest

from src.data_pipeline.rank_tiers import (
    describe_window,
    mmr_to_rank_tier,
    rank_tier_name,
    sample_window,
)


@pytest.mark.parametrize(
    "mmr, expected",
    [
        (0, 11),      # Herald 1
        (800, 21),    # Guardian 1
        (2000, 33),   # Crusader 3
        (3080, 51),   # ровно нижняя граница Legend
        (3849, 55),   # верх Legend
        (4200, 63),   # Ancient 3
        (4620, 71),   # нижняя граница Divine
        (5619, 75),   # верх Divine -> звезда обрезается до 5
        (6000, 80),   # Immortal, звёзд нет
    ],
)
def test_mmr_maps_to_expected_tier(mmr, expected):
    tier = mmr_to_rank_tier(mmr)
    assert tier // 10 == expected // 10, f"{mmr} MMR попал не в ту медаль"
    assert 1 <= tier % 10 <= 5 or tier == 80


def test_star_never_exceeds_five():
    for mmr in range(0, 6000, 37):
        tier = mmr_to_rank_tier(mmr)
        assert tier == 80 or 1 <= tier % 10 <= 5


def test_tier_is_monotonic_in_mmr():
    tiers = [mmr_to_rank_tier(mmr) for mmr in range(0, 7000, 50)]
    assert tiers == sorted(tiers)


def test_negative_mmr_rejected():
    with pytest.raises(ValueError):
        mmr_to_rank_tier(-1)


def test_rank_tier_name():
    assert rank_tier_name(54) == "Legend 4"
    assert rank_tier_name(80) == "Immortal"


def test_sample_window_brackets_the_player():
    low, high = sample_window(4200, spread_medals=1)  # Ancient
    assert low == 50 and high == 80  # Legend .. Ancient+Divine
    tier = mmr_to_rank_tier(4200)
    assert low <= tier <= high


def test_zero_spread_is_own_medal_only():
    low, high = sample_window(3200, spread_medals=0)  # Legend
    assert (low, high) == (50, 60)


def test_window_is_clamped_at_the_edges():
    assert sample_window(0, spread_medals=2)[0] == 10
    assert sample_window(7000, spread_medals=2)[1] == 90


def test_describe_window_names_included_medals_only():
    # 4200 = Ancient, spread 1 -> Legend..Divine. Immortal (max_rank=80)
    # в выдачу API уже не попадает и упоминаться не должен.
    text = describe_window(4200, spread_medals=1)
    assert "Ancient" in text and "Legend" in text and "Divine" in text
    assert "Immortal" not in text
