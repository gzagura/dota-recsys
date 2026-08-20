"""
MMR -> бракет (rank_tier) OpenDota.

OpenDota кодирует медаль числом из двух цифр: первая — медаль (1 = Herald,
... 7 = Divine, 8 = Immortal), вторая — звезда (1..5). Например, 54 =
Legend 4. Эндпоинт /publicMatches принимает границы min_rank / max_rank
в этом же формате и фильтрует на своей стороне.

Границы MMR по медалям Valve официально не публикует. Здесь используется
общепринятая оценка: медали идут шагом 770 MMR, внутри медали 5 звёзд по
154 MMR. Для выбора популяции обучения этой точности достаточно, но в
разделе методологии стоит оговорить, что граница приблизительная.
"""
from __future__ import annotations

# (номер медали, название, нижняя граница MMR)
MEDALS: list[tuple[int, str, int]] = [
    (1, "Herald", 0),
    (2, "Guardian", 770),
    (3, "Crusader", 1540),
    (4, "Archon", 2310),
    (5, "Legend", 3080),
    (6, "Ancient", 3850),
    (7, "Divine", 4620),
    (8, "Immortal", 5620),
]

STAR_STEP = 154  # 770 MMR на медаль / 5 звёзд
IMMORTAL = 8
MIN_TIER, MAX_TIER = 10, 90


def mmr_to_rank_tier(mmr: int) -> int:
    """MMR -> rank_tier вида 54 (Legend 4). У Immortal звёзд нет -> 80."""
    if mmr < 0:
        raise ValueError("MMR не может быть отрицательным")

    medal, _, floor_mmr = next(
        (m for m in reversed(MEDALS) if mmr >= m[2]),
        MEDALS[0],
    )
    if medal == IMMORTAL:
        return IMMORTAL * 10

    star = min(int((mmr - floor_mmr) // STAR_STEP) + 1, 5)
    return medal * 10 + star


def rank_tier_name(rank_tier: int) -> str:
    """54 -> 'Legend 4', 80 -> 'Immortal'."""
    medal, star = divmod(rank_tier, 10)
    name = next((m[1] for m in MEDALS if m[0] == medal), f"tier {medal}")
    return name if medal == IMMORTAL or star == 0 else f"{name} {star}"


def sample_window(mmr: int, spread_medals: int = 1) -> tuple[int, int]:
    """Границы выборки вокруг уровня игрока: его медаль ± spread_medals.

    Возвращает (min_rank, max_rank) в формате API. Смысл в том, чтобы
    модель училась на популяции, похожей на ту, в которой игрок реально
    играет: на Herald и на Divine ценность одних и тех же героев разная.
    """
    if spread_medals < 0:
        raise ValueError("spread_medals не может быть отрицательным")

    medal = mmr_to_rank_tier(mmr) // 10
    low = max(MIN_TIER, (medal - spread_medals) * 10)
    high = min(MAX_TIER, (medal + spread_medals + 1) * 10)
    return low, high


def describe_window(mmr: int, spread_medals: int = 1) -> str:
    """Человекочитаемое описание выборки — печатается перед выгрузкой."""
    low, high = sample_window(mmr, spread_medals)
    tier = mmr_to_rank_tier(mmr)
    # max_rank у API — верхняя отсечка: при max_rank=60 приходят тиры 51..55,
    # то есть последняя ВКЛЮЧЁННАЯ медаль на единицу меньше high // 10.
    top_medal = min(high // 10 - 1, IMMORTAL)
    top_name = next((m[1] for m in MEDALS if m[0] == top_medal), f"tier {top_medal}")
    return (
        f"{mmr} MMR -> {rank_tier_name(tier)}; "
        f"выборка: {rank_tier_name(low)} .. {top_name} (min_rank={low}, max_rank={high})"
    )
