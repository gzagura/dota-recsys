"""
Тесты L3: матрицы -> фичи -> обучение -> рекомендация.

Данные синтетические, с заложенным сигналом: так проверяется не просто
«код не падает», а что модель действительно вылавливает контекст драфта.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build_synergy_matrix import MatchArrays, build_matrices_from_matches
from src.features.build_training_dataset import ALLY_COLUMNS, ENEMY_COLUMNS, build_dataset
from src.features.draft_features import FEATURE_COLUMNS, DraftFeatureBuilder
from src.models.l3_lgbm import L3Recommender, train_l3

N_HEROES = 20
# Внутренние индексы 0..19 соответствуют hero_id 1..20.
STRONG = 0  # hero_id 1  — сильный сам по себе
PAIR_A, PAIR_B = 1, 2  # hero_id 2 и 3 — сильны только ВМЕСТЕ


@pytest.fixture(scope="module")
def heroes() -> pd.DataFrame:
    attrs = ["str", "agi", "int", "all"]
    roles = [["Carry", "Escape"], ["Support", "Disabler"], ["Nuker"], ["Durable", "Initiator"]]
    return pd.DataFrame(
        {
            "hero_id": list(range(1, N_HEROES + 1)),
            "localized_name": [f"Hero {i}" for i in range(1, N_HEROES + 1)],
            "primary_attr": [attrs[i % len(attrs)] for i in range(N_HEROES)],
            "roles": [roles[i % len(roles)] for i in range(N_HEROES)],
        }
    )


def make_matches(n_matches: int = 4000, seed: int = 0) -> MatchArrays:
    """Синтетические матчи с двумя заложенными эффектами.

    1) STRONG выигрывает чаще сам по себе (это ловится базовым winrate);
    2) PAIR_A + PAIR_B сильны только в ОДНОЙ команде (чистая синергия —
       по одиночным winrate такой эффект не виден).
    """
    rng = np.random.default_rng(seed)
    picks = np.array([rng.choice(N_HEROES, size=10, replace=False) for _ in range(n_matches)])
    radiant, dire = picks[:, :5], picks[:, 5:]

    p = np.full(n_matches, 0.5)
    p = np.where((radiant == STRONG).any(axis=1), 0.75, p)
    p = np.where((dire == STRONG).any(axis=1), 0.25, p)

    pair_r = (radiant == PAIR_A).any(axis=1) & (radiant == PAIR_B).any(axis=1)
    pair_d = (dire == PAIR_A).any(axis=1) & (dire == PAIR_B).any(axis=1)
    p = np.where(pair_r, 0.92, p)
    p = np.where(pair_d, 0.08, p)

    return MatchArrays(
        radiant=radiant,
        dire=dire,
        radiant_win=(rng.random(n_matches) < p).astype(np.int8),
        match_id=np.arange(n_matches, dtype=np.int64) + 1_000_000,
        hero_ids=np.arange(1, N_HEROES + 1, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def matches() -> MatchArrays:
    return make_matches()


@pytest.fixture(scope="module")
def artifacts(matches):
    return build_matrices_from_matches(matches, prior_weight=10.0)


# --------------------------- матрицы ---------------------------


def test_base_winrate_reflects_planted_strength(artifacts):
    _, _, hero_stats = artifacts
    stats = hero_stats.set_index("hero_id")
    assert stats.loc[STRONG + 1, "winrate"] > 0.6
    assert stats.loc[10, "winrate"] == pytest.approx(0.5, abs=0.05)  # нейтральный герой


def test_synergy_catches_pair_effect(artifacts):
    synergy, _, _ = artifacts
    a, b = PAIR_A + 1, PAIR_B + 1
    assert synergy.loc[a, b] > 0.1
    assert synergy.loc[b, a] > 0.1
    # с посторонним героем синергии быть не должно
    assert abs(synergy.loc[a, 10]) < 0.05


def test_counter_is_antisymmetric_in_sign(artifacts):
    _, counter, _ = artifacts
    # STRONG выигрывает у всех, значит его контрпик-дельта против
    # произвольного героя положительна, а обратная — отрицательна.
    assert counter.loc[STRONG + 1, 10] > 0
    assert counter.loc[10, STRONG + 1] < 0


def test_diagonal_is_zero(artifacts):
    synergy, counter, _ = artifacts
    assert np.allclose(np.diag(synergy.to_numpy()), 0.0)
    assert np.allclose(np.diag(counter.to_numpy()), 0.0)


# --------------------------- фичи ---------------------------


@pytest.fixture(scope="module")
def builder(artifacts, heroes) -> DraftFeatureBuilder:
    synergy, counter, hero_stats = artifacts
    return DraftFeatureBuilder.from_frames(hero_stats, synergy, counter, heroes)


def test_build_context_shape_and_columns(builder):
    features = builder.build_context(candidate_ids=[4, 5, 6], allies=[1], enemies=[2, 3])
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == 3
    assert (features["n_allies"] == 1).all()
    assert (features["n_enemies"] == 2).all()


def test_empty_draft_gives_zero_aggregates(builder):
    features = builder.build_context(candidate_ids=[7], allies=[], enemies=[])
    for col in ("syn_mean", "syn_max", "syn_min", "ctr_mean", "ally_base_wr_mean"):
        assert features[col].iloc[0] == 0.0
    assert features["n_allies"].iloc[0] == 0


def test_synergy_aggregate_matches_matrix(builder, artifacts):
    synergy, _, _ = artifacts
    a, b = PAIR_A + 1, PAIR_B + 1
    features = builder.build_context(candidate_ids=[b], allies=[a, 10], enemies=[])
    expected_mean = (synergy.loc[b, a] + synergy.loc[b, 10]) / 2
    assert features["syn_mean"].iloc[0] == pytest.approx(expected_mean, abs=1e-5)
    assert features["syn_max"].iloc[0] == pytest.approx(max(synergy.loc[b, a], synergy.loc[b, 10]), abs=1e-5)


def test_ally_role_counts(builder, heroes):
    # hero_id 1 и 5 — оба Carry/Escape (roles заданы циклом по 4)
    features = builder.build_context(candidate_ids=[9], allies=[1, 5], enemies=[])
    assert features["ally_role_carry"].iloc[0] == 2
    assert features["ally_role_support"].iloc[0] == 0


def test_unknown_heroes_are_dropped_from_context(builder):
    """Герой, которого нет в матрицах, не должен ронять инференс."""
    features = builder.build_context(candidate_ids=[4], allies=[1, 999], enemies=[])
    assert features["n_allies"].iloc[0] == 1


# --------------------------- датасет ---------------------------


@pytest.fixture(scope="module")
def dataset(matches, heroes) -> pd.DataFrame:
    data, _ = build_dataset(
        matches=matches, heroes=heroes, val_fraction=0.2, prior_weight=10.0, seed=42
    )
    return data


def test_dataset_has_ten_samples_per_match(dataset, matches):
    assert len(dataset) == 10 * len(matches)
    assert set(dataset["split"]) == {"train", "val"}


def test_dataset_labels_are_balanced(dataset):
    # На каждый матч приходится 5 победивших и 5 проигравших кандидатов.
    assert dataset["label"].mean() == pytest.approx(0.5, abs=1e-9)


def test_split_is_temporal(dataset):
    train_max = dataset.loc[dataset["split"] == "train", "match_id"].max()
    val_min = dataset.loc[dataset["split"] == "val", "match_id"].min()
    assert train_max < val_min


def test_context_columns_agree_with_counts(dataset):
    sample = dataset.head(2000)
    allies_taken = (sample[ALLY_COLUMNS] >= 0).sum(axis=1)
    enemies_taken = (sample[ENEMY_COLUMNS] >= 0).sum(axis=1)
    assert (allies_taken == sample["n_allies"]).all()
    assert (enemies_taken == sample["n_enemies"]).all()


def test_candidate_never_appears_in_its_own_context(dataset):
    sample = dataset.head(2000)
    context = sample[ALLY_COLUMNS + ENEMY_COLUMNS].to_numpy()
    assert not (context == sample["hero_id"].to_numpy()[:, None]).any()


# --------------------------- модель ---------------------------


@pytest.fixture(scope="module")
def trained(dataset):
    return train_l3(
        dataset, params={"num_iterations": 150, "num_leaves": 15}, early_stopping_rounds=30
    )


def test_model_beats_winrate_only_baseline(trained):
    _, metrics = trained
    assert metrics["val_auc"] > 0.55
    # Смысл L3 — выучить контекст, а не только силу самого героя.
    assert metrics["val_auc"] > metrics["val_auc_base_wr_only"]


@pytest.fixture(scope="module")
def recommender(trained, builder, heroes) -> L3Recommender:
    booster, _ = trained
    return L3Recommender(
        booster=booster,
        feature_builder=builder,
        hero_names=heroes[["hero_id", "localized_name"]],
    )


def test_recommend_excludes_taken_heroes(recommender):
    taken = {1, 2, 3}
    result = recommender.recommend(allies=[2], enemies=[3], exclude_ids=taken, top_k=5)
    assert len(result) == 5
    assert not set(result["hero_id"]) & taken
    assert list(result.columns) == [
        "hero_id",
        "score",
        "p_win",
        "player_bonus",
        "localized_name",
    ]
    assert result["score"].is_monotonic_decreasing


def test_score_split_into_probability_and_personal_bonus(trained, builder, heroes):
    """score = p_win + w_player * дельта игрока, и слагаемые видны раздельно.

    Без разделения бот показывал бы скор с персонализацией как «шанс
    победы», хотя вероятностью он уже не является.
    """
    booster, _ = trained
    profile = pd.DataFrame({"hero_id": [5], "games": [100], "wins": [90], "winrate": [0.9]})
    rec = L3Recommender(
        booster=booster,
        feature_builder=builder,
        hero_names=heroes[["hero_id", "localized_name"]],
        player_profile=profile,
        w_player=1.0,
    )
    result = rec.recommend(allies=[2], enemies=[3], exclude_ids={1, 2, 3}, top_k=10)

    assert ((result["p_win"] >= 0.0) & (result["p_win"] <= 1.0)).all()
    assert np.allclose(result["score"], result["p_win"] + result["player_bonus"])
    boosted = result.loc[result["hero_id"] == 5]
    assert not boosted.empty and boosted["player_bonus"].iloc[0] > 0.0


def test_recommendation_reacts_to_ally_context(recommender):
    """Синергичная пара должна подниматься именно рядом со своим партнёром."""
    a, b = PAIR_A + 1, PAIR_B + 1
    with_partner = recommender.score_candidates([b], allies=[a], enemies=[])[0]
    without_partner = recommender.score_candidates([b], allies=[10], enemies=[])[0]
    assert with_partner > without_partner


def test_player_profile_shifts_scores(trained, builder, heroes):
    booster, _ = trained
    profile = pd.DataFrame({"hero_id": [7], "games": [100], "wins": [90]})
    plain = L3Recommender(
        booster=booster, feature_builder=builder, hero_names=heroes[["hero_id", "localized_name"]]
    )
    personal = L3Recommender(
        booster=booster,
        feature_builder=builder,
        hero_names=heroes[["hero_id", "localized_name"]],
        player_profile=profile,
        w_player=1.0,
    )
    assert personal.score_candidates([7], [], [])[0] > plain.score_candidates([7], [], [])[0]
    # Герой без записей в профиле не должен сдвигаться.
    assert personal.score_candidates([9], [], [])[0] == pytest.approx(
        plain.score_candidates([9], [], [])[0]
    )
