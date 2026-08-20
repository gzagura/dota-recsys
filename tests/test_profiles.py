"""Профили игроков: разбор steam-id, фолбэки, кеш."""
import json

import pandas as pd
import pytest

from src.profiles import MIN_GAMES, PlayerProfile, ProfileStore, parse_account_id


def make_matches(n: int, hero_id: int = 1, wins: int | None = None) -> list[dict]:
    """n матчей на одном герое; первые `wins` — победы."""
    wins = n if wins is None else wins
    return [
        {
            "hero_id": hero_id,
            "player_slot": 0,  # radiant
            "radiant_win": i < wins,
            "match_id": 1000 + i,
        }
        for i in range(n)
    ]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


@pytest.fixture
def store(tmp_path, monkeypatch):
    def make(payload):
        def fake_get(url, params=None, timeout=None):
            return FakeResponse(payload)

        monkeypatch.setattr("src.profiles.requests.get", fake_get)
        return ProfileStore(cache_dir=tmp_path)

    return make


# ----------------------------------------------------------------- steam id


def test_steam64_переводится_в_steam32():
    assert parse_account_id("76561198078622586") == 118356858


def test_steam32_остаётся_как_есть():
    assert parse_account_id("118356858") == 118356858


@pytest.mark.parametrize("bad", ["", "   ", "abc", "-5", "0", "12.5", None])
def test_мусорный_ввод_отбрасывается(bad):
    assert parse_account_id(bad) is None


# ------------------------------------------------------------------ фолбэки


def test_закрытая_история_не_ломает_выдачу(store):
    """OpenDota отдаёт пустой список — это норма, а не ошибка."""
    profile = store([]).get(1)
    assert profile.status == "private"
    assert profile.delta == {}
    assert not profile.personalized


def test_мало_игр_персонализацию_не_включает(store):
    profile = store(make_matches(MIN_GAMES - 1)).get(1)
    assert profile.status == "too_few_games"
    assert profile.delta == {}


def test_достаточно_игр_включает_персонализацию(store):
    profile = store(make_matches(MIN_GAMES + 10)).get(1)
    assert profile.status == "ok"
    assert profile.personalized
    assert profile.delta[1] > 0  # сплошные победы -> поправка вверх


def test_сеть_упала_персонализация_молча_выключается(tmp_path, monkeypatch):
    import requests

    def boom(url, params=None, timeout=None):
        raise requests.RequestException("нет сети")

    monkeypatch.setattr("src.profiles.requests.get", boom)
    profile = ProfileStore(cache_dir=tmp_path).get(1)
    assert profile.status == "unavailable"
    assert profile.delta == {}


def test_битые_матчи_отбрасываются(store):
    """Незавершённые матчи приходят с radiant_win = null."""
    matches = make_matches(MIN_GAMES + 5) + [
        {"hero_id": 5, "player_slot": 0, "radiant_win": None},
        {"hero_id": None, "player_slot": 0, "radiant_win": True},
    ]
    profile = store(matches).get(1)
    assert profile.games == MIN_GAMES + 5
    assert 5 not in profile.delta


# --------------------------------------------------------------------- кеш


def test_второй_запрос_идёт_из_кеша_без_сети(tmp_path, monkeypatch):
    calls = []

    def counting_get(url, params=None, timeout=None):
        calls.append(url)
        return FakeResponse(make_matches(MIN_GAMES + 1))

    monkeypatch.setattr("src.profiles.requests.get", counting_get)
    store = ProfileStore(cache_dir=tmp_path)
    store.get(42)
    store.get(42)
    assert len(calls) == 1


def test_кеш_переживает_перезапуск_процесса(tmp_path, monkeypatch):
    """Второй ProfileStore не должен ходить в сеть за тем же игроком."""
    monkeypatch.setattr(
        "src.profiles.requests.get",
        lambda url, params=None, timeout=None: FakeResponse(make_matches(MIN_GAMES + 1)),
    )
    ProfileStore(cache_dir=tmp_path).get(42)

    def boom(url, params=None, timeout=None):
        raise AssertionError("полез в сеть, хотя профиль лежит на диске")

    monkeypatch.setattr("src.profiles.requests.get", boom)
    assert ProfileStore(cache_dir=tmp_path).get(42).status == "ok"


def test_горячий_путь_не_ходит_в_сеть(tmp_path, monkeypatch):
    """allow_fetch=False: промах кеша возвращает None, а не тормозит запрос."""

    def boom(url, params=None, timeout=None):
        raise AssertionError("рекомендации не должны ходить в OpenDota")

    monkeypatch.setattr("src.profiles.requests.get", boom)
    assert ProfileStore(cache_dir=tmp_path).get(7, allow_fetch=False) is None


def test_протухший_кеш_перечитывается(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.profiles.requests.get",
        lambda url, params=None, timeout=None: FakeResponse(make_matches(MIN_GAMES + 1)),
    )
    store = ProfileStore(cache_dir=tmp_path, ttl_sec=0)
    store.get(42)
    meta = json.loads((tmp_path / "42.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok"
    assert store._read_disk(42) is None  # ttl=0 -> кеш сразу считается старым


# ------------------------------------------------------------ сжатие к 0.5


def test_приор_сжимает_короткую_выборку_сильнее_длинной(tmp_path, monkeypatch):
    """Чем короче выборка, тем дальше оценка уезжает от сырого winrate.

    Герой с 8 победами из 8 сырым winrate имеет 100%, а после сжатия ~60%:
    поправка теряет почти всё. Герой с 60 из 100 остаётся около своих 60%.
    Обратное сравнение (кто выше по итогу) осмысленной проверкой не является:
    8 из 8 — это тоже свидетельство в пользу героя, просто слабое.
    """
    matches = make_matches(8, hero_id=1) + make_matches(100, hero_id=2, wins=60)
    monkeypatch.setattr(
        "src.profiles.requests.get",
        lambda url, params=None, timeout=None: FakeResponse(matches),
    )
    profile = ProfileStore(cache_dir=tmp_path).get(1)

    короткая_потеря = 0.5 - profile.delta[1]  # сырая поправка была +0.5
    длинная_потеря = 0.1 - profile.delta[2]  # сырая поправка была +0.1
    assert короткая_потеря > длинная_потеря * 10


def test_анонимный_профиль_ничего_не_меняет():
    anon = PlayerProfile(account_id=0, status="anonymous")
    assert not anon.personalized
    assert anon.delta == {}
    assert anon.stats == {}
