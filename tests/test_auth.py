"""Калитка: проверка паролей, рейт-лимит, заведение пользователей."""
import json
import time

import pytest

from src.web.auth import RateLimiter, UserRegistry, hash_password, init_users


@pytest.fixture
def registry():
    return UserRegistry(users={"друг": hash_password("правильный-пароль")})


def test_верный_пароль_пускает(registry):
    assert registry.verify("друг", "правильный-пароль")


def test_неверный_пароль_не_пускает(registry):
    assert not registry.verify("друг", "неправильный")


def test_неизвестный_логин_не_пускает(registry):
    assert not registry.verify("чужой", "правильный-пароль")


def test_пароль_в_файле_не_хранится_открытым(tmp_path):
    path = tmp_path / "users.json"
    created = init_users(["друг"], path=path)
    password = created["друг"]

    raw = path.read_text(encoding="utf-8")
    assert password not in raw
    assert UserRegistry.load(path).verify("друг", password)


def test_у_каждого_пользователя_своя_соль(tmp_path):
    """Одинаковые пароли не должны давать одинаковые хеши."""
    same = "один-и-тот-же-пароль"
    users = {
        "первый": hash_password(same),
        "второй": hash_password(same),
    }
    assert users["первый"]["hash"] != users["второй"]["hash"]
    assert users["первый"]["salt"] != users["второй"]["salt"]


def test_повторный_init_не_затирает_старых(tmp_path):
    path = tmp_path / "users.json"
    first = init_users(["первый"], path=path)
    init_users(["второй"], path=path)

    registry = UserRegistry.load(path)
    assert registry.verify("первый", first["первый"])
    assert set(json.loads(path.read_text(encoding="utf-8"))["users"]) == {"первый", "второй"}


def test_пароли_генерируются_разные(tmp_path):
    created = init_users([f"друг{i}" for i in range(10)], path=tmp_path / "users.json")
    assert len(set(created.values())) == 10
    assert all(len(p) >= 12 for p in created.values())


def test_отсутствие_файла_это_понятная_ошибка(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.web.auth init"):
        UserRegistry.load(tmp_path / "нет-такого.json")


def test_кеш_проверки_не_ломает_отказ(registry):
    """После удачного входа неверный пароль всё равно должен отлетать."""
    assert registry.verify("друг", "правильный-пароль")
    assert not registry.verify("друг", "правильный-пароль-почти")


# ------------------------------------------------------------- рейт-лимит


def test_рейт_лимит_отсекает_после_исчерпания():
    limiter = RateLimiter(capacity=3, refill_sec=60)
    assert all(limiter.allow("друг") for _ in range(3))
    assert not limiter.allow("друг")


def test_лимит_считается_на_каждого_отдельно():
    limiter = RateLimiter(capacity=1, refill_sec=60)
    assert limiter.allow("первый")
    assert limiter.allow("второй")
    assert not limiter.allow("первый")


def test_ведро_наполняется_со_временем(monkeypatch):
    limiter = RateLimiter(capacity=2, refill_sec=10)
    assert limiter.allow("друг")
    assert limiter.allow("друг")
    assert not limiter.allow("друг")

    now = time.time() + 10
    monkeypatch.setattr("src.web.auth.time.time", lambda: now)
    assert limiter.allow("друг")
