"""Ручки сайта: калитка и фолбэки профиля.

Эти тесты появились не от хорошей жизни. Проверка пароля жила в
require_user, но FastAPI отвечал 401 раньше неё — у HTTPBasic по умолчанию
auto_error=True. Из-за этого выключатель DRAFT_NO_AUTH молча не работал, и
поймать это на уровне юнит-тестов реестра было невозможно: баг жил в
проводке между FastAPI и нашей функцией, а не в самой проверке.
"""
import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
pytest.importorskip("httpx")

from src.web.auth import UserRegistry, hash_password  # noqa: E402

PASSWORD = "test-password"


def make_app(monkeypatch, no_auth: bool = False):
    """Поднимает приложение с подменённым реестром и без похода за моделью."""
    monkeypatch.setenv("DRAFT_NO_AUTH", "1" if no_auth else "0")

    import src.web.app as app_module

    # AUTH_DISABLED читается на импорте, поэтому модуль перезагружаем.
    app_module = importlib.reload(app_module)
    app_module.registry = UserRegistry(users={"friend": hash_password(PASSWORD)})
    return app_module


@pytest.fixture
def client(monkeypatch):
    app_module = make_app(monkeypatch)
    return fastapi_testclient.TestClient(app_module.app)


# ------------------------------------------------------------------ калитка


def test_без_пароля_не_пускает(client):
    assert client.get("/api/heroes").status_code == 401


def test_без_пароля_браузеру_предлагают_ввести_его(client):
    """Без WWW-Authenticate браузер не покажет окно логина."""
    assert client.get("/api/heroes").headers.get("www-authenticate") == "Basic"


def test_неверный_пароль_не_пускает(client):
    r = client.get("/api/heroes", auth=("friend", "wrong-password"))
    assert r.status_code == 401


def test_неизвестный_логин_не_пускает(client):
    r = client.get("/api/heroes", auth=("stranger", PASSWORD))
    assert r.status_code == 401


def test_рекомендации_тоже_закрыты(client):
    assert client.post("/api/recommend", json={}).status_code == 401


def test_профиль_тоже_закрыт(client):
    assert client.post("/api/profile", json={"steam_id": "1"}).status_code == 401


def test_страница_отдаётся_без_пароля(client):
    """На самой странице нет данных — только разметка."""
    assert client.get("/").status_code == 200


def test_выключатель_реально_выключает_калитку(monkeypatch):
    """DRAFT_NO_AUTH=1 должен пускать вообще без заголовка Authorization."""
    app_module = make_app(monkeypatch, no_auth=True)
    with fastapi_testclient.TestClient(app_module.app) as c:
        assert c.post("/api/profile", json={"steam_id": "не число"}).status_code == 200


# ------------------------------------------------------------------ профиль


def test_мусорный_steam_id_не_ходит_в_сеть(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("на мусорном вводе в OpenDota ходить не надо")

    monkeypatch.setattr("src.profiles.requests.get", boom)
    r = client.post("/api/profile", json={"steam_id": "не число"}, auth=("friend", PASSWORD))
    assert r.status_code == 200
    assert r.json()["status"] == "bad_id"
    assert r.json()["personalized"] is False


def test_у_каждого_статуса_есть_текст_для_человека(client):
    """Интерфейс показывает r.message, поэтому пустым он быть не может."""
    r = client.post("/api/profile", json={"steam_id": "0"}, auth=("friend", PASSWORD))
    assert r.json()["message"]
