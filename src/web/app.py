"""Веб-интерфейс драфта: сетка портретов, как на экране пика.

Состояние драфта живёт в браузере, бэкенд не имеет сессий: каждый запрос
присылает драфт целиком и получает пересчитанные рекомендации. Так любой
пик можно снять или заменить в любой момент, включая своего героя, и не
надо чинить рассинхрон между состоянием на клиенте и на сервере.

Персонализация устроена так же — без состояния. Клиент присылает свой
steam32-id вместе с драфтом, сервер достаёт готовый профиль из кеша и
подставляет поправку на время одного расчёта. Профиль общий на процесс
хранить нельзя: пользователей несколько, и чужая статистика в выдаче —
это не «неточность», а показ чужих данных.

Вся модельная логика лежит в src/service.py.

Запуск:
    python -m src.web.app
    # затем открыть http://127.0.0.1:8000
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from src.profiles import STATUS_MESSAGES, PlayerProfile, parse_account_id
from src.service import DraftService
from src.web.auth import RateLimiter, UserRegistry

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Dota 2 Draft Recommender")
security = HTTPBasic()

service: DraftService | None = None
registry: UserRegistry | None = None
limiter = RateLimiter()

# Аварийный выключатель калитки: пускает всех без пароля. Нужен для
# локальной разработки, где заводить учётки ради одного запуска глупо.
# На сервере переменную не ставим никогда.
AUTH_DISABLED = os.environ.get("DRAFT_NO_AUTH") == "1"


def get_service() -> DraftService:
    global service
    if service is None:
        service = DraftService.load()
    return service


def get_registry() -> UserRegistry:
    global registry
    if registry is None:
        registry = UserRegistry.load()
    return registry


def require_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Проверка логина. Возвращает имя пользователя — оно же ключ рейт-лимита."""
    if AUTH_DISABLED:
        return "dev"
    if not get_registry().verify(credentials.username, credentials.password):
        # WWW-Authenticate обязателен: без него браузер не покажет окно
        # ввода пароля, а просто отрисует голую ошибку.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class DraftIn(BaseModel):
    allies: list[int] = []
    enemies: list[int] = []
    my_hero: int | None = None
    is_radiant: bool = True
    role: str | None = None
    top_k: int | None = None
    steam_id: str | None = None


class ProfileIn(BaseModel):
    steam_id: str
    refresh: bool = False


@app.get("/")
def index() -> FileResponse:
    # Саму страницу отдаём без пароля: на ней нет ни данных, ни выдачи —
    # только разметка. Пароль спросит первый же запрос к /api.
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/heroes")
def heroes(user: str = Depends(require_user)) -> dict:
    svc = get_service()
    return {
        "heroes": svc.hero_catalog(player=anonymous()),
        "roles": sorted(svc.role_to_ids),
        "max_allies": svc.max_allies,
        "max_enemies": svc.max_enemies,
        "level": svc.level,
        "user": user,
    }


@app.post("/api/profile")
def profile(payload: ProfileIn, user: str = Depends(require_user)) -> dict:
    """Подтягивает историю игрока из OpenDota. Единственная ручка, которая
    может ходить наружу, поэтому только она ограничена по частоте."""
    account_id = parse_account_id(payload.steam_id)
    if account_id is None:
        return {"status": "bad_id", "personalized": False, "games": 0,
                "message": STATUS_MESSAGES["bad_id"], "stats": {}}

    if not limiter.allow(user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Слишком часто. Профиль обновляется не чаще раза в минуту.",
        )

    svc = get_service()
    loaded = svc.profile_store().get(account_id, force=payload.refresh)
    result = loaded.as_dict()
    # Личную статистику отдаём сразу, чтобы фронт подписал её под портретами
    # и не спрашивал вторым запросом.
    result["stats"] = {
        str(hid): {"games": games, "winrate": winrate}
        for hid, (games, winrate) in loaded.stats.items()
    }
    return result


@app.post("/api/recommend")
def recommend(draft: DraftIn, user: str = Depends(require_user)) -> dict:
    svc = get_service()
    player = resolve_player(svc, draft.steam_id)

    # Свой герой исключается из кандидатов, но в контекст союзников не
    # входит: модель ранжирует замену именно ему, иначе он соперничал бы
    # сам с собой и всегда занимал первое место.
    exclude = set(draft.allies) | set(draft.enemies)
    if draft.my_hero is not None:
        exclude.add(draft.my_hero)

    recommendations = svc.recommend(
        allies=draft.allies,
        enemies=draft.enemies,
        exclude=exclude,
        is_radiant=draft.is_radiant,
        role=draft.role,
        top_k=draft.top_k,
        player=player,
    )

    team = draft.allies + ([draft.my_hero] if draft.my_hero is not None else [])
    p_win = svc.predict_team(team, draft.enemies, draft.is_radiant) if team else None
    threats = svc.threat_ranking(team, draft.enemies)

    return {
        "recommendations": recommendations,
        "p_win": p_win,
        "threats": threats,
        "team_size": len(team),
        "enemy_size": len(draft.enemies),
        "complete": len(team) == svc.max_allies + 1 and len(draft.enemies) == svc.max_enemies,
        "tactics": svc.tactics_for(team, draft.enemies),
        "my_roles": svc.role_counts(team),
        "enemy_roles": svc.role_counts(draft.enemies),
        "personalized": player.personalized,
    }


def anonymous() -> PlayerProfile:
    """Пустой профиль: чистый L3, без чьей-либо личной статистики."""
    return PlayerProfile(account_id=0, status="anonymous")


def resolve_player(svc: DraftService, steam_id: str | None) -> PlayerProfile:
    """Профиль для расчёта — строго из кеша.

    В сеть отсюда не ходим сознательно: рекомендации пересчитываются на
    каждый клик по герою, и поход в OpenDota превратил бы 7 мс в секунды.
    Профиль кладёт в кеш ручка /api/profile, а её вызывает фронт, когда
    пользователь ввёл steam-id.
    """
    account_id = parse_account_id(steam_id)
    if account_id is None:
        return anonymous()
    cached = svc.profile_store().get(account_id, allow_fetch=False)
    return cached or anonymous()


def main():
    get_service()  # поднимаем модель до старта сервера, а не на первом запросе
    if not AUTH_DISABLED:
        get_registry()  # и файл пользователей тоже: пусть падает сразу, а не в бою
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
