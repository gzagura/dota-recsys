"""Профили игроков по steam32-id: подгрузка из OpenDota, кеш, фолбэк.

Зачем отдельный слой. В курсовой версии профиль был один и запекался в
рекомендатель при старте. На сайте у каждого пользователя свой аккаунт,
поэтому профиль стал данными запроса: модель и матрицы общие, а поправка
на личный winrate подставляется на время одного вызова.

Три вещи, которые здесь решаются:

1. История игрока может быть закрыта. В Dota публикация матчей — это
   отдельная галка «Expose Public Match Data», и без неё OpenDota отдаёт
   пустой список. Это не ошибка, а нормальный исход, поэтому статус
   возвращается явно и сайт показывает чистый L3 вместо персонализации.
2. Игр может быть слишком мало. На двадцати матчах личный winrate — шум,
   и подмешивать его в выдачу вреднее, чем не подмешивать вовсе.
3. Ходить в OpenDota на каждый клик нельзя: у бесплатного тарифа лимит
   ~60 запросов в минуту на IP, и это лимит всего сервера, а не одного
   пользователя. Поэтому профиль кешируется на диск и переспрашивается
   не чаще раза в сутки.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from src.features.player_profile import build_player_profile
from src.models.l3_lgbm import player_delta

# Ниже этого числа матчей персонализацию не включаем: поправка считается
# со сжатием к 0.5, но на такой выборке она всё равно шумит.
MIN_GAMES = 50

# Сколько «виртуальных» игр по 50% подмешивается к личной статистике героя.
# Смысл числа: примерно столько игр на герое нужно, чтобы личный winrate
# получил половину своего веса. Тридцать, а не десять, из-за многопользо-
# вательности: у друзей истории короче, и на восьми играх со случайными
# семью победами герой иначе выпрыгивает в топ вперёд объективно лучших.
PRIOR_GAMES = 30.0

# Сколько живёт кеш профиля. Сутки: за игровую сессию winrate по герою
# заметно не меняется, а лимит OpenDota экономится сильно.
CACHE_TTL_SEC = 24 * 3600

# А вот отказы залипать на сутки не должны. Человек, увидевший «история
# закрыта», идёт эту историю открывать — и возвращается через пять минут.
# Если держать отрицательный ответ сутки, он увидит ровно то же самое и
# решит, что сайт сломан. Именно так и вышло с первым же живым
# пользователем: галку в Доте он включил, данные появились, а сайт
# продолжал показывать вчерашний отказ.
NEGATIVE_TTL_SEC = 10 * 60

# Запас по времени ответа: OpenDota на больших аккаунтах отвечает секунды.
REQUEST_TIMEOUT_SEC = 30

STATUS_MESSAGES = {
    "ok": "Персонализация включена",
    "anonymous": "Steam ID не указан — рекомендации по чистой модели драфта",
    "private": (
        "История матчей закрыта. Включи в Dota 2 настройку "
        "«Expose Public Match Data» — и профиль подтянется"
    ),
    "too_few_games": (
        "Слишком мало матчей в истории — личная статистика пока "
        "не влияет на выдачу"
    ),
    "bad_id": "Steam32 ID должен быть положительным числом",
    "unavailable": "OpenDota сейчас недоступна, работаем без персонализации",
}


@dataclass
class PlayerProfile:
    """Результат попытки подтянуть профиль.

    `delta` пустой во всех неуспешных случаях, поэтому вызывающему коду не
    нужно разбирать статус, чтобы посчитать рекомендации: пустая поправка
    просто означает чистый L3. Статус нужен только для сообщения в UI.
    """

    account_id: int
    status: str
    games: int = 0
    delta: dict[int, float] = field(default_factory=dict)
    stats: dict[int, tuple[int, float]] = field(default_factory=dict)

    @property
    def personalized(self) -> bool:
        return self.status == "ok" and bool(self.delta)

    @property
    def message(self) -> str:
        return STATUS_MESSAGES.get(self.status, STATUS_MESSAGES["unavailable"])

    def as_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "status": self.status,
            "personalized": self.personalized,
            "games": self.games,
            "message": self.message,
        }


def parse_account_id(value: object) -> int | None:
    """Steam32 из пользовательского ввода.

    Люди путают steam32 (account_id) и steam64 (steamid), потому что оба
    называют «мой стим айди». Steam64 распознаётся по величине и молча
    переводится в steam32 — иначе пользователь получил бы «профиль не
    найден» и не понял, почему.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text.isdigit():
        return None
    number = int(text)
    if number > 76561197960265728:
        number -= 76561197960265728
    return number if 0 < number < 2**32 else None


class ProfileStore:
    """Кеш профилей: память -> диск -> OpenDota."""

    def __init__(
        self,
        cache_dir: Path,
        base_url: str = "https://api.opendota.com/api",
        api_key: str | None = None,
        min_games: int = MIN_GAMES,
        ttl_sec: int = CACHE_TTL_SEC,
        match_limit: int = 20000,
        prior_games: float = PRIOR_GAMES,
        negative_ttl_sec: int = NEGATIVE_TTL_SEC,
    ):
        self.negative_ttl_sec = negative_ttl_sec
        self.prior_games = prior_games
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.min_games = min_games
        self.ttl_sec = ttl_sec
        self.match_limit = match_limit
        self._memory: dict[int, tuple[float, PlayerProfile]] = {}

    # ------------------------------------------------------------ публичное

    def get(
        self, account_id: int, force: bool = False, allow_fetch: bool = True
    ) -> PlayerProfile | None:
        """Профиль из кеша, при необходимости — из OpenDota.

        `allow_fetch=False` нужен горячему пути рекомендаций: там профиль
        уже должен лежать в кеше после явной загрузки, и ходить в сеть на
        каждый клик по герою нельзя. В этом случае промах кеша возвращает
        None, а не блокирует запрос на секунды.
        """
        cached = self._memory.get(account_id)
        if cached and not force and time.time() - cached[0] < self._ttl_for(cached[1].status):
            return cached[1]

        if not force:
            from_disk = self._read_disk(account_id)
            if from_disk is not None:
                self._memory[account_id] = (time.time(), from_disk)
                return from_disk

        if not allow_fetch:
            return None

        profile = self._fetch(account_id)
        # Неудачу сети не кешируем на диск: она про сеть, а не про игрока,
        # и через минуту запрос может пройти. В памяти держим короткое
        # время, чтобы серия кликов не превратилась в серию запросов.
        if profile.status != "unavailable":
            self._write_disk(profile)
        self._memory[account_id] = (time.time(), profile)
        return profile

    # -------------------------------------------------------------- внутрь

    def _ttl_for(self, status: str) -> float:
        """Удачный профиль живёт сутки, отказ — минуты (см. NEGATIVE_TTL_SEC)."""
        return self.ttl_sec if status == "ok" else self.negative_ttl_sec

    def _paths(self, account_id: int) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{account_id}.parquet",
            self.cache_dir / f"{account_id}.json",
        )

    def _read_disk(self, account_id: int) -> PlayerProfile | None:
        _, meta_path = self._paths(account_id)
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        status = str(meta.get("status", "unavailable"))
        games = int(meta.get("games", 0))
        if time.time() - float(meta.get("fetched_at", 0)) > self._ttl_for(status):
            return None

        if status != "ok":
            return PlayerProfile(account_id=account_id, status=status, games=games)

        table_path, _ = self._paths(account_id)
        if not table_path.exists():
            return None
        try:
            frame = pd.read_parquet(table_path)
        except (OSError, ValueError):
            return None
        return self._from_frame(account_id, frame, games)

    def _write_disk(self, profile: PlayerProfile) -> None:
        table_path, meta_path = self._paths(profile.account_id)
        meta = {
            "account_id": profile.account_id,
            "status": profile.status,
            "games": profile.games,
            "fetched_at": time.time(),
        }
        try:
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except OSError:
            pass

    def _fetch(self, account_id: int) -> PlayerProfile:
        params: dict[str, object] = {"limit": self.match_limit}
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            resp = requests.get(
                f"{self.base_url}/players/{account_id}/matches",
                params=params,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            matches = resp.json()
        except (requests.RequestException, ValueError):
            return PlayerProfile(account_id=account_id, status="unavailable")

        if not isinstance(matches, list) or not matches:
            return PlayerProfile(account_id=account_id, status="private")

        usable = [
            m
            for m in matches
            if isinstance(m, dict)
            and m.get("hero_id")
            and m.get("player_slot") is not None
            and m.get("radiant_win") is not None
        ]
        if len(usable) < self.min_games:
            return PlayerProfile(
                account_id=account_id,
                status="private" if not usable else "too_few_games",
                games=len(usable),
            )

        frame = build_player_profile(usable)
        profile = self._from_frame(account_id, frame, len(usable))
        table_path, _ = self._paths(account_id)
        try:
            frame.to_parquet(table_path)
        except (OSError, ValueError):
            pass
        return profile

    def _from_frame(
        self, account_id: int, frame: pd.DataFrame, games: int
    ) -> PlayerProfile:
        return PlayerProfile(
            account_id=account_id,
            status="ok",
            games=games,
            delta=player_delta(frame, self.prior_games),
            stats={
                int(r.hero_id): (int(r.games), float(r.winrate))
                for r in frame.itertuples()
            },
        )
