"""Аутентификация на десяток человек: логин, пароль, ничего лишнего.

Зачем она вообще нужна на сайте, который ничего не хранит. Ломать тут
нечего — записи нет, базы нет, секретов на странице нет. Дело в другом:
как только появился ввод steam-id, любой прохожий может заставить наш
сервер ходить в OpenDota произвольными аккаунтами. Лимит там на IP, то
есть общий на весь сервер, и выбьют его нам, а не гостю. Плюс краулеры
без калитки будут молотить L3 просто потому, что страница открыта.

Почему HTTP Basic, а не форма с сессией. Ради десяти друзей заводить
куки, подписи и страницу логина — больше кода, чем защиты. Basic целиком
лежит на браузере: он сам показывает окно и сам помнит пароль. Цена —
пароль летит в заголовке каждого запроса, поэтому HTTPS обязателен, и
выйти из аккаунта можно только закрыв браузер.

Пароли в файле лежат хешами (scrypt из стандартной библиотеки): файл
может утечь вместе с бэкапом, а на VPS его увидит любой, у кого есть
root. Сравнение хешей — через compare_digest, чтобы по времени ответа
нельзя было подбирать пароль посимвольно.

Файл пользователей создаётся так:

    python -m src.web.auth init друг1 друг2 друг3

Команда напечатает сгенерированные пароли (единственный раз, когда их
видно) и положит рядом configs/users.txt со списком «логин: пароль»,
чтобы было что раздать. Оба файла — в .gitignore.
"""
from __future__ import annotations

import json
import secrets
import sys
import time
from dataclasses import dataclass, field
from hashlib import scrypt
from pathlib import Path

USERS_PATH = Path("configs/users.json")
PLAINTEXT_PATH = Path("configs/users.txt")

# Параметры scrypt. n=2**14 — примерно 50 мс на проверку: человек разницы
# не заметит, а перебору это дорого. Basic шлёт пароль на каждый запрос,
# поэтому результат проверки кешируется (см. UserRegistry.verify).
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
DK_LEN = 32

ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_password(length: int = 14) -> str:
    """Пароль без символов, которые путают при диктовке: 0/O, 1/l/I."""
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def hash_password(password: str, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    digest = scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=DK_LEN
    )
    return {"salt": salt.hex(), "hash": digest.hex()}


@dataclass
class UserRegistry:
    """Пользователи из файла + кеш успешных проверок."""

    users: dict[str, dict]
    # Basic-аутентификация присылает пароль с КАЖДЫМ запросом, а scrypt
    # намеренно медленный. Без кеша один клик по герою стоил бы лишних
    # 50 мс на ровном месте, поэтому удачная пара логин+пароль запоминается
    # на короткое время. Кеш живёт в памяти процесса и умирает с ним.
    _verified: dict[str, float] = field(default_factory=dict, repr=False)
    cache_ttl_sec: float = 300.0

    @classmethod
    def load(cls, path: Path = USERS_PATH) -> "UserRegistry":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Нет файла пользователей {path}. Создай его: "
                f"python -m src.web.auth init <логин> [<логин> ...]"
            )
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(users=data.get("users", data))

    def verify(self, login: str, password: str) -> bool:
        record = self.users.get(login)
        if record is None:
            # Пароль всё равно прогоняем через scrypt: иначе несуществующий
            # логин отвечал бы заметно быстрее существующего, и по времени
            # ответа можно было бы собрать список пользователей.
            hash_password(password)
            return False

        key = f"{login}:{password}"
        seen = self._verified.get(key)
        if seen is not None and time.time() - seen < self.cache_ttl_sec:
            return True

        expected = bytes.fromhex(record["hash"])
        actual = scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(record["salt"]),
            n=record.get("n", SCRYPT_N),
            r=record.get("r", SCRYPT_R),
            p=record.get("p", SCRYPT_P),
            dklen=len(expected),
        )
        if secrets.compare_digest(actual, expected):
            self._verified[key] = time.time()
            return True
        return False


@dataclass
class RateLimiter:
    """Ограничитель обращений к OpenDota, по одному ведру на пользователя.

    Логин защищает от улицы, но не от своих: один друг с открытой вкладкой
    и автообновлением способен выбрать общий лимит OpenDota на всех. Ведро
    маленькое, потому что профиль кешируется на сутки — честному человеку
    хватает одного запроса за сессию.
    """

    capacity: int = 10
    refill_sec: float = 60.0
    _buckets: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)

    def allow(self, key: str) -> bool:
        now = time.time()
        tokens, last = self._buckets.get(key, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - last) * self.capacity / self.refill_sec)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1.0, now)
        return True


def init_users(logins: list[str], path: Path = USERS_PATH) -> dict[str, str]:
    """Создаёт файл пользователей и возвращает пары логин -> пароль."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("users", {})

    created: dict[str, str] = {}
    for login in logins:
        password = generate_password()
        existing[login] = hash_password(password)
        created[login] = password

    path.write_text(json.dumps({"users": existing}, indent=2), encoding="utf-8")
    path.chmod(0o600)
    return created


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "init":
        print("Использование: python -m src.web.auth init <логин> [<логин> ...]")
        return 1

    logins = args[1:]
    created = init_users(logins)

    lines = [f"{login}: {password}" for login, password in created.items()]
    PLAINTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAINTEXT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    PLAINTEXT_PATH.chmod(0o600)

    print(f"Добавлено пользователей: {len(created)} -> {USERS_PATH}")
    print(f"Пароли (они же дописаны в {PLAINTEXT_PATH}):\n")
    print("\n".join(lines))
    print("\nОба файла в .gitignore. На сервер копируй только users.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
