"""Общий слой рекомендаций: вся модельная логика сайта.

Зачем отдельный модуль, а не код прямо в обработчиках HTTP: загрузка
артефактов, ранжирование, прогноз исхода и подсказки по составу не зависят
от способа ввода драфта. Интерфейс отвечает только за ввод-вывод, поэтому
его можно менять или добавить второй, не трогая формулы.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:  # только для аннотаций: src.profiles тянет за собой lightgbm
    from src.profiles import PlayerProfile

from src.features.build_training_dataset import MAX_ALLIES, MAX_ENEMIES
from src.hero_aliases import ALIASES, build_alias_lookup
from src.models.baseline import BaselineRecommender
from src.models.synergy_model import SynergyRecommender

CONFIG_PATH = Path("configs/config.yaml")

# Портреты берём с CDN Valve: hero name в справочнике имеет вид
# npc_dota_hero_antimage, в URL идёт хвост после префикса.
HERO_IMAGE_URL = (
    "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{slug}.png"
)
HERO_NAME_PREFIX = "npc_dota_hero_"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "").replace("'", "").replace("-", "")


# Со скольких выбранных героев команда считается достаточно определившейся,
# чтобы говорить о её составе. Правила вида «саппортов мало» или «инициации
# нет» на пустом драфте срабатывали бы всегда и означали бы не состав, а
# просто то, что герои ещё не введены.
MIN_TEAM_FOR_OWN_HINTS = 4
MIN_TEAM_FOR_ENEMY_HINTS = 3


def tactics(
    my_roles: dict[str, int],
    enemy_roles: dict[str, int],
    n_my: int,
    n_enemy: int,
) -> list[str]:
    """Короткие подсказки по составу команд.

    ВАЖНО про природу этих строк: это правила, написанные руками по тегам
    ролей, а НЕ вывод модели. Модель предсказывает только вероятность
    победы и ничего не знает про лейны, тайминги и построение драк — учить
    её этому не на чем, в публичных матчах таких меток нет. Поэтому
    подсказки подписаны как эвристика и намеренно осторожны.

    Подсказки про свою команду появляются только когда она почти собрана, а
    про вражескую — когда открыто хотя бы несколько пиков: иначе «нет
    инициации» говорилось бы про ещё не введённый драфт.
    """
    my = my_roles.get
    en = enemy_roles.get
    own_ready = n_my >= MIN_TEAM_FOR_OWN_HINTS
    enemy_ready = n_enemy >= MIN_TEAM_FOR_ENEMY_HINTS

    both = own_ready and enemy_ready
    # Кто входит в драку первым — одно решение, и советовать про него надо
    # один раз. Без этого флага в списке уживались «навязывайте драку
    # первыми» и «не входите в драку», что читается как шум.
    we_initiate = both and my("Initiator", 0) >= 2 and en("Initiator", 0) == 0

    rules = [
        (
            both and en("Initiator", 0) >= 2 and my("Initiator", 0) <= 1,
            "Инициация у них — играйте вторым номером: ждите их вход и отвечайте на него.",
        ),
        (
            we_initiate,
            "Инициатива у вас — навязывайте драку первыми, не давайте им собраться.",
        ),
        (
            both and not we_initiate and en("Disabler", 0) - my("Disabler", 0) >= 2,
            "Контроля у них заметно больше — не подходите поодиночке и не входите первыми.",
        ),
        (
            enemy_ready and not we_initiate and en("Disabler", 0) >= 3,
            "У них много контроля — не входите в драку толпой, ждите BKB.",
        ),
        (
            own_ready and my("Support", 0) <= 1,
            "Саппортов мало — лейны будут тяжёлыми, играйте от осторожной фазы.",
        ),
        (
            enemy_ready and en("Nuker", 0) >= 3,
            "Много бурста у врага — держите ХП и не стойте кучно.",
        ),
        (
            own_ready and enemy_ready and my("Pusher", 0) >= 2 and en("Pusher", 0) == 0,
            "Перевес в пуше — давите башни рано.",
        ),
        (enemy_ready and en("Carry", 0) >= 3, "У них тяжёлый лейт — не затягивайте игру."),
        (
            own_ready and my("Initiator", 0) == 0,
            "Инициации нет — не начинайте драки первыми, играйте вторым темпом.",
        ),
        (own_ready and my("Durable", 0) >= 3, "Вы толще — размены в драках в вашу пользу."),
        (
            both and en("Durable", 0) >= 3 and my("Durable", 0) <= 1,
            "Они толще — затяжной размен не в вашу пользу, бейте по одному.",
        ),
        (
            own_ready and my("Escape", 0) >= 3,
            "Много мобильности — играйте от разменов и сплитпуша.",
        ),
    ]
    return [text for ok, text in rules if ok][:3]


def threat_ranking(
    recommender,
    id_to_name: dict[int, str],
    team: list[int],
    enemies: list[int],
) -> list[dict]:
    """Враги по тому, насколько плохо ваш состав с ними справляется.

    Для каждого врага берётся средняя контр-дельта ваших героев против
    него: counter[мой][враг] — насколько мой герой в среднем выигрывает у
    этого врага. Самое низкое значение = герой, против которого драфт не
    подобрал ответа, то есть первый кандидат в фокус.

    Это по-прежнему попарная статистика, а не понимание боя: она ничего не
    знает про предметы, тайминги и позиции. На 50 тыс. матчей медиана по
    паре — 94 игры, так что оценка не из воздуха, но и не приговор.

    Функция вынесена на уровень модуля, чтобы её можно было позвать с любым
    рекомендателем, не поднимая вторую копию модели.
    """
    if not team or not enemies:
        return []

    rows = []
    builder = getattr(recommender, "feature_builder", None)
    if builder is not None:
        idx = builder.id_to_index
        mine = [idx[h] for h in team if h in idx]
        if not mine:
            return []
        for enemy in enemies:
            if enemy in idx:
                rows.append(
                    {
                        "hero_id": enemy,
                        "advantage": float(builder.counter[mine, idx[enemy]].mean()),
                    }
                )
    else:
        ctr = getattr(recommender, "counter_matrix", None)
        if ctr is None:
            return []
        for enemy in enemies:
            rows.append({"hero_id": enemy, "advantage": float(ctr.loc[team, enemy].mean())})

    for row in rows:
        row["name"] = id_to_name.get(row["hero_id"], str(row["hero_id"]))
    return sorted(rows, key=lambda r: r["advantage"])


@dataclass
class DraftService:
    """Рекомендатель + справочники, поднятые из артефактов пайплайна."""

    cfg: dict
    level: str
    recommender: object
    heroes: pd.DataFrame
    id_to_name: dict[int, str]
    id_to_roles: dict[int, list[str]]
    role_to_ids: dict[str, set[int]]
    player_stats: dict[int, tuple[int, float]]
    top_k: int
    name_lookup: dict[str, int] = field(default_factory=dict)
    alias_lookup: dict[str, int] = field(default_factory=dict)
    id_to_aliases: dict[int, list[str]] = field(default_factory=dict)
    _profiles: object = field(default=None, init=False, repr=False)

    # ------------------------------------------------------- профили юзеров

    def profile_store(self):
        """Кеш профилей игроков, общий на процесс.

        Создаётся лениво: без ввода steam-id он не нужен вовсе, а его
        конструктор лезет на диск за каталогом кеша.
        """
        if self._profiles is None:
            from src.profiles import PRIOR_GAMES, ProfileStore

            opendota = self.cfg.get("opendota", {})
            l3_cfg = self.cfg.get("model", {}).get("l3", {})
            self._profiles = ProfileStore(
                cache_dir=Path(self.cfg["data"].get("profiles_dir", "data/profiles")),
                base_url=opendota.get("base_url", "https://api.opendota.com/api"),
                api_key=opendota.get("api_key") or None,
                prior_games=float(l3_cfg.get("player_prior_games", PRIOR_GAMES)),
            )
        return self._profiles

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, cfg: dict | None = None) -> "DraftService":
        cfg = cfg or load_config()
        raw_dir = Path(cfg["data"]["raw_dir"])
        processed_dir = Path(cfg["data"]["processed_dir"])

        heroes = pd.DataFrame(
            json.loads((raw_dir / "heroes.json").read_text(encoding="utf-8"))
        ).rename(columns={"id": "hero_id"})

        id_to_name: dict[int, str] = {}
        id_to_roles: dict[int, list[str]] = {}
        role_to_ids: dict[str, set[int]] = {}
        name_lookup: dict[str, int] = {}
        for _, row in heroes.iterrows():
            hid = int(row["hero_id"])
            id_to_name[hid] = str(row["localized_name"])
            roles = list(row.get("roles") or [])
            id_to_roles[hid] = roles
            name_lookup[normalize_name(str(row["localized_name"]))] = hid
            for role in roles:
                role_to_ids.setdefault(role, set()).add(hid)

        name_to_id = {name: hid for hid, name in id_to_name.items()}
        alias_lookup = build_alias_lookup(name_to_id)
        id_to_aliases: dict[int, list[str]] = {}
        for canonical, variants in ALIASES.items():
            hid = name_to_id.get(canonical)
            if hid is not None:
                id_to_aliases[hid] = list(variants)

        hero_stats = cls._load_hero_stats(processed_dir, heroes)
        profile = cls._load_player_profile(cfg, processed_dir)
        player_stats = (
            {int(r.hero_id): (int(r.games), float(r.winrate)) for r in profile.itertuples()}
            if profile is not None
            else {}
        )

        level = str(cfg["model"].get("level", "L1")).upper()
        recommender = cls._build_recommender(cfg, level, heroes, hero_stats, profile)

        return cls(
            cfg=cfg,
            level=level,
            recommender=recommender,
            heroes=heroes,
            id_to_name=id_to_name,
            id_to_roles=id_to_roles,
            role_to_ids=role_to_ids,
            player_stats=player_stats,
            top_k=int(cfg["model"].get("top_k_recommendations", 5)),
            name_lookup=name_lookup,
            alias_lookup=alias_lookup,
            id_to_aliases=id_to_aliases,
        )

    def match_heroes(self, query: str, limit: int = 8) -> list[int]:
        """Кандидаты по свободному вводу: английское имя, русское или сленг.

        Порядок проверок — от самого уверенного совпадения к самому широкому:
        точное имя, точный алиас, префикс, подстрока. Точный алиас идёт
        раньше префикса не случайно: «па» — это Phantom Assassin, а не
        Pangolier с Pudge, которые тоже начинаются на эти буквы.
        """
        key = normalize_name(query)
        if not key:
            return []
        if key in self.name_lookup:
            return [self.name_lookup[key]]
        if key in self.alias_lookup:
            return [self.alias_lookup[key]]

        hits: list[int] = []
        for source in (self.name_lookup, self.alias_lookup):
            for name, hid in source.items():
                if name.startswith(key) and hid not in hits:
                    hits.append(hid)
        if hits:
            return hits[:limit]

        for source in (self.name_lookup, self.alias_lookup):
            for name, hid in source.items():
                if key in name and hid not in hits:
                    hits.append(hid)
        return hits[:limit]

    @staticmethod
    def _load_hero_stats(processed_dir: Path, heroes: pd.DataFrame) -> pd.DataFrame:
        stats_path = processed_dir / "hero_stats.parquet"
        if stats_path.exists():
            stats = pd.read_parquet(stats_path)
            return stats.merge(heroes[["hero_id", "localized_name"]], on="hero_id", how="left")
        stats = heroes[["hero_id", "localized_name"]].copy()
        stats["winrate"] = 0.5
        return stats

    @staticmethod
    def _load_player_profile(cfg: dict, processed_dir: Path) -> pd.DataFrame | None:
        account_id = cfg["opendota"].get("player_account_id")
        if not account_id:
            return None
        path = processed_dir / f"player_{account_id}_profile.parquet"
        return pd.read_parquet(path) if path.exists() else None

    @staticmethod
    def _build_recommender(
        cfg: dict,
        level: str,
        heroes: pd.DataFrame,
        hero_stats: pd.DataFrame,
        profile: pd.DataFrame | None,
    ):
        processed_dir = Path(cfg["data"]["processed_dir"])
        model_cfg = cfg["model"]

        if level == "L0":
            return BaselineRecommender(hero_stats=hero_stats)

        if level == "L3":
            from src.models.l3_lgbm import L3Recommender  # lightgbm нужен только здесь

            l3_cfg = model_cfg.get("l3", {})
            return L3Recommender.from_artifacts(
                processed_dir=processed_dir,
                heroes=heroes,
                model_file=l3_cfg.get("model_file", "l3_lgbm.txt"),
                player_profile=profile,
                w_player=l3_cfg.get("w_player", 0.0),
            )

        used_profile = profile if level == "L2" else None
        return SynergyRecommender(
            hero_stats=hero_stats,
            synergy_matrix=pd.read_parquet(processed_dir / "synergy_matrix.parquet"),
            counter_matrix=pd.read_parquet(processed_dir / "counter_matrix.parquet"),
            player_profile=used_profile,
            w_base=model_cfg.get("w_base", 1.0),
            w_synergy=model_cfg.get("w_synergy", 1.0),
            w_counter=model_cfg.get("w_counter", 1.0),
            w_player=model_cfg.get("w_player", 0.5) if used_profile is not None else 0.0,
        )

    # -------------------------------------------------------------- каталог

    def hero_catalog(self, player: "PlayerProfile | None" = None) -> list[dict]:
        """Справочник для интерфейса: имя, портрет, роли, личная статистика."""
        stats = player.stats if player is not None else self.player_stats
        out = []
        for _, row in self.heroes.iterrows():
            hid = int(row["hero_id"])
            slug = str(row["name"]).replace(HERO_NAME_PREFIX, "")
            games, winrate = stats.get(hid, (0, 0.0))
            out.append(
                {
                    "hero_id": hid,
                    "name": self.id_to_name[hid],
                    "slug": slug,
                    "image": HERO_IMAGE_URL.format(slug=slug),
                    "roles": self.id_to_roles.get(hid, []),
                    "aliases": self.id_to_aliases.get(hid, []),
                    "attr": row.get("primary_attr"),
                    "my_games": games,
                    "my_winrate": winrate,
                }
            )
        return sorted(out, key=lambda h: h["name"])

    def role_counts(self, hero_ids: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for hid in hero_ids:
            for role in self.id_to_roles.get(hid, []):
                counts[role] = counts.get(role, 0) + 1
        return counts

    # --------------------------------------------------------- рекомендации

    def pair_deltas(
        self, hero_id: int, allies: list[int], enemies: list[int]
    ) -> tuple[float | None, float | None]:
        """Средняя синергия со своими и контра против врагов — вход модели.

        Это не разложение предсказания по вкладам (для него нужен SHAP), а
        те самые агрегаты, которые модель получает на входе.
        """
        builder = getattr(self.recommender, "feature_builder", None)
        if builder is not None:
            idx = builder.id_to_index
            if hero_id not in idx:
                return None, None
            i = idx[hero_id]
            a = [idx[h] for h in allies if h in idx]
            e = [idx[h] for h in enemies if h in idx]
            return (
                float(builder.synergy[i, a].mean()) if a else None,
                float(builder.counter[i, e].mean()) if e else None,
            )

        syn_m = getattr(self.recommender, "synergy_matrix", None)
        ctr_m = getattr(self.recommender, "counter_matrix", None)
        if syn_m is None or ctr_m is None:
            return None, None
        return (
            float(syn_m.loc[hero_id, allies].mean()) if allies else None,
            float(ctr_m.loc[hero_id, enemies].mean()) if enemies else None,
        )

    def recommend(
        self,
        allies: list[int],
        enemies: list[int],
        exclude: set[int] | None = None,
        is_radiant: bool = True,
        role: str | None = None,
        top_k: int | None = None,
        player: "PlayerProfile | None" = None,
    ) -> list[dict]:
        """Топ кандидатов с расшифровкой, из чего сложилась оценка.

        `player` — профиль пользователя, приславшего запрос. Он передаётся
        аргументом, а не хранится в сервисе, потому что сервис один на всех:
        держать в нём чей-то профиль означало бы отдавать чужую статистику
        следующему посетителю.
        """
        top_k = top_k or self.top_k
        excluded = set(exclude or set()) | set(allies) | set(enemies)
        if role:
            all_ids = set(self.id_to_name)
            excluded |= all_ids - self.role_to_ids.get(role, set())

        stats = player.stats if player is not None else self.player_stats

        if self.level == "L0":
            result = self.recommender.recommend(exclude_ids=excluded, top_k=top_k)
        else:
            extra = {"is_radiant": is_radiant} if self.level == "L3" else {}
            if self.level == "L3" and player is not None:
                extra["delta"] = player.delta
            result = self.recommender.recommend(
                allies=allies, enemies=enemies, exclude_ids=excluded, top_k=top_k, **extra
            )

        out = []
        for row in result.itertuples():
            hid = int(row.hero_id)
            syn, ctr = self.pair_deltas(hid, allies, enemies)
            games, winrate = stats.get(hid, (0, 0.0))
            out.append(
                {
                    "hero_id": hid,
                    "name": self.id_to_name.get(hid, str(hid)),
                    "p_win": float(getattr(row, "p_win", float("nan")))
                    if hasattr(row, "p_win")
                    else None,
                    "player_bonus": float(getattr(row, "player_bonus", 0.0))
                    if hasattr(row, "player_bonus")
                    else None,
                    "score": float(getattr(row, "score", getattr(row, "winrate", 0.0))),
                    "synergy": syn,
                    "counter": ctr,
                    "my_games": games,
                    "my_winrate": winrate,
                }
            )
        return out

    def predict_team(
        self, team: list[int], enemies: list[int], is_radiant: bool = True
    ) -> float | None:
        """Вероятность победы команды.

        Каждый из героев команды по очереди берётся кандидатом: модель
        обучалась именно в такой постановке, и все оценки относятся к одной
        величине. Разброс между ними доходит до нескольких процентных
        пунктов, поэтому берётся среднее, а не одна оценка.
        """
        components = getattr(self.recommender, "score_components", None)
        if components is None or not team:
            return None
        probs = []
        for hero in team:
            others = [h for h in team if h != hero]
            p_win, _ = components([hero], others, enemies, is_radiant)
            probs.append(float(p_win[0]))
        return sum(probs) / len(probs)

    def threat_ranking(self, team: list[int], enemies: list[int]) -> list[dict]:
        return threat_ranking(self.recommender, self.id_to_name, team, enemies)

    def tactics_for(self, team: list[int], enemies: list[int]) -> list[str]:
        return tactics(
            self.role_counts(team), self.role_counts(enemies), len(team), len(enemies)
        )

    @property
    def max_allies(self) -> int:
        return MAX_ALLIES

    @property
    def max_enemies(self) -> int:
        return MAX_ENEMIES
