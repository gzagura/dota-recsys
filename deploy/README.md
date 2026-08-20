# Развёртывание на VPS

Проверялось на OVH VPS-2 (4 vCore, 8 ГБ, Ubuntu 26.04).

## Сколько это стоит по ресурсам

Замеры на локальной машине, в скобках — консервативная оценка для 4 ядер:

| Что | Сколько |
|---|---|
| RAM, модель загружена | 237 МБ (в простое), ~320 МБ под нагрузкой |
| Один `/api/recommend` | 7 мс |
| Пропускная способность | 105 req/s на 28 ядрах (~15–20 req/s на 4) |
| Артефакты на диске | ~600 КБ |
| Профиль игрока в кеше | ~8 КБ на человека |

Десять человек в драфте дают 2–3 запроса в секунду на пике. Запас
шестикратный, и на сервере остаётся место под другие проекты.

## Порядок действий

### 1. Пользователь и код

```bash
adduser --system --group --home /opt/dota-recsys draft
cd /opt && git clone https://github.com/<логин>/dota-recsys.git
chown -R draft:draft /opt/dota-recsys
```

### 2. Зависимости

```bash
apt install python3-venv nginx
sudo -u draft python3 -m venv /opt/dota-recsys/.venv
sudo -u draft /opt/dota-recsys/.venv/bin/pip install -r /opt/dota-recsys/requirements-serve.txt
```

`requirements-serve.txt`, а не `requirements.txt`: сбор данных и обучение
на сервере не нужны, а вместе с ними отпадают tqdm, pytest и scikit-learn.
requests остаётся — им сайт ходит в OpenDota за профилем игрока.

### 3. Артефакты и секреты

С локальной машины (в репозитории их нет):

```bash
./deploy/sync-artifacts.sh draft@54.38.203.246
```

Скрипт везёт модель, матрицы, справочник героев, `configs/config.yaml` и
`configs/users.json`. Учётки заводятся заранее локально:

```bash
python -m src.web.auth init друг1 друг2
```

### 4. Сервис

```bash
cp /opt/dota-recsys/deploy/dota-draft.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now dota-draft
systemctl status dota-draft
```

### 5. Домен и HTTPS

Сайт закрыт HTTP Basic, а он шлёт пароль в заголовке каждого запроса.
По голому HTTP пароль летит открытым текстом, поэтому сначала сертификат,
потом раздача доступов друзьям.

Let's Encrypt не выдаёт сертификаты на голый IP — нужен домен. Либо свой
(в панели OVH), либо бесплатный поддомен вроде duckdns.org, указывающий
на `54.38.203.246`.

```bash
cp /opt/dota-recsys/deploy/nginx.conf /etc/nginx/sites-available/dota-draft
# заменить ДОМЕН на свой
ln -s /etc/nginx/sites-available/dota-draft /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

apt install certbot python3-certbot-nginx
certbot --nginx -d <домен>        # сам допишет 443 и редирект с 80
```

### 6. Файрвол

```bash
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw enable
```

Порт 8000 наружу не открывается: uvicorn слушает только `127.0.0.1`, и
единственный вход снаружи — через nginx.

## Обновление

```bash
ssh draft@54.38.203.246 'cd /opt/dota-recsys && git pull'
sudo systemctl restart dota-draft
```

Переобучил модель — прогони `sync-artifacts.sh` заново и перезапусти
сервис. Кеш профилей в `data/profiles/` при этом можно не трогать: там
сырая статистика игроков, от версии модели она не зависит.
