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

Код принадлежит обычному пользователю (`ubuntu`), чтобы `git pull` и `scp`
работали без sudo. Сервис при этом бегает под отдельным `draft` без
пароля и без шелла: у `ubuntu` есть sudo, и запускать веб под ним значило
бы отдать sudo любому, кто пролезет через сайт.

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin draft
sudo mkdir -p /opt/dota-recsys && sudo chown ubuntu:ubuntu /opt/dota-recsys
git clone https://github.com/gzagura/dota-recsys.git /opt/dota-recsys
```

### 2. Зависимости

```bash
sudo apt install python3-venv nginx certbot python3-certbot-nginx ufw
cd /opt/dota-recsys && python3 -m venv .venv
.venv/bin/pip install -r requirements-serve.txt
```

`requirements-serve.txt`, а не `requirements.txt`: сбор данных и обучение
на сервере не нужны, а вместе с ними отпадают tqdm, pytest и scikit-learn.
requests остаётся — им сайт ходит в OpenDota за профилем игрока.

### 3. Артефакты и секреты

С локальной машины (в репозитории их нет):

```bash
./deploy/sync-artifacts.sh ubuntu@54.38.203.246
```

Скрипт везёт модель, матрицы, справочник героев, `configs/config.yaml` и
`configs/users.json`. Учётки заводятся заранее локально:

```bash
python -m src.web.auth init друг1 друг2
```

### 4. Права и сервис

Сервису нужно читать конфиг с ключами и писать только в кеш профилей:

```bash
sudo chown -R draft:draft /opt/dota-recsys/data/profiles
sudo chown draft:draft /opt/dota-recsys/configs/config.yaml /opt/dota-recsys/configs/users.json
sudo chmod 600 /opt/dota-recsys/configs/config.yaml /opt/dota-recsys/configs/users.json

sudo cp /opt/dota-recsys/deploy/dota-draft.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now dota-draft
sudo systemctl status dota-draft
```

### 5. Домен и HTTPS

Сайт закрыт HTTP Basic, а он шлёт пароль в заголовке каждого запроса.
По голому HTTP пароль летит открытым текстом, поэтому сначала сертификат,
потом раздача доступов друзьям.

Let's Encrypt не выдаёт сертификаты на голый IP — нужен домен. A-запись
поддомена должна указывать на `54.38.203.246`. Если домен живёт на
Cloudflare, проксирование (оранжевая тучка) надо **выключить**: в режиме
Flexible участок Cloudflare → сервер идёт по голому HTTP, и пароль от
Basic поедет открытым текстом через полинтернета.

```bash
sudo sed "s/ДОМЕН/dota.zagzags.com/" /opt/dota-recsys/deploy/nginx.conf \
    | sudo tee /etc/nginx/sites-available/dota-draft > /dev/null
sudo ln -sf /etc/nginx/sites-available/dota-draft /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

sudo certbot --nginx -d dota.zagzags.com --redirect   # допишет 443 и редирект
```

Дальше — HSTS в 443-й блок, который создал certbot:

```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

Зона `.com`, в отличие от `.dev`, не обязывает браузер ходить по HTTPS.
Заголовок закрывает эту дыру: браузер, увидевший его хоть раз, больше
никогда не пойдёт на адрес по голому HTTP — а именно там пароль от Basic
и утекал бы.

### 6. Файрвол

**Порядок важен.** Сначала правила, потом включение, и обязательно
проверить, что правило для ssh на месте, — `ufw enable` при пустом наборе
правил закрывает всё входящее, включая 22-й порт, и машина остаётся
доступна только через KVM-консоль в панели OVH.

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw show added                  # убедиться, что 22/tcp в списке
sudo ufw enable
```

Порт 8000 наружу не открывается: uvicorn слушает только `127.0.0.1`, и
единственный вход снаружи — через nginx.

## Обновление

```bash
ssh ubuntu@54.38.203.246 'cd /opt/dota-recsys && git pull && sudo systemctl restart dota-draft'
```

Переобучил модель — прогони `sync-artifacts.sh` заново и перезапусти
сервис. Кеш профилей в `data/profiles/` при этом можно не трогать: там
сырая статистика игроков, от версии модели она не зависит.
