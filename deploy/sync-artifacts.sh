#!/usr/bin/env bash
# Копирует на сервер то, чего нет в git: модель, матрицы, справочник героев,
# конфиг и файл пользователей.
#
# Почему артефакты не в репозитории: обучающий датасет весит 24 МБ и в
# публичном репозитории ему делать нечего, а держать половину артефактов
# в git, а половину рядом — путаница. Проще один скрипт.
#
#   ./deploy/sync-artifacts.sh draft@54.38.203.246
set -euo pipefail

TARGET="${1:?Укажи цель: ./deploy/sync-artifacts.sh draft@54.38.203.246}"
REMOTE_DIR="${2:-/opt/dota-recsys}"

# Инференсу нужны ровно эти файлы. l3_dataset.parquet (24 МБ) — только для
# обучения, на сервер он не едет.
ARTIFACTS=(
    data/processed/l3_lgbm.txt
    data/processed/l3_lgbm.meta.json
    data/processed/synergy_matrix.parquet
    data/processed/counter_matrix.parquet
    data/processed/hero_stats.parquet
    data/raw/heroes.json
)

SECRETS=(
    configs/config.yaml
    configs/users.json
)

for f in "${ARTIFACTS[@]}" "${SECRETS[@]}"; do
    [[ -f "$f" ]] || { echo "нет файла: $f" >&2; exit 1; }
done

echo "Создаю каталоги на $TARGET"
ssh "$TARGET" "mkdir -p $REMOTE_DIR/data/{processed,raw,profiles} $REMOTE_DIR/configs"

echo "Копирую артефакты модели"
for f in "${ARTIFACTS[@]}"; do
    scp "$f" "$TARGET:$REMOTE_DIR/$f"
done

echo "Копирую конфиг и пользователей"
for f in "${SECRETS[@]}"; do
    scp "$f" "$TARGET:$REMOTE_DIR/$f"
done
# Файл с хешами паролей не должен быть читаем всей машиной.
ssh "$TARGET" "chmod 600 $REMOTE_DIR/configs/users.json"

echo "Готово. Перезапусти сервис: ssh $TARGET 'sudo systemctl restart dota-draft'"
