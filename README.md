# CHRONOS 0.2 — Historical Intelligence & Civilization Forecasting Engine

Локальная исследовательская система для изучения исторической динамики и построения **условных** сценариев.
Это не машина предсказаний: каждое число прослеживается до версии источника, каждый прогноз сравнивается с
простыми baseline-моделями на данных, которых модель не видела, у каждого причинного утверждения есть явный
уровень доказательности, а сценариям не присваиваются вероятности.

* Что сделано и чего нет — **[REPORT.md](REPORT.md)** (отчёт о реализации и ограничениях).
* Аудит исходной v0.1 — [docs/AUDIT_v0.1.md](docs/AUDIT_v0.1.md); архитектура — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md);
  источники, API и лицензии — [docs/SOURCES.md](docs/SOURCES.md); результаты на реальных данных — [docs/results/](docs/results/).

## 1. Установка

Нужен Python **3.11+**.

```bash
git clone <repo> && cd CHRONOS
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[all]"              # ядро + Streamlit/plotly + pyreadr (V-Dem) + pytest/ruff/mypy
chronos --help                       # или: python -m chronos --help
```

Минимальная установка без интерфейса: `pip install -e .` (numpy, pandas, scipy, statsmodels, requests, openpyxl).

## 2. Проверка без интернета

```bash
pytest                                # 92 теста; 5 сетевых пропускаются
chronos --db demo.sqlite demo         # СИНТЕТИЧЕСКИЙ ряд (помечен как synthetic везде)
chronos --db demo.sqlite backtest --indicator DEMO_SYNTHETIC_INDEX --entity "Synthetic World"
```

## 3. Загрузка реальных данных

Одной командой (≈ 1–2 мин; V-Dem требует ~4 ГБ ОЗУ, пропустить: `CHRONOS_SKIP_VDEM=1`):

```bash
scripts/bootstrap_data.sh chronos.sqlite
```

Или по шагам:

```bash
chronos init
chronos fetch owid-github --dataset co2        # население, ВВП (Maddison 2023), CO2, ПГ, потепление (OWID)
chronos fetch owid-github --dataset energy     # энергия и её структура (OWID / Energy Institute / EIA)
chronos fetch seshat                           # Seshat Equinox-2020: социальная сложность, 35 регионов, по векам
chronos fetch vdem                             # V-Dem v16: институты + внешние переменные (продолжительность жизни и др.)
chronos kg import data/curated/tech_transitions.json   # курированный граф знаний (печать / интернет / ген. ИИ)

# API-коннекторы (реализованы по документации; в среде разработки хосты были недоступны):
chronos fetch worldbank --indicator IT.NET.USER.ZS --country WLD --date 1960:2025
chronos fetch owid-chart --slug life-expectancy --entity World --indicator life_expectancy
chronos fetch gdelt --query '"artificial intelligence"' --start 20250101000000 --end 20251001000000

# официальные файлы, скачанные вручную:
chronos import maddison --file mpd2023_web.xlsx
chronos import pwt --file pwt110.xlsx
chronos import cow-nmc --file NMC-60-abridged.csv
chronos import-csv --file my.csv --indicator x --entity World --source-url https://… [--synthetic]

chronos sources            # статус каждого коннектора и лицензии
chronos sources --check    # какие хосты доступны прямо сейчас
chronos series --summary   # что загружено
chronos show --indicator population --entity World      # ряд + происхождение
chronos revisions --indicator population --entity World # пересмотры между версиями
```

Повторный `fetch` создаёт новую версию данных (vintage) только если значения изменились; старые значения не
перезаписываются — так видны пересмотры исторической статистики.

## 4. Анализ

```bash
# Walk-forward проверка: каждая модель видит только данные до точки прогноза; сравнение с baseline,
# MAE/RMSE/MASE/MAPE, калибровка интервалов, тест Диболда–Мариано, устойчивость по подпериодам
chronos backtest --indicator gdp_per_capita --entity "United States" --source owid --horizons 1,5,10,20 --record

# «Данные, известные к 1950 г. → прогноз на 1970 г.» (одна точка — иллюстрация, не доказательство)
chronos experiment --indicator gdp_per_capita --entity "United States" --source owid --origin 1950 --target 1970

# Проверка по всем странам сразу (много независимых окон)
chronos panel-backtest --indicator gdp_per_capita --source owid --horizons 5,10,20 [--pooled]

# Режимы и структурные сдвиги; похожие траектории
chronos breaks --indicator gdp_per_capita --entity Japan --source owid --model meanvar
chronos breaks --indicator seshat_spc --entity "Middle Yellow River Valley" --signal level --min-size 5
chronos similar --indicator gdp_per_capita --source owid --entity Japan --start 1955 --end 1975

# Многомерные исторические аналоги (экономика, демография, энергия, институты, позиция к лидеру)
chronos analogs --entity China --year 2022 --horizon 10
chronos analogs --backtest            # предсказывают ли аналоги лучше baseline?

# Лестница причинной доказательности + реестр гипотез (предрегистрация, поправка BH)
chronos causal --x v2x_polyarchy:vdem:level --y gdp_per_capita:owid:growth \
  --controls population:owid:growth --treatment "v2x_regime:vdem:level>=2" \
  --event-outcome gdp_per_capita:owid:log --min-year 1950 --max-year 2022 \
  --expected-sign 1 --register H1 --statement "Democratisation raises GDP per capita"
chronos hypotheses
chronos experiments [--show ID]       # воспроизводимые записи: конфиг, коммит, отпечаток данных, хеш результата
```

## 5. Состояние мира, сценарии, исследовательский запрос

```bash
chronos state                          # 8 доменов; CURRENT / LAGGED / STALE / NO DATA, nowcast помечен как модельный
chronos scenarios --record             # 4 условных сценария 2026–2126 (scenarios/world_2026.toml)
chronos research transitions --year 2040 --out reports/dossier.md
```

## 6. Интерфейс

```bash
chronos ui                      # http://localhost:8501
# или: streamlit run src/chronos/ui/app.py -- --db chronos.sqlite
```

Разделы: источники и версии данных · временные ряды · граф знаний · аналоги · backtesting · паттерны ·
причинные гипотезы · состояние цивилизации · сценарии · исследовательский запрос.

## 7. Совместимость с v0.1

Все команды v0.1 работают с теми же флагами: `init`, `demo`, `fetch --slug … --entity …`, `import-csv`,
`analyze --indicator … --entity … --horizon 10 --backtest --json-out report.json`, `import-events`,
`import-claims`, `events`. Базы v0.1 мигрируют автоматически (старые таблицы сохраняются как `v01_*`).
Исходный код v0.1 лежит в `legacy/`; тест `tests/unit/test_v01_regression.py` проверяет, что новая модель
аналогов воспроизводит v0.1 точно.

## 8. Конфигурация

`config/chronos.toml` → переменные окружения → флаги CLI. Переменные: `CHRONOS_DB`, `CHRONOS_CACHE_DIR`,
`CHRONOS_HTTP_TIMEOUT`, `CHRONOS_HTTP_RETRIES`, `CHRONOS_LOG_LEVEL`, `CHRONOS_LOG_FORMAT` (`text`/`json`),
`CHRONOS_REFERENCE_YEAR`, `CHRONOS_CONFIG`. Секреты не нужны ни одному коннектору и в репозитории не хранятся.
Логи — структурированные, в stderr; `--log-format json` для машинной обработки.

## 9. Разработка

```bash
pytest                                              # офлайн-тесты
CHRONOS_NETWORK_TESTS=1 pytest tests/integration/test_live_sources.py
ruff check src tests && mypy
```

## Лицензии данных

Код — MIT. Данные подчиняются лицензиям источников (см. `chronos sources` и docs/SOURCES.md):
Seshat — **CC BY-NC-SA 4.0 (некоммерческое использование)**; OWID — CC BY 4.0 для собственных данных;
условия V-Dem проверьте на v-dem.net.
