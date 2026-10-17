# Источники данных: проверка API, лицензий и статус коннекторов

Проверено 2026-10-17. Среда разработки имела выход в интернет **только** к `raw.githubusercontent.com` и PyPI;
хосты `ourworldindata.org`, `api.worldbank.org`, `dataverse.nl`, `rug.nl`, `api.gdeltproject.org`,
`correlatesofwar.org`, `seshat-db.com`, `ucdpapi.pcr.uu.se` отвечали отказом прокси (403). Поэтому статус
«проверен вживую» есть только у источников, официально публикующих данные на GitHub.

| Источник | Доступ | Статус в CHRONOS | Лицензия (как заявлено издателем) | Что проверено |
|---|---|---|---|---|
| **Our World in Data — датасеты на GitHub** (`owid/co2-data`, `owid/energy-data`) | официальные CSV + codebook | **LIVE_VERIFIED** | CC BY 4.0 для данных OWID; сторонние данные — по лицензиям первоисточников (столбец `source` codebook) | загрузка, разбор, версии (ETag), хеш; ВВП = Maddison Project Database 2023 |
| **Seshat Equinox-2020** (`seshatdb/Equinox_Data`) | официальный XLSX на GitHub | **LIVE_VERIFIED** | CC BY-NC-SA 4.0 (некоммерческая, share-alike) | 13 листов; импутированные значения и «заполнители» помечаются статусом |
| **V-Dem v16** (`vdeminstitute/vdemdata`) | официальный RData на GitHub или CSV с v-dem.net | **LIVE_VERIFIED** | **не подтверждена в этой сборке** (R-пакет — GPL-3); проверьте условия на v-dem.net | 28 092 × 4 618; версия из DESCRIPTION («vdemdata 16.0»); ~4 ГБ ОЗУ для RData |
| **OWID Chart API** (`/grapher/<slug>.csv?v=1&csvType=full&useColumnShortNames=true` + `.metadata.json`) | API | IMPLEMENTED_UNVERIFIED | как у OWID | шаблон URL подтверждён по страницам графиков OWID (поиск); разбор метаданных защитный |
| **World Bank Indicators API v2** (`/v2/country/{c}/indicator/{i}?format=json`) | API | IMPLEMENTED_UNVERIFIED | большинство WDI — CC BY 4.0 (проверять по индикатору) | формат ответа `[метаданные, строки]` и формат ошибок; разбор протестирован на фикстурах, собранных по документации |
| **GDELT DOC 2.0** (`/api/v2/doc/doc?mode=timelinevol&format=json`) | API | IMPLEMENTED_UNVERIFIED | открытые данные, требуется ссылка (условия не проверены) | только современные новости; годовые средние; неполные годы помечаются `estimated` |
| **Maddison Project Database 2023** | файл (DataverseNL doi:10.34894/INZBF2) | FILE_IMPORT | CC BY 4.0 | разбор листа `countrycode,country,year,gdppc,pop`; на реальном файле **не проверен**; те же данные ВВП доступны через OWID |
| **Penn World Table 11.0** | файл (DataverseNL doi:10.34894/FABVLR) | FILE_IMPORT | CC BY 4.0 | лист `Data`; на реальном файле **не проверен** |
| **Correlates of War — NMC** | файл (correlatesofwar.org, есть v7.0) | FILE_IMPORT | свободно для исследований со ссылкой (условия не проверены) | раскладка v6.0, `-9` = пропуск; на реальном файле **не проверен** |
| **Seshat REST API** | API с учётной записью | NOT_IMPLEMENTED | — | существует Python-клиент `seshat_api` (MIT), требует логин/пароль |
| **UCDP** | API | NOT_IMPLEMENTED | — | условия доступа к API изменились; не реализовано |
| UN WPP, Clio-Infra, HYDE, Polity5, Epoch AI | — | NOT_IMPLEMENTED | — | кандидаты на следующие версии |

## Как проверить API у себя

```bash
chronos sources --check        # пробует каждый хост и печатает доступен/нет
CHRONOS_NETWORK_TESTS=1 pytest tests/integration/test_live_sources.py
```

## Происхождение каждого наблюдения

Каждый `fetch`/`import` создаёт **vintage**: источник, датасет, URL/путь, время получения (UTC), версию издателя
(ETag, `lastupdated`, номер релиза), SHA-256 исходного файла (файл сохраняется в `data/cache/<source>/`).
У ряда хранятся единицы, исходная переменная, описание и первоисточник из codebook; у каждого наблюдения —
статус (`observed`, `estimated`, `imputed`, `interpolated`, `projected`, `filler`, `derived`).
