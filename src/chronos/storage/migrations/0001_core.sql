-- Quantitative data with full provenance and data vintages.
-- A "vintage" is one retrieval/import of one source dataset. Observations are never
-- overwritten: a re-fetch that changes values creates a new vintage, so revisions of
-- historical statistics stay visible and real-time backtests become possible.

CREATE TABLE source (
  source_id   TEXT PRIMARY KEY,               -- e.g. 'owid', 'worldbank', 'seshat'
  name        TEXT NOT NULL,
  publisher   TEXT,
  homepage    TEXT,
  license     TEXT,                           -- as stated by the publisher, or 'UNVERIFIED: ...'
  citation    TEXT,
  terms_url   TEXT
);

CREATE TABLE series (
  series_id        INTEGER PRIMARY KEY,
  indicator        TEXT NOT NULL,             -- local code, e.g. 'population'
  entity           TEXT NOT NULL,             -- e.g. 'World', 'Sweden', 'Latium'
  entity_code      TEXT,                      -- ISO3 / NGA code where available
  source_id        TEXT NOT NULL REFERENCES source(source_id),
  source_dataset   TEXT NOT NULL,             -- e.g. 'owid/co2-data'
  source_variable  TEXT NOT NULL,             -- original column / indicator code
  title            TEXT,
  description      TEXT,
  unit             TEXT,
  frequency        TEXT NOT NULL DEFAULT 'annual' CHECK (frequency IN ('annual','decadal','century','irregular')),
  data_kind        TEXT NOT NULL CHECK (data_kind IN ('real','synthetic')),
  created_utc      TEXT NOT NULL,
  UNIQUE (indicator, entity, source_id)
);
CREATE INDEX idx_series_indicator ON series(indicator, entity);

CREATE TABLE vintage (
  vintage_id      INTEGER PRIMARY KEY,
  source_id       TEXT NOT NULL REFERENCES source(source_id),
  source_dataset  TEXT NOT NULL,
  retrieved_utc   TEXT NOT NULL,              -- when CHRONOS obtained the data
  source_version  TEXT,                       -- publisher version/date if known
  as_of_date      TEXT,                       -- for archival vintages: date the data reflected knowledge
  location        TEXT,                       -- URL or local file path
  content_sha256  TEXT,                       -- hash of the raw payload
  notes           TEXT
);

-- Which vintages contain which series. Values are stored once per distinct content:
-- rows_vintage_id points at the vintage under which the identical values were stored.
CREATE TABLE series_vintage (
  series_id        INTEGER NOT NULL REFERENCES series(series_id),
  vintage_id       INTEGER NOT NULL REFERENCES vintage(vintage_id),
  rows_vintage_id  INTEGER NOT NULL REFERENCES vintage(vintage_id),
  values_sha256    TEXT NOT NULL,
  n_obs            INTEGER NOT NULL,
  first_year       INTEGER,
  last_year        INTEGER,
  PRIMARY KEY (series_id, vintage_id)
);

CREATE TABLE observation (
  series_id   INTEGER NOT NULL REFERENCES series(series_id),
  vintage_id  INTEGER NOT NULL REFERENCES vintage(vintage_id),
  year        INTEGER NOT NULL,               -- negative = BCE (astronomical year numbering not applied)
  value       REAL NOT NULL,
  status      TEXT NOT NULL DEFAULT 'observed'
              CHECK (status IN ('observed','estimated','imputed','interpolated','projected','filler','derived')),
  note        TEXT,
  PRIMARY KEY (series_id, vintage_id, year)
);
CREATE INDEX idx_obs_series ON observation(series_id, vintage_id);
