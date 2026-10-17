-- Reproducible experiments and a pre-registration style hypothesis registry.

CREATE TABLE experiment (
  experiment_id     TEXT PRIMARY KEY,         -- sha256 of (kind, config, data fingerprint)[:16] + run suffix
  kind              TEXT NOT NULL,            -- 'backtest', 'panel_backtest', 'analog_backtest', 'causal', ...
  created_utc       TEXT NOT NULL,
  chronos_version   TEXT NOT NULL,
  git_commit        TEXT,
  config_json       TEXT NOT NULL,
  data_fingerprint  TEXT NOT NULL,            -- JSON: series_id -> rows_vintage_id/values hash
  seed              INTEGER,
  results_json      TEXT NOT NULL,
  results_sha256    TEXT NOT NULL,
  notes             TEXT
);

-- Hypotheses must be registered (with their specification) before they are tested.
-- Every test run is logged, so the number of tests per hypothesis is visible and
-- multiple-testing corrections can be applied across the registry.
CREATE TABLE hypothesis (
  hyp_id          TEXT PRIMARY KEY,
  statement       TEXT NOT NULL,
  kind            TEXT NOT NULL CHECK (kind IN ('variable_relation','event_claim')),
  claim_id        TEXT REFERENCES claim(claim_id),
  spec_json       TEXT NOT NULL,              -- driver, outcome, expected sign, lags, controls, design
  registered_utc  TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'registered'
                  CHECK (status IN ('registered','tested','retired'))
);

CREATE TABLE hypothesis_test (
  test_id        INTEGER PRIMARY KEY,
  hyp_id         TEXT NOT NULL REFERENCES hypothesis(hyp_id),
  method         TEXT NOT NULL,
  p_value        REAL,
  estimate       REAL,
  evidence_tier  TEXT NOT NULL,
  result_json    TEXT NOT NULL,
  experiment_id  TEXT REFERENCES experiment(experiment_id),
  run_utc        TEXT NOT NULL
);
