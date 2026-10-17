-- Forward-looking hypotheses derived from historical mechanisms ("what the mechanisms suggest").
-- Registered with a falsification condition and a check date, so they can be scored later.

CREATE TABLE outlook_hypothesis (
  oh_id            TEXT PRIMARY KEY,
  statement        TEXT NOT NULL,
  mechanism_id     TEXT REFERENCES mechanism(mechanism_id),
  based_on_claims  TEXT NOT NULL DEFAULT '[]',   -- JSON list of claim ids
  falsified_if     TEXT NOT NULL,
  check_by_year    INTEGER NOT NULL,
  signpost_json    TEXT,                         -- machine-checkable rule, if operationalised
  operationalised  INTEGER NOT NULL DEFAULT 0 CHECK (operationalised IN (0, 1)),
  scope            TEXT,
  notes            TEXT,
  data_kind        TEXT NOT NULL DEFAULT 'real' CHECK (data_kind IN ('real','example','synthetic')),
  registered_utc   TEXT NOT NULL
);
