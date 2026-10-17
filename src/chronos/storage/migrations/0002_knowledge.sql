-- Historical knowledge graph: events, references, mechanisms and claims
--   Event A --(claimed mechanism)--> Event B
-- Claims are hypotheses with an explicit evidence level. A temporal sequence is never
-- stored or upgraded as causality automatically.

CREATE TABLE reference (
  ref_id        TEXT PRIMARY KEY,
  ref_type      TEXT NOT NULL CHECK (ref_type IN
                ('journal_article','book','book_chapter','working_paper','dataset','report',
                 'web_page','encyclopedia','primary_source')),
  authors       TEXT,
  title         TEXT NOT NULL,
  year          INTEGER,
  venue         TEXT,
  publisher     TEXT,
  doi           TEXT,
  url           TEXT,
  accessed      TEXT,
  -- unverified: entered, not checked; metadata_checked: existence/bibliographic data checked
  -- against an external index; content_checked: a person confirmed the cited passage.
  verification  TEXT NOT NULL DEFAULT 'unverified'
                CHECK (verification IN ('unverified','metadata_checked','content_checked')),
  notes         TEXT
);

CREATE TABLE event (
  event_id          TEXT PRIMARY KEY,
  title             TEXT NOT NULL,
  description       TEXT NOT NULL DEFAULT '',
  category          TEXT NOT NULL CHECK (category IN
                    ('war','revolution','discovery','invention','technology_transition','epidemic',
                     'migration','political_reform','economic_crisis','institutional_change',
                     'regulation','demographic','environmental','cultural','other')),
  start_year        INTEGER NOT NULL,
  end_year          INTEGER,
  year_uncertainty  INTEGER NOT NULL DEFAULT 0 CHECK (year_uncertainty >= 0),
  date_precision    TEXT NOT NULL DEFAULT 'year'
                    CHECK (date_precision IN ('day','month','year','decade','century','approximate')),
  date_detail       TEXT,                     -- e.g. '1993-04-30'
  region            TEXT,
  tags              TEXT NOT NULL DEFAULT '[]',  -- JSON list
  data_kind         TEXT NOT NULL DEFAULT 'real' CHECK (data_kind IN ('real','example','synthetic')),
  created_utc       TEXT NOT NULL,
  CHECK (end_year IS NULL OR end_year >= start_year)
);

CREATE TABLE event_reference (
  event_id  TEXT NOT NULL REFERENCES event(event_id) ON DELETE CASCADE,
  ref_id    TEXT NOT NULL REFERENCES reference(ref_id),
  locator   TEXT,
  PRIMARY KEY (event_id, ref_id)
);

CREATE TABLE mechanism (
  mechanism_id  TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  description   TEXT NOT NULL,
  domain        TEXT NOT NULL CHECK (domain IN
                ('economic','social','political','demographic','technological','informational',
                 'ecological','military','cultural'))
);

-- evidence_level:
--   0 unassessed | 1 speculative | 2 plausible argument | 3 supported by qualitative scholarship
--   4 supported by quantitative association studies | 5 supported by identified causal designs
CREATE TABLE claim (
  claim_id         TEXT PRIMARY KEY,
  source_event_id  TEXT NOT NULL REFERENCES event(event_id),
  mechanism_id     TEXT REFERENCES mechanism(mechanism_id),
  target_event_id  TEXT NOT NULL REFERENCES event(event_id),
  relation         TEXT NOT NULL CHECK (relation IN
                   ('causal_hypothesis','contributing_condition','documented_claim','analogy','temporal_sequence')),
  statement        TEXT NOT NULL,
  evidence_level   INTEGER NOT NULL DEFAULT 0 CHECK (evidence_level BETWEEN 0 AND 5),
  status           TEXT NOT NULL DEFAULT 'proposed'
                   CHECK (status IN ('proposed','under_test','supported','contested','refuted')),
  lag_min_years    INTEGER,
  lag_max_years    INTEGER,
  curator          TEXT,
  notes            TEXT,
  data_kind        TEXT NOT NULL DEFAULT 'real' CHECK (data_kind IN ('real','example','synthetic')),
  created_utc      TEXT NOT NULL,
  CHECK (lag_min_years IS NULL OR lag_max_years IS NULL OR lag_max_years >= lag_min_years)
);

CREATE TABLE claim_reference (
  claim_id  TEXT NOT NULL REFERENCES claim(claim_id) ON DELETE CASCADE,
  ref_id    TEXT NOT NULL REFERENCES reference(ref_id),
  stance    TEXT NOT NULL CHECK (stance IN ('supports','contradicts','qualifies','discusses')),
  locator   TEXT,
  PRIMARY KEY (claim_id, ref_id, stance)
);

CREATE TABLE alternative_explanation (
  alt_id       INTEGER PRIMARY KEY,
  claim_id     TEXT NOT NULL REFERENCES claim(claim_id) ON DELETE CASCADE,
  description  TEXT NOT NULL,
  ref_id       TEXT REFERENCES reference(ref_id)
);

-- Documented quantitative statements that are not time series (e.g. a survey share).
CREATE TABLE fact (
  fact_id      TEXT PRIMARY KEY,
  subject      TEXT NOT NULL,
  statement    TEXT NOT NULL,
  value        REAL,
  unit         TEXT,
  year         INTEGER,
  ref_id       TEXT NOT NULL REFERENCES reference(ref_id),
  locator      TEXT,
  uncertainty  TEXT,
  tags         TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE claim_status_log (
  log_id      INTEGER PRIMARY KEY,
  claim_id    TEXT NOT NULL REFERENCES claim(claim_id) ON DELETE CASCADE,
  old_status  TEXT, new_status TEXT NOT NULL,
  old_level   INTEGER, new_level INTEGER NOT NULL,
  reason      TEXT NOT NULL,
  changed_utc TEXT NOT NULL
);
