#!/usr/bin/env python3
"""CHRONOS MVP: traceable historical observations, analog scenarios and honest backtesting.

Python 3.10+; standard library only. No LLM and no claims of automatic causality.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
  indicator TEXT NOT NULL,
  entity TEXT NOT NULL,
  year INTEGER NOT NULL,
  value REAL NOT NULL CHECK(value > 0),
  source_url TEXT NOT NULL,
  source_kind TEXT NOT NULL CHECK(source_kind IN ('real','synthetic')),
  imported_utc TEXT NOT NULL,
  PRIMARY KEY (indicator, entity, year)
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  year INTEGER NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  evidence_url TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
  cause_event_id TEXT NOT NULL REFERENCES events(event_id),
  effect_event_id TEXT NOT NULL REFERENCES events(event_id),
  mechanism TEXT NOT NULL,
  claim_type TEXT NOT NULL CHECK (claim_type IN ('hypothesis','analogy','documented')),
  evidence_url TEXT NOT NULL,
  PRIMARY KEY (cause_event_id,effect_event_id,mechanism)
);
"""


def connect(path: str) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target))
    con.execute('PRAGMA foreign_keys = ON')
    con.executescript(SCHEMA)
    return con


def parse_row_value(raw: str | None) -> float | None:
    try:
        value = float(raw) if raw is not None and str(raw).strip() else None
        return value if value is not None and value > 0 and math.isfinite(value) else None
    except (ValueError, OverflowError):
        return None


def store_observations(con, indicator, entity, pairs, source_url, source_kind='real') -> int:
    now = datetime.now(timezone.utc).isoformat()
    records = [(indicator, entity, int(year), float(value), source_url, source_kind, now)
               for year, value in pairs if parse_row_value(str(value)) is not None]
    if not records:
        raise ValueError('No positive numeric annual observations found.')
    # Prevent accidental confusion between synthetic and real observations.
    existing = con.execute('SELECT DISTINCT source_kind FROM observations WHERE indicator=? AND entity=?',
                           (indicator, entity)).fetchall()
    if existing and any(row[0] != source_kind for row in existing):
        raise ValueError('Refusing to mix REAL and SYNTHETIC data in one indicator/entity series.')
    con.executemany('''INSERT INTO observations VALUES (?,?,?,?,?,?,?)
                      ON CONFLICT(indicator,entity,year) DO UPDATE SET
                      value=excluded.value,source_url=excluded.source_url,
                      source_kind=excluded.source_kind,imported_utc=excluded.imported_utc''', records)
    con.commit()
    return len(records)


def fetch_owid(con, slug: str, entity: str, column: str | None, indicator: str | None) -> None:
    if not slug or '/' in slug or '..' in slug:
        raise ValueError('Invalid OWID chart slug.')
    url = f'https://ourworldindata.org/grapher/{quote(slug)}.csv'
    req = Request(url, headers={'User-Agent': 'CHRONOS-MVP/0.1 (historical-research)'})
    try:
        with urlopen(req, timeout=40) as response:
            body = response.read().decode('utf-8-sig')
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f'Could not download OWID chart ({url}): {exc}') from exc
    reader = csv.DictReader(io.StringIO(body))
    cols = reader.fieldnames or []
    if not all(k in cols for k in ('Entity', 'Year')):
        raise ValueError('Chart must have annual Entity and Year columns.')
    data_cols = [c for c in cols if c not in ('Entity', 'Code', 'Year')]
    if column and column not in data_cols:
        raise ValueError(f'Column not found. Available data columns: {data_cols}')
    if not data_cols:
        raise ValueError('No numeric columns in the downloaded chart.')
    chosen = column or data_cols[0]
    rows = []
    for record in reader:
        if record['Entity'] != entity:
            continue
        try:
            year = int(record['Year'])
        except (ValueError, TypeError):
            continue
        v = parse_row_value(record.get(chosen))
        if v is not None:
            rows.append((year, v))
    if not rows:
        raise ValueError(f'No annual records for {entity!r} in {slug!r}; inspect entity and column.')
    key = indicator or slug
    total = store_observations(con, key, entity, rows, url, 'real')
    print(f'Imported {total} observations: {key} / {entity} ({min(y for y,_ in rows)}–{max(y for y,_ in rows)})')
    print('Data column:', chosen)
    print('Source:', url)


def import_csv(con, filepath, indicator, entity, source_url, source_kind):
    with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'Year','Value'}.issubset(reader.fieldnames or []):
            raise ValueError('Required CSV columns: Year,Value')
        records = []
        for row in reader:
            try:
                year = int(row['Year'])
            except (TypeError, ValueError):
                continue
            val = parse_row_value(row.get('Value'))
            if val is not None:
                records.append((year, val))
    count = store_observations(con, indicator, entity, records, source_url, source_kind)
    print(f'Imported {count} rows for {indicator} / {entity} ({source_kind}).')


def load_series(con, indicator, entity):
    rows = con.execute('SELECT year,value,source_url,source_kind FROM observations WHERE indicator=? AND entity=? ORDER BY year',
                       (indicator, entity)).fetchall()
    if not rows:
        raise ValueError(f'No series: {indicator} / {entity}. Run fetch, import-csv, or demo first.')
    return {int(year): float(value) for year, value, _, _ in rows}, rows[-1][2], rows[-1][3]


def valid_window(values: dict[int,float], start: int, end: int) -> bool:
    return all(y in values and values[y] > 0 for y in range(start, end+1))


def percentile(xs, fraction):
    ordered = sorted(xs)
    if not ordered:
        raise ValueError('Empty percentile input.')
    pos = (len(ordered)-1) * fraction
    i = int(pos)
    rem = pos - i
    return ordered[i]*(1-rem) + ordered[min(i+1, len(ordered)-1)]*rem


def forecast(series: dict[int,float], origin: int, horizon: int, lookback: int, neighbors: int):
    """Analog futures use only outcomes fully known BY origin (no future leakage)."""
    if horizon < 1 or lookback < 3 or neighbors < 2:
        raise ValueError('Need horizon>=1, lookback>=3, neighbors>=2.')
    if not valid_window(series, origin-lookback+1, origin):
        return None
    current_log_returns = [math.log(series[y] / series[y-1])
                           for y in range(origin-lookback+2, origin+1)]
    scored = []
    for historical_end in sorted(series):
        if historical_end > origin-horizon or historical_end >= origin:
            continue
        if not (valid_window(series, historical_end-lookback+1, historical_end) and
                valid_window(series, historical_end, historical_end+horizon)):
            continue
        old_returns = [math.log(series[y] / series[y-1])
                       for y in range(historical_end-lookback+2, historical_end+1)]
        pattern_gap = math.sqrt(sum((a-b)**2 for a,b in zip(current_log_returns,old_returns)) / len(old_returns))
        # Endpoint level mismatch is a SMALL penalty, otherwise older epochs would never match.
        level_gap = abs(math.log(series[origin] / series[historical_end]))
        distance = pattern_gap + 0.007 * level_gap
        outcome = math.log(series[historical_end+horizon] / series[historical_end])
        scored.append((distance, historical_end, outcome))
    if len(scored) < neighbors:
        return None
    selected = sorted(scored, key=lambda x: (x[0],x[1]))[:neighbors]
    outcomes = [x[2] for x in selected]
    last = series[origin]
    future = {
        'low': last*math.exp(percentile(outcomes, 0.1)),
        'middle': last*math.exp(percentile(outcomes, 0.5)),
        'high': last*math.exp(percentile(outcomes, 0.9)),
    }
    baseline = last * math.exp(statistics.mean(current_log_returns)*horizon)
    return {
        'origin': origin, 'target_year': origin+horizon, 'last_value': last,
        'analog_scenarios': {k:round(v,4) for k,v in future.items()},
        'linear_log_trend_baseline': round(baseline,4),
        'analogs': [{'year':t, 'distance':round(d,6),
                     'observed_change_pct':round(100*(math.exp(ret)-1),2)}
                    for d,t,ret in selected],
        'n_candidates':len(scored),
    }


def backtest(series, horizon, lookback, neighbors, min_history=25, step=1):
    years = sorted(series)
    if not years:
        return {'n':0, 'note':'Empty series'}
    actuals = []
    for origin in range(years[0]+min_history, years[-1]-horizon+1, step):
        if origin+horizon not in series:
            continue
        result = forecast(series, origin, horizon, lookback, neighbors)
        if result is None:
            continue
        truth = series[origin+horizon]
        p = result['analog_scenarios']['middle']
        b = result['linear_log_trend_baseline']
        actuals.append((origin, truth, p, b,
                        result['analog_scenarios']['low'],result['analog_scenarios']['high']))
    if not actuals:
        return {'n':0, 'note':'Not enough uninterrupted observations for historical backtesting.'}
    mae_model = statistics.mean(abs(true-pred) for _,true,pred,*_ in actuals)
    mae_baseline = statistics.mean(abs(true-base) for _,true,_,base,*_ in actuals)
    mape_model = statistics.mean(abs(true-pred)/true for _,true,pred,*_ in actuals) * 100
    mape_baseline = statistics.mean(abs(true-base)/true for _,true,_,base,*_ in actuals) * 100
    coverage = statistics.mean(low<=true<=high for _,true,_,_,low,high in actuals)
    return {'n':len(actuals), 'first_origin':actuals[0][0], 'last_origin':actuals[-1][0],
            'analog_MAE':round(mae_model,4), 'baseline_MAE':round(mae_baseline,4),
            'analog_MAPE_percent':round(mape_model,3), 'baseline_MAPE_percent':round(mape_baseline,3),
            'historical_interval_coverage_percent':round(coverage*100,1),
            'analogs_better_than_baseline':mae_model<mae_baseline}


def demo(con):
    """Explicitly SYNTHETIC data, to test the code without any external requests."""
    rng = random.Random(104)
    val=30.0
    rows=[]
    for year in range(1900,2026):
        growth = (0.013 if year<1950 else 0.024 if year<1980 else 0.007 if year<2000 else 0.003)
        val *= math.exp(growth + rng.gauss(0,0.007))
        rows.append((year,val))
    n=store_observations(con,'DEMO_SYNTHETIC_INDEX','Synthetic World',rows,
                         'SYNTHETIC:generated-by-chronos-demo','synthetic')
    print(f'Created {n} synthetic observations (not historical facts, not a real-world forecast).')


def import_events(con, path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        expected={'event_id','year','title','description','evidence_url'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError('Events CSV requires: '+','.join(sorted(expected)))
        rows=[(r['event_id'],int(r['year']),r['title'],r['description'],r['evidence_url']) for r in reader]
    con.executemany('INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)',rows)
    con.commit()
    print(f'Imported {len(rows)} events.')


def import_claims(con,path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        expected={'cause_event_id','effect_event_id','mechanism','claim_type','evidence_url'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError('Claims CSV requires: '+','.join(sorted(expected)))
        rows=[]
        for r in reader:
            if r['claim_type']=='documented' and not r['evidence_url'].strip():
                raise ValueError('Documented claims must include an evidence URL.')
            rows.append(tuple(r[k] for k in ['cause_event_id','effect_event_id','mechanism','claim_type','evidence_url']))
    con.executemany('INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?)',rows)
    con.commit()
    print(f'Imported {len(rows)} labeled links. A documentary URL is not proof of causality.')


def show_events(con):
    for row in con.execute('SELECT event_id,year,title,evidence_url FROM events ORDER BY year'):
        print(f'{row[1]:>6} {row[0]} | {row[2]} | {row[3]}')
    print('\nConnections (claims/analogies, NOT model-proven causes):')
    for row in con.execute('SELECT cause_event_id,effect_event_id,claim_type,mechanism,evidence_url FROM claims'):
        print(f'  {row[0]} -> {row[1]} | {row[2]} | {row[3]} | evidence={row[4] or "none"}')


def main(argv=None):
    parser=argparse.ArgumentParser(description='CHRONOS MVP: historical observations, analog scenarios, backtesting, source links.')
    parser.add_argument('--db',default='chronos.sqlite',help='SQLite database file (default: chronos.sqlite)')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('init')
    commands.add_parser('demo',help='Load a clearly labeled synthetic dataset for offline testing')
    f=commands.add_parser('fetch',help='Download real chart observations from OWID')
    f.add_argument('--slug',default='life-expectancy')
    f.add_argument('--entity',default='World')
    f.add_argument('--column',help='CSV data column; default first data column')
    f.add_argument('--indicator',help='Local indicator name; default chart slug')
    imp=commands.add_parser('import-csv',help='Import annual observations from Year,Value CSV')
    imp.add_argument('--file',required=True)
    imp.add_argument('--indicator',required=True)
    imp.add_argument('--entity',required=True)
    imp.add_argument('--source-url',required=True)
    imp.add_argument('--synthetic',action='store_true',help='Flag CSV as synthetic (never hide synthetic origin)')
    a=commands.add_parser('analyze',help='Time-series analog scenarios and rolling historical evaluation')
    a.add_argument('--indicator',required=True)
    a.add_argument('--entity',required=True)
    a.add_argument('--horizon',type=int,default=10)
    a.add_argument('--lookback',type=int,default=5)
    a.add_argument('--neighbors',type=int,default=7)
    a.add_argument('--backtest',action='store_true')
    a.add_argument('--json-out',help='Write results to JSON')
    ev=commands.add_parser('import-events')
    ev.add_argument('--file',required=True)
    cl=commands.add_parser('import-claims')
    cl.add_argument('--file',required=True)
    commands.add_parser('events')
    args=parser.parse_args(argv)
    try:
        with connect(args.db) as con:
            if args.command=='init':
                print('Initialized:',args.db)
            elif args.command=='demo':
                demo(con)
            elif args.command=='fetch':
                fetch_owid(con,args.slug,args.entity,args.column,args.indicator)
            elif args.command=='import-csv':
                import_csv(con,args.file,args.indicator,args.entity,args.source_url,
                           'synthetic' if args.synthetic else 'real')
            elif args.command=='import-events':
                import_events(con,args.file)
            elif args.command=='import-claims':
                import_claims(con,args.file)
            elif args.command=='events':
                show_events(con)
            elif args.command=='analyze':
                series,source,kind=load_series(con,args.indicator,args.entity)
                origin=max(series)
                result=forecast(series,origin,args.horizon,args.lookback,args.neighbors)
                if result is None:
                    raise ValueError('Not enough contiguous historic data to find analogs; change horizon/lookback/neighbors.')
                report={'version':'0.1','indicator':args.indicator,'entity':args.entity,
                        'source_url':source,'data_kind':kind,'n_observations':len(series),
                        'latest_observed_year':origin,'forecast':result,
                        'method_notice': 'Historical analog percentiles are NOT calibrated outcome probabilities and do NOT prove causation.'}
                if args.backtest:
                    report['backtest']=backtest(series,args.horizon,args.lookback,args.neighbors)
                print(json.dumps(report,ensure_ascii=False,indent=2))
                if args.json_out:
                    Path(args.json_out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                    print('Saved report:',args.json_out)
    except (ValueError,RuntimeError,sqlite3.Error,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
