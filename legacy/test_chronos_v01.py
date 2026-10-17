import tempfile
import unittest
from pathlib import Path
import chronos

class ChronosTests(unittest.TestCase):
    def test_analog_no_future_leak(self):
        values = {year: 25.0 + (year-1900)*0.4 + 2.0*__import__('math').sin((year-1900)/9) for year in range(1900, 2026)}
        r = chronos.forecast(values, 2000, 10, 5, 7)
        self.assertIsNotNone(r)
        self.assertTrue(all(a['year'] <= 1990 for a in r['analogs']))
        self.assertEqual(r['target_year'], 2010)

    def test_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with chronos.connect(str(Path(tmp)/'db.sqlite')) as con:
                chronos.demo(con)
                data,source,kind = chronos.load_series(con,'DEMO_SYNTHETIC_INDEX','Synthetic World')
                self.assertEqual(kind,'synthetic')
                b = chronos.backtest(data,10,5,7)
                self.assertGreater(b['n'],10)
                r = chronos.forecast(data,max(data),10,5,7)
                self.assertLess(r['analog_scenarios']['low'],r['analog_scenarios']['high'])

    def test_source_kind_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with chronos.connect(str(Path(tmp)/'db.sqlite')) as con:
                chronos.store_observations(con,'x','world',[(2000,2)],'fake','synthetic')
                with self.assertRaises(ValueError):
                    chronos.store_observations(con,'x','world',[(2001,3)],'real','real')

if __name__=='__main__':
    unittest.main()
