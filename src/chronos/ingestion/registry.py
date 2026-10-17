"""Catalogue of data sources and the honest status of each connector.

Status vocabulary:
  LIVE_VERIFIED       connector run end-to-end against the real source during development
  IMPLEMENTED_UNVERIFIED  written against the official documentation, but the host was not
                      reachable from the development sandbox; run `chronos sources check`
  FILE_IMPORT         parser for an official downloadable file; the user supplies the file
  NOT_IMPLEMENTED     catalogued only
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from chronos.storage.repository import SourceInfo


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    name: str
    publisher: str
    homepage: str
    access: str          # 'api' | 'official_file' | 'none'
    status: str
    license: str
    citation: str
    terms_url: str | None
    command: str | None  # CLI command that uses it
    notes: str

    def info(self) -> SourceInfo:
        return SourceInfo(self.source_id, self.name, self.publisher, self.homepage, self.license,
                          self.citation, self.terms_url)

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


SOURCES: dict[str, SourceEntry] = {
    e.source_id: e
    for e in [
        SourceEntry(
            "owid", "Our World in Data", "Global Change Data Lab", "https://ourworldindata.org",
            "official_file", "LIVE_VERIFIED",
            "CC BY 4.0 for OWID-produced data; third-party data keep their original licenses (see codebook 'source')",
            "Our World in Data (dataset as published on GitHub: owid/co2-data, owid/energy-data)",
            "https://ourworldindata.org/faqs", "chronos fetch owid-github --dataset co2|energy",
            "Official OWID GitHub data repositories (raw.githubusercontent.com). Codebook gives title, unit and "
            "original source of every column; GDP is the Maddison Project Database 2023 (Bolt & van Zanden).",
        ),
        SourceEntry(
            "owid-grapher", "Our World in Data – Chart API", "Global Change Data Lab",
            "https://docs.owid.io/projects/etl/api/chart-api/", "api", "IMPLEMENTED_UNVERIFIED",
            "CC BY 4.0 for OWID-produced data; third-party data keep their original licenses",
            "Our World in Data chart data (per chart citation in metadata.json)",
            "https://ourworldindata.org/faqs", "chronos fetch owid-chart --slug <slug>",
            "URL pattern /grapher/<slug>.csv?v=1&csvType=full&useColumnShortNames=true and .metadata.json "
            "(pattern confirmed from OWID chart pages). Host blocked in the development sandbox.",
        ),
        SourceEntry(
            "worldbank", "World Bank Indicators API (v2)", "The World Bank", "https://data.worldbank.org",
            "api", "IMPLEMENTED_UNVERIFIED",
            "Most WDI data: CC BY 4.0 (check the indicator's license in the WDI metadata)",
            "World Bank, World Development Indicators", "https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets",
            "chronos fetch worldbank --indicator SP.POP.TOTL --country WLD",
            "api.worldbank.org/v2/country/{c}/indicator/{i}?format=json; response is [page-meta, rows]. "
            "Host blocked in the development sandbox.",
        ),
        SourceEntry(
            "seshat", "Seshat Global History Databank – Equinox-2020 release", "Seshat Global History Databank",
            "https://seshatdatabank.info", "official_file", "LIVE_VERIFIED",
            "CC BY-NC-SA 4.0 (non-commercial, share-alike) as stated in the Equinox_Data repository",
            "Turchin P. et al. (2020) Equinox-2020 Seshat data release; doi:10.5281/zenodo.6642229",
            "https://seshat-db.com/downloads_page", "chronos fetch seshat",
            "File Equinox_on_GitHub_June9_2022.xlsx from github.com/seshatdb/Equinox_Data. Century resolution, "
            "NGA level. ImpSCDat contains single-imputed values (Seshat: use only as predictors); rows with "
            "uniq=n are fillers, not new data. Newer Polaris releases exist; the Seshat REST API requires an "
            "account and is NOT IMPLEMENTED here.",
        ),
        SourceEntry(
            "vdem", "Varieties of Democracy (V-Dem) Country-Year v16", "V-Dem Institute, University of Gothenburg",
            "https://www.v-dem.net", "official_file", "LIVE_VERIFIED",
            "UNVERIFIED in this build: check v-dem.net terms (the vdemdata R package is GPL-3)",
            "Coppedge et al. (2026) V-Dem Country-Year Dataset v16, https://doi.org/10.23696/vdemds26",
            "https://www.v-dem.net/data/the-v-dem-dataset/", "chronos fetch vdem",
            "vdem.RData from the official github.com/vdeminstitute/vdemdata repository (~34 MB; ~4 GB RAM to "
            "load) or an official V-Dem CSV supplied with --file. External e_* variables come from third parties.",
        ),
        SourceEntry(
            "maddison", "Maddison Project Database 2023", "Groningen Growth and Development Centre",
            "https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023",
            "official_file", "FILE_IMPORT", "CC BY 4.0",
            "Bolt J. & van Zanden J.L. (2024) Maddison style estimates of the evolution of the world economy: "
            "a new 2023 update. Journal of Economic Surveys, doi:10.1111/joes.12618",
            None, "chronos import maddison --file mpd2023_web.xlsx",
            "Excel/Stata on DataverseNL doi:10.34894/INZBF2 (host blocked in sandbox). Parser expects a sheet "
            "with countrycode, country, year, gdppc, pop; NOT verified against the real file here. Maddison GDP "
            "is also available through the 'owid' source (column gdp).",
        ),
        SourceEntry(
            "pwt", "Penn World Table 11.0", "Groningen Growth and Development Centre",
            "https://www.rug.nl/ggdc/productivity/pwt/", "official_file", "FILE_IMPORT", "CC BY 4.0",
            "Feenstra R.C., Inklaar R. & Timmer M.P. (2015) The Next Generation of the Penn World Table. "
            "American Economic Review 105(10)", None, "chronos import pwt --file pwt110.xlsx",
            "DataverseNL doi:10.34894/FABVLR. Parser expects a 'Data' sheet with countrycode, country, year "
            "and variables such as rgdpe, pop, hc, ctfp; NOT verified against the real file here.",
        ),
        SourceEntry(
            "cow", "Correlates of War – National Material Capabilities", "Correlates of War Project",
            "https://correlatesofwar.org/data-sets/national-material-capabilities", "official_file",
            "FILE_IMPORT", "Free for research with citation (terms not verified in this build)",
            "Singer, Bremer & Stuckey (1972); cite dataset version used", None,
            "chronos import cow-nmc --file NMC-60-abridged.csv",
            "CSV with stateabb, ccode, year, milex, milper, irst, pec, tpop, upop, cinc (v6.0 layout; v7.0 "
            "exists). NOT verified against the real file here. War-level datasets: NOT IMPLEMENTED.",
        ),
        SourceEntry(
            "gdelt", "GDELT DOC 2.0 API", "The GDELT Project", "https://www.gdeltproject.org", "api",
            "IMPLEMENTED_UNVERIFIED", "Open; GDELT asks for citation (terms not verified in this build)",
            "The GDELT Project, https://www.gdeltproject.org", "https://www.gdeltproject.org/about.html#termsofuse",
            "chronos fetch gdelt --query '\"artificial intelligence\"'",
            "Contemporary news coverage only (timelinevol, percent of monitored articles). Not a historical "
            "events source. Host blocked in the development sandbox.",
        ),
        SourceEntry(
            "ucdp", "Uppsala Conflict Data Program", "Uppsala University", "https://ucdp.uu.se", "api",
            "NOT_IMPLEMENTED", "CC BY 4.0 (per UCDP; verify)", "UCDP datasets", None, None,
            "REST API access conditions changed recently (token); not implemented.",
        ),
        SourceEntry(
            "user-csv", "User-supplied CSV", "user", "", "official_file", "FILE_IMPORT",
            "As declared by the user", "As declared by the user", None,
            "chronos import-csv --file f.csv --indicator x --entity y --source-url URL",
            "Provenance is whatever the user declares; CHRONOS does not verify it.",
        ),
        SourceEntry(
            "synthetic", "CHRONOS synthetic generator", "CHRONOS", "", "none", "LIVE_VERIFIED",
            "n/a", "SYNTHETIC — not historical data", None, "chronos demo",
            "Clearly labelled synthetic data for offline tests. Never mixed with real series.",
        ),
    ]
}


def entry(source_id: str) -> SourceEntry:
    return SOURCES[source_id]
