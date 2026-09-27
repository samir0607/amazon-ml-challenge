"""Deterministic multi-representation normalization of names and addresses.

Raw text is kept untouched in the source tables; this module derives extra columns.
Nothing here is specific to a closed set of countries: legal-form and street-type
vocabularies cover US / India / France-style text but any unknown token is kept as-is.
"""
import math
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

from .paths import CACHE

# ----------------------------------------------------------------------------- vocab
LEGAL = {
    # canonical form -> variants (after punctuation stripping, lowercase)
    "private": ["pvt", "pvte", "prv", "priv", "private"],
    "limited": ["ltd", "ltda", "limited", "lim", "ltd."],
    "incorporated": ["inc", "incorporated", "incorp"],
    "corporation": ["corp", "corporation", "crp"],
    "company": ["co", "company", "cos", "comp"],
    "llc": ["llc", "l l c"],
    "llp": ["llp", "l l p"],
    "lp": ["lp"],
    "plc": ["plc"],
    "pllc": ["pllc"],
    "pc": ["pc"],
    "pa": ["pa"],
    "sarl": ["sarl", "s a r l"],
    "sas": ["sas", "s a s", "sasu"],
    "sa": ["sa"],
    "eurl": ["eurl"],
    "sci": ["sci"],
    "ei": ["ei", "eirl"],
    "compagnie": ["cie", "compagnie"],
    "societe": ["ste", "societe", "soc"],
    "gmbh": ["gmbh"],
    "opc": ["opc"],
}
LEGAL_MAP = {v: k for k, vs in LEGAL.items() for v in vs if " " not in v}
LEGAL_CANON = set(LEGAL)
# Honorific / filler prefixes the noise generator sprinkles on names.
FILLER = {"the", "sri", "shri", "shree", "m s", "ms", "messrs", "and", "of", "de", "la", "le", "les", "du", "des", "et"}

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "dr": "drive", "drv": "drive",
    "ln": "lane", "ct": "court", "crt": "court", "pl": "place", "plz": "plaza", "sq": "square",
    "hwy": "highway", "hway": "highway", "pkwy": "parkway", "pky": "parkway", "fwy": "freeway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "terr": "terrace", "cres": "crescent",
    "expy": "expressway", "tpke": "turnpike", "aly": "alley", "xing": "crossing", "rte": "route",
    "mt": "mount", "ft": "fort", "hts": "heights", "jct": "junction", "ctr": "center", "cntr": "center",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "fl": "floor", "flr": "floor", "ste": "suite", "apt": "apartment", "bldg": "building",
    "rm": "room", "unit": "unit", "no": "number", "num": "number", "nbr": "number",
    "hno": "house", "h": "house",
    "opp": "opposite", "nr": "near", "ngr": "nagar", "mkt": "market", "sec": "sector",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluka",
    "chs": "society", "soc": "society", "indl": "industrial", "ind": "industrial",
    "estt": "estate", "est": "estate", "extn": "extension", "ext": "extension", "cplx": "complex",
    "bldng": "building", "po": "post", "ps": "police",
    "keralam": "kerala", "orissa": "odisha", "pondicherry": "puducherry", "bombay": "mumbai",
    "bangalore": "bengaluru", "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
    "ch": "chemin", "imp": "impasse", "fbg": "faubourg", "rte.": "route",
}
# Country-scoped abbreviations (applied only when the record's country label matches;
# unknown countries fall back to the generic table above).
COUNTRY_ADDR_ABBR = {
    "france": {"r": "rue", "bd": "boulevard", "bld": "boulevard", "av": "avenue", "ave": "avenue",
               "ch": "chemin", "che": "chemin", "imp": "impasse", "pl": "place", "rte": "route",
               "st": "saint", "ste": "sainte", "all": "allee", "sq": "square", "fg": "faubourg",
               "fbg": "faubourg", "qu": "quai", "cr": "cours", "crs": "cours", "res": "residence",
               "resid": "residence", "zi": "zone industrielle", "za": "zone activite", "ndeg": "numero",
               "no": "numero", "n": "numero", "bis": "bis", "ter": "ter", "bat": "batiment", "esc": "escalier",
               "cedex": "cedex", "lieu dit": "lieudit", "ld": "lieudit"},
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "chandigarh": "ch",
}
# Multi-word region names are collapsed to a single token "st_<code>" so that
# "New York" and "NY" agree; region codes are country-scoped to avoid collisions.
_REGION_PATTERNS = []
for _names, _tag in ((US_STATES, "us"), (IN_STATES, "in")):
    for _full, _code in sorted(_names.items(), key=lambda kv: -len(kv[0])):
        _REGION_PATTERNS.append((re.compile(r"\b" + _full.replace(" ", r"\s+") + r"\b"), f"st{_tag}{_code}"))
_US_CODES = {c: f"stus{c}" for c in US_STATES.values()}
_IN_CODES = {c: f"stin{c}" for c in IN_STATES.values()}

_RE_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_RE_WS = re.compile(r"\s+")
_RE_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co|in|biz|info|us|io|fr|co\.in|org\.in)(?:\.[a-z]{2})?$")
_RE_NUM = re.compile(r"\d+")
_RE_NUM_SUFFIX = re.compile(r"\b(\d+)(bis|ter|quater|[a-z])\b")
_RE_DOTTED_ACRO = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?\.?)")
_RE_AKA = re.compile(r"\b(?:trading as|t/a|d/b/a|dba|doing business as|formerly known as|formerly|f/k/a|fka|aka|a\.k\.a\.?)\b[:\s]*")
_VOWELS = set("aeiouy")
_RE_SOFT_C = re.compile(r"c(?=[eiy])")
_RE_SOFT_G = re.compile(r"g(?=[ei])")


def ascii_fold(s: str) -> str:
    return unidecode(s) if s and not s.isascii() else (s or "")


def _collapse_dotted(s: str) -> str:
    # "l.l.c." -> "llc", "u.s.a." -> "usa"
    return _RE_DOTTED_ACRO.sub(lambda m: m.group(1).replace(".", ""), s)


def skeleton(token: str) -> str:
    """Phonetic consonant skeleton used to bridge transliteration and typos:
    fold digraphs, drop non-initial vowels/h, collapse repeats."""
    t = token
    for a, b in (("x", "ks"), ("ph", "f"), ("sh", "s"), ("ch", "x"), ("kh", "k"), ("gh", "g"),
                 ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("q", "k"),
                 ("z", "s"), ("w", "v")):
        t = t.replace(a, b)
    t = _RE_SOFT_C.sub("s", t)
    t = _RE_SOFT_G.sub("j", t).replace("c", "k")
    if not t:
        return t
    out = [t[0]]
    for ch in t[1:]:
        if ch in _VOWELS or ch == "h":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


# Transliterated legal words ("praaivett", "limittedd") share the English skeleton.
_LEGAL_SKEL = {skeleton(w): w for w in ("private", "limited", "incorporated", "corporation", "company")}
_REGION_SKEL = {}
for _names, _tag in ((US_STATES, "us"), (IN_STATES, "in")):
    for _full, _code in _names.items():
        _REGION_SKEL.setdefault(" ".join(skeleton(t) for t in _full.split()), f"st{_tag}{_code}")


def normalize_name(raw: str) -> tuple:
    """Returns (norm, canon, core, sorted_core, skel, alias, is_domain).

    norm   : folded, lowercased, punctuation->space
    canon  : legal forms canonicalized (pvt->private, ltd->limited, ...)
    core   : canon minus legal forms and filler words
    sorted : sorted unique core tokens (robust to shuffles / doubled words)
    skel   : sorted unique skeletons of core tokens (transliteration/typo robust)
    alias  : the other side of "X trading as Y" / "formerly" constructs (else "")
    """
    s = ascii_fold(raw).lower()
    is_domain = 0
    m = _RE_DOMAIN.match(s.strip())
    if m:
        s, is_domain = m.group(1).replace("-", " "), 1
    s = _collapse_dotted(s).replace("&", " and ").replace("+", " plus ")
    alias = ""
    am = _RE_AKA.search(s)
    if am:
        left, right = s[: am.start()], s[am.end():]
        s, alias = (right, left) if len(right.strip()) >= 2 else (left, right)
    norm = _RE_WS.sub(" ", _RE_NON_ALNUM.sub(" ", s)).strip()
    alias = _RE_WS.sub(" ", _RE_NON_ALNUM.sub(" ", alias)).strip()
    toks = norm.split()
    canon_toks = [LEGAL_MAP.get(t) or (_LEGAL_SKEL.get(skeleton(t)) if len(t) >= 5 else None) or t
                  for t in toks]
    core_toks = [t for t in canon_toks if t not in LEGAL_CANON and t not in FILLER]
    if not core_toks:  # name made only of legal words: keep them
        core_toks = canon_toks
    sorted_core = " ".join(sorted(set(core_toks)))
    skel = " ".join(sorted({skeleton(t) for t in core_toks if t}))
    return (norm, " ".join(canon_toks), " ".join(core_toks), sorted_core, skel, alias, is_domain)


def normalize_address(raw, country: str) -> tuple:
    """Returns (norm, sorted_tokens, nums, postal, region).

    norm   : folded, lowercased, abbreviations expanded, regions -> st<cc><code>
    sorted : sorted unique tokens
    nums   : space-joined numeric tokens in order of appearance
    postal : 5-6 digit numeric tokens (postal-code-like), space-joined
    region : region tokens found (st...)
    """
    if raw is None:
        return ("", "", "", "", "")
    c = (country or "").lower()
    codes = _US_CODES if c in ("us", "usa", "united states") else _IN_CODES if c in ("india", "in") else {}
    cabbr = COUNTRY_ADDR_ABBR.get(c, {})
    toks = []
    # Region codes are only trusted when they form a whole comma-separated component
    # ("..., Troy, IL"); inside a component "fl"/"ct"/"in" mean floor/court/in.
    for comp in _collapse_dotted(ascii_fold(raw).lower()).split(","):
        comp = _RE_WS.sub(" ", _RE_NON_ALNUM.sub(" ", comp)).strip()
        if not comp:
            continue
        if comp in codes:
            toks.append(codes[comp])
            continue
        comp = _RE_NUM_SUFFIX.sub(r"\1 \2", comp)
        comp = " ".join(cabbr.get(t) or ADDR_ABBR.get(t, t) for t in comp.split())
        reg = _REGION_SKEL.get(" ".join(skeleton(t) for t in comp.split()))
        if reg:
            toks.append(reg)
            continue
        for pat, tag in _REGION_PATTERNS:
            if pat.search(comp):
                comp = pat.sub(tag, comp)
        toks.extend(comp.split())
    norm = " ".join(toks)
    nums = [t for t in toks if t.isdigit()]
    # digits glued to letters ("7-12/248" handled by punctuation split; "3rd" keeps 3)
    nums += [m for t in toks if not t.isdigit() for m in _RE_NUM.findall(t)]
    postal = [n for n in nums if 5 <= len(n) <= 6]
    region = [t for t in toks if t.startswith("st") and (t.startswith("stus") or t.startswith("stin")) and len(t) == 6]
    return (norm, " ".join(sorted(set(toks))), " ".join(nums), " ".join(postal), " ".join(region))


# ----------------------------------------------------------------------------- batch
NAME_COLS = ("n_norm", "n_canon", "n_core", "n_sorted", "n_skel", "n_alias", "n_domain")
ADDR_COLS = ("a_norm", "a_sorted", "a_nums", "a_postal", "a_region")


def _name_chunk(xs):
    return [normalize_name(x) for x in xs]


def _addr_chunk(args):
    xs, cs = args
    return [normalize_address(x, c) for x, c in zip(xs, cs)]


def _parallel(fn, chunks, procs):
    with Pool(procs) as p:
        out = []
        for r in p.imap(fn, chunks, chunksize=1):
            out.extend(r)
    return out


def normalize_frame(df: pl.DataFrame, procs: int = 9, chunk: int = 50_000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    ctry = df["country"].to_list()
    n = len(names)
    nres = _parallel(_name_chunk, [names[i:i + chunk] for i in range(0, n, chunk)], procs)
    ares = _parallel(_addr_chunk, [(addrs[i:i + chunk], ctry[i:i + chunk]) for i in range(0, n, chunk)], procs)
    ncols = {c: [r[j] for r in nres] for j, c in enumerate(NAME_COLS)}
    acols = {c: [r[j] for r in ares] for j, c in enumerate(ADDR_COLS)}
    extra = pl.DataFrame({**ncols, **acols}, schema_overrides={"n_domain": pl.Int8})
    return pl.concat([df, extra], how="horizontal")


# Bump whenever normalization code changes; all downstream caches key on it.
# v1 rules; v2 +domain segmentation; v3 +France abbreviations, number suffix split.
NORM_VERSION = "v3"


def load_base(split: str, source: str) -> pl.DataFrame:
    from .io import load_source
    cache = CACHE / f"normbase_{NORM_VERSION}_{split}_{source}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    out = normalize_frame(load_source(split, source))
    out.write_parquet(cache)
    return out


def load_normalized(split: str, source: str) -> pl.DataFrame:
    """Rule-based normalization followed by word segmentation of domain-style names."""
    cache = CACHE / f"norm_{NORM_VERSION}_{split}_{source}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    out = _segment_domains(load_base(split, source))
    out.write_parquet(cache)
    return out


def _segmenter() -> "Segmenter":
    counts: dict = {}
    for split in ("train", "test"):
        df = load_base(split, "source1")
        c = df.select(tok=pl.col("n_canon").str.split(" ")).explode("tok").group_by("tok").len()
        for w, n in zip(c["tok"].to_list(), c["len"].to_list()):
            if w:
                counts[w] = counts.get(w, 0) + n
    return Segmenter(counts)


def _segment_domains(df: pl.DataFrame) -> pl.DataFrame:
    dom = df.filter(pl.col("n_domain") == 1)
    if dom.height == 0:
        return df
    seg = _segmenter()
    res = [normalize_name(seg(n.replace(" ", ""))) for n in dom["n_norm"].to_list()]
    upd = pl.DataFrame({c: [r[j] for r in res] for j, c in enumerate(NAME_COLS)},
                       schema_overrides={"n_domain": pl.Int8})
    upd = upd.with_columns(n_domain=pl.lit(1, pl.Int8), idx=dom["idx"])
    return df.update(upd, on="idx")


# ----------------------------------------------------------------------------- v2 extras
class Segmenter:
    """Unigram DP word segmenter for glued names ("pediatricdentalcenter" ->
    "pediatric dental center"). Vocabulary = core-name tokens of the (clean) Source 1
    records, i.e. only challenge-provided text."""

    def __init__(self, counts: dict, maxlen: int = 20):
        total = sum(counts.values())
        self.logp = {w: math.log(c / total) for w, c in counts.items() if len(w) >= 2 or w.isdigit()}
        self.unk = math.log(1 / total) - 5.0
        self.maxlen = maxlen

    def __call__(self, s: str) -> str:
        n = len(s)
        best = [0.0] + [-1e18] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.maxlen), i):
                w = s[j:i]
                lp = self.logp.get(w, self.unk * (i - j))
                if best[j] + lp > best[i]:
                    best[i], back[i] = best[j] + lp, j
        out, i = [], n
        while i > 0:
            out.append(s[back[i]:i])
            i = back[i]
        return " ".join(reversed(out))
