"""Independent Python re-implementation of the Climpact 3.3 / climdex.pcic 1.2 station indices
used in the Wu thesis (Pass 1).  Written from the public Climpact/climdex source definitions,
NOT from the authors' scripts or outputs.

Input: daily table (year, month, day, pr, tx, tn) with NaN for missing; rows may omit 29 Feb
(365-day model calendars).  Like Climpact's station path, the series is placed on a full
Gregorian date axis; absent days become missing.
"""
import numpy as np
import pandas as pd

BASE = (1950, 1986)


# ---------------------------------------------------------------- helpers
def q8_sorted(s, n, q):
    """R type-8 quantile on rows of sorted array s (NaNs at end), n = valid counts (array)."""
    a = b = 1.0 / 3.0
    fuzz = 4 * np.finfo(float).eps
    nppm = a + q * (n + 1 - a - b) - 1
    j = np.floor(nppm + fuzz)
    h = np.where(np.abs(nppm - j) <= fuzz, 0.0, nppm - j)
    jl = np.clip(j, 0, n - 1).astype(int)
    jr = np.clip(j + 1, 0, n - 1).astype(int)
    rows = np.arange(s.shape[0]) if s.ndim == 2 else None
    if s.ndim == 2:
        left = s[rows, jl]; right = s[rows, jr]
    else:
        left = np.take_along_axis(s, jl[..., None], -1)[..., 0]
        right = np.take_along_axis(s, jr[..., None], -1)[..., 0]
    out = np.where(h == 0, left, (1 - h) * left + h * right)
    out = np.where(n == 1, s[..., 0], out)
    return np.where(n == 0, np.nan, out)


def blocks_ge(x, L):
    """select.blocks.gt.length(x, L-1): True on days belonging to runs of >= L Trues (NaN->False)."""
    x = np.asarray(x, dtype=bool)
    if L <= 1:
        return x.copy()
    out = np.zeros_like(x)
    n = len(x)
    if L > n:
        return out
    d = np.diff(np.concatenate(([0], x.view(np.int8), [0])))
    st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    for s, e in zip(st, en):
        if e - s >= L:
            out[s:e] = True
    return out


def runs(x):
    x = np.asarray(x, dtype=bool)
    d = np.diff(np.concatenate(([0], x.view(np.int8), [0])))
    return np.where(d == 1)[0], np.where(d == -1)[0]


class Series:
    def __init__(self, df, name=""):
        self.name = name
        df = df.copy()
        y0, y1 = int(df.year.min()), int(df.year.max())
        dates = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
        key = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
        full = pd.DataFrame(index=dates)
        d2 = df.set_index(key)[["pr", "tx", "tn"]]
        if d2.index.duplicated().any():
            raise ValueError("duplicate dates in " + name)
        full = full.join(d2)
        self.dates = dates
        self.year = dates.year.values
        self.month = dates.month.values
        self.day = dates.day.values
        self.pr = full.pr.values.astype(float)
        self.tx = full.tx.values.astype(float)
        self.tn = full.tn.values.astype(float)
        self.tm = (self.tx + self.tn) / 2
        self.years = np.arange(y0, y1 + 1)
        self.feb29 = (self.month == 2) & (self.day == 29)
        doy = dates.dayofyear.values.copy()
        leap = dates.is_leap_year
        doy[leap & (doy >= 60)] -= 1          # 29 Feb -> 59 (same as 28 Feb), 1 Mar -> 60
        self.jd = doy - 1                      # 0-based 0..364
        self.yidx = self.year - y0
        self.ym = (self.year - y0) * 12 + self.month - 1
        self.ny = len(self.years)
        self._masks()
        self._thresholds()

    # NA masks: annual NA if >15 missing days, or any month with >3 missing
    def _masks(self):
        self.amask, self.mmask = {}, {}
        for v in ("pr", "tx", "tn", "tm"):
            na = np.isnan(getattr(self, v))
            ann = np.bincount(self.yidx, na, self.ny)
            mon = np.bincount(self.ym, na, self.ny * 12)
            mm = np.where(mon > 3, np.nan, 1.0)
            am = np.where(ann > 15, np.nan, 1.0) * np.where(np.isnan(mm.reshape(self.ny, 12)).any(1), np.nan, 1.0)
            self.amask[v], self.mmask[v] = am, mm

    def _base_matrix(self, v):
        """[nbase_years, 365] base-period data, 29 Feb removed."""
        sel = (self.year >= BASE[0]) & (self.year <= BASE[1]) & ~self.feb29
        x = getattr(self, v)[sel]
        return x.reshape(BASE[1] - BASE[0] + 1, 365)

    def _window(self, B, n=5):
        hw = n // 2
        idx = (np.arange(365)[:, None] + np.arange(-hw, hw + 1)[None, :]) % 365   # wraps within the same year
        return B[:, idx]                       # [ny, 365, n]

    def _thresholds(self):
        self.thr, self.thr_in = {}, {}
        qs = {"tx": [0.05, 0.1, 0.5, 0.9, 0.95], "tn": [0.05, 0.1, 0.9, 0.95]}
        for v, ql in qs.items():
            B = self._base_matrix(v)
            W = self._window(B)                # ny,365,5
            nyb = W.shape[0]
            flat = np.transpose(W, (1, 0, 2)).reshape(365, -1)
            s = np.sort(flat, axis=1); n = (~np.isnan(flat)).sum(1)
            ok = n >= 0.1 * flat.shape[1]
            for q in ql:
                self.thr[(v, q)] = np.where(ok, q8_sorted(s, n, q), np.nan)
            # Zhang bootstrap for in-base percent indices
            qb = [q for q in ql if q in (0.1, 0.5, 0.9)]
            res = {q: np.full((nyb, nyb - 1, 365), np.nan) for q in qb}
            for r in range(nyb):
                others = [j for j in range(nyb) if j != r]
                Wr = np.transpose(W[others], (1, 0, 2)).reshape(365, -1)   # 365, (ny-1)*5
                stack = np.concatenate([np.repeat(Wr[None], nyb - 1, 0),
                                        np.transpose(W[others], (0, 1, 2))], axis=2)  # (ny-1),365,ny*5
                s2 = np.sort(stack, axis=2); n2 = (~np.isnan(stack)).sum(2)
                ok2 = n2 >= 0.1 * stack.shape[2]
                for q in qb:
                    res[q][r] = np.where(ok2, q8_sorted(s2, n2, q), np.nan)
            for q in qb:
                self.thr_in[(v, q)] = res[q]
        # precipitation: wet days >= 1 mm in base period, type 8
        sel = (self.year >= BASE[0]) & (self.year <= BASE[1]) & (self.pr >= 1)
        w = np.sort(self.pr[sel])
        for q in (0.95, 0.99):
            # RARB guard (documented diff from cdx.py): a series with no wet
            # days in the base period (the Morris AHCCD layer, which carries
            # no precipitation at all) gets NaN precipitation thresholds
            # instead of an index error, so its temperature indices can be
            # computed and its precipitation indices come out missing.
            self.thr[("pr", q)] = float(q8_sorted(w[None, :], np.array([len(w)]), q)[0]) if len(w) else np.nan

    # ------------------------------------------------------------ aggregations
    def agg(self, x, fn, freq="annual"):
        f = self.yidx if freq == "annual" else self.ym
        n = self.ny if freq == "annual" else self.ny * 12
        s = pd.Series(x).groupby(f)
        r = getattr(s, fn)() if fn != "count_true" else s.sum()
        return r.reindex(range(n)).values.astype(float)

    def count(self, cond, var, freq="annual"):
        m = self.amask[var] if freq == "annual" else self.mmask[var]
        f = self.yidx if freq == "annual" else self.ym
        n = self.ny if freq == "annual" else self.ny * 12
        return np.bincount(f, np.nan_to_num(cond.astype(float)), n) * m

    def pct_days(self, v, q, op):
        x = getattr(self, v)
        thr = self.thr[(v, q)][self.jd]
        with np.errstate(invalid="ignore"):
            dat = (x > thr) if op == ">" else (x < thr)
        dat = np.where(np.isnan(x) | np.isnan(thr), np.nan, dat.astype(float))
        inset = (self.year >= BASE[0]) & (self.year <= BASE[1])
        bi = self.thr_in[(v, q)]                  # nyb, nyb-1, 365
        yb = self.year[inset] - BASE[0]
        tb = bi[yb, :, self.jd[inset]]            # ndays, nyb-1
        xb = x[inset][:, None]
        with np.errstate(invalid="ignore"):
            fr = (xb > tb) if op == ">" else (xb < tb)
        fr = np.where(np.isnan(xb) | np.isnan(tb), np.nan, fr.astype(float))
        allna = np.isnan(fr).all(1)
        val = np.nansum(fr, 1) / (bi.shape[0] - 1)
        dat[inset] = np.where(allna, np.nan, val)
        na = np.isnan(dat)
        nmiss = np.bincount(self.yidx, na, self.ny)
        m = np.where(nmiss > 15, np.nan, 1.0)
        mean = pd.Series(dat).groupby(self.yidx).mean().values
        return mean * 100 * m * self.amask[v]

    def spell_days(self, v, q, op, L):
        """WSDI/CSDI-type: days in runs >= L, spells cannot span years, outbase thresholds."""
        x = getattr(self, v)
        thr = self.thr[(v, q)][self.jd]
        with np.errstate(invalid="ignore"):
            c = (x > thr) if op == ">" else (x < thr)
        nam = np.isnan(x + thr)
        out = np.zeros(self.ny)
        for i in range(self.ny):
            s = self.yidx == i
            out[i] = blocks_ge(c[s] & ~nam[s], L).sum()
        miss = np.bincount(self.yidx, nam, self.ny)
        return out * np.where(miss > 15, np.nan, 1.0) * self.amask[v]

    def dual_events(self, q, op, n):
        """Climpact txdtnd / txbdtnbd: number of events of runs > n days with both TX and TN beyond q."""
        thx = self.thr[("tx", q)][self.jd]; thn = self.thr[("tn", q)][self.jd]
        with np.errstate(invalid="ignore"):
            if op == ">":
                c = (self.tx > thx) & (self.tn > thn)
            else:
                c = (self.tx < thx) & (self.tn < thn)
        m1 = np.bincount(self.yidx, np.isnan(self.tx + thx), self.ny) > 15
        m2 = np.bincount(self.yidx, np.isnan(self.tn + thn), self.ny) > 15
        out = np.zeros(self.ny)
        for i in range(self.ny):
            s = self.yidx == i
            p = blocks_ge(c[s], n + 1)
            inv = blocks_ge(~p, 4)          # gaps of <=3 days merged
            p = ~inv
            st, en = runs(p)
            out[i] = len(st)
        return out * np.where(m1 | m2, np.nan, 1.0) * self.amask["tx"]

    def spell_max(self, cond, var, freq="annual"):
        """climdex spell.length.max with spells.can.span.years=TRUE: run length at its end day."""
        b = np.nan_to_num(cond.astype(float)).astype(bool)
        st, en = runs(b)
        L = np.zeros(len(b))
        L[en - 1] = en - st
        f = self.yidx if freq == "annual" else self.ym
        n = self.ny if freq == "annual" else self.ny * 12
        mx = pd.Series(L).groupby(f).max().reindex(range(n)).values
        allt = pd.Series(b).groupby(f).all().reindex(range(n)).values.astype(bool)
        mx = np.where((mx == 0) & allt, np.nan, mx)
        mask = self.amask["pr"] if freq == "annual" else self.mmask["pr"]
        return mx * mask

    def rxn(self, n, freq="annual"):
        p = np.nan_to_num(self.pr)
        if n == 1:
            m = self.amask["pr"] if freq == "annual" else self.mmask["pr"]
            return self.agg(self.pr, "max", freq) * m
        k = np.convolve(p, np.ones(n), "valid")
        left = n // 2; right = n - left - 1
        rs = np.concatenate([np.zeros(left), k, np.zeros(right)])
        m = self.amask["pr"] if freq == "annual" else self.mmask["pr"]
        return self.agg(rs, "max", freq) * m

    def gsl(self):
        out = np.full(self.ny, np.nan)
        for i in range(self.ny):
            s = self.yidx == i
            t = self.tm[s]; mo = self.month[s]
            mid = np.where(mo == 7)[0][0]
            with np.errstate(invalid="ignore"):
                b = np.where(blocks_ge(np.nan_to_num(t[:mid] > 5, nan=0).astype(bool) & ~np.isnan(t[:mid]), 6))[0]
                e = np.where(blocks_ge((t[mid:] < 5) & ~np.isnan(t[mid:]), 6))[0] - 1
            if len(b) == 0:
                out[i] = 0
            elif len(e) == 0:
                out[i] = len(t) - b[0]
            else:
                out[i] = (e[0] + mid) - b[0] + 1   # 0-based start b, end = e+mid (0-based) -> len
        return out * self.amask["tm"]

    # ------------------------------------------------------------ heatwaves (EHF NF13 / ECF)
    def heatwaves(self):
        tavg_full = self.tm
        base = (self.year >= BASE[0]) & (self.year <= BASE[1])
        tb = tavg_full[base]; tb = tb[~np.isnan(tb)]
        t95 = np.quantile(tb, 0.95); t05 = np.quantile(tb, 0.05)    # R type 7
        keep = ~self.feb29
        t = tavg_full[keep]; yf = self.yidx[keep]
        n = len(t)
        T3 = np.full(n, np.nan); T30 = np.full(n, np.nan)
        T3[2:] = (t[2:] + t[1:-1] + t[:-2]) / 3
        c = np.concatenate([[0], np.cumsum(np.nan_to_num(t))])
        i = np.arange(32, n)
        T30[i] = (c[i - 2] - c[i - 32]) / 30.0                       # t[i-32..i-3]
        T3[:32] = np.nan
        accl = T3 - T30
        with np.errstate(invalid="ignore"):
            EHF = np.where(np.isnan(accl), np.nan, np.maximum(1, accl)) * (T3 - t95)
            ECF = np.where(np.isnan(accl), np.nan, np.minimum(-1, accl)) * (-(T3 - t05))
            eb = blocks_ge((EHF > 0) & ~np.isnan(EHF), 3)
            cb = blocks_ge((ECF < 0) & ~np.isnan(ECF), 3)
        hm = self.amask["tn"] * self.amask["tx"]
        H = self._aspects(eb, EHF, yf, ecf=False) ; C = self._aspects(cb, ECF, yf, ecf=True)
        return {"HWA": H[0] * hm, "HWF": H[1] * hm, "HWN": H[2] * hm,
                "CWA": C[0] * hm, "CWF": C[1] * hm, "CWN": C[2] * hm}

    def _aspects(self, bstr, data, yf, ecf):
        n = len(bstr)
        hwa = np.full(self.ny, np.nan); hwf = np.zeros(self.ny); hwn = np.zeros(self.ny)
        for y in range(self.ny):
            si = np.where(yf == y)[0]
            ext = np.arange(si[0], si[-1] + 91)
            eb = np.array([bstr[k] if k < n else np.nan for k in ext], dtype=float)
            ed = np.array([data[k] if k < n else np.nan for k in ext], dtype=float)
            # R rle on logical with NA: each NA its own run
            vals, lens = [], []
            for v in eb:
                if len(vals) and not np.isnan(v) and not np.isnan(vals[-1]) and v == vals[-1]:
                    lens[-1] += 1
                else:
                    vals.append(v); lens.append(1)
            vals = np.array(vals); lens = np.array(lens)
            cs = np.cumsum(lens)
            last = len(eb) - 90
            tv = list(np.where((lens >= 3) & (cs <= last) & (vals == 1))[0])
            p1 = bstr[si[0] - 1] if si[0] - 1 >= 0 else np.nan
            p2 = bstr[si[0] - 2] if si[0] - 2 >= 0 else np.nan
            if not np.isnan(eb[0]) and not np.isnan(eb[1]) and not (isinstance(p1, float) and np.isnan(p1)):
                if eb[0] == 1 and eb[1] == 1 and eb[2] == 0 and p1 == 1:
                    if len(tv) == 0 or tv[0] != 0:
                        tv = [0] + tv
            if not np.isnan(eb[0]) and not (isinstance(p1, float) and np.isnan(p1)) and not (isinstance(p2, float) and np.isnan(p2)):
                if eb[0] == 1 and eb[1] == 0 and p2 == 1 and p1 == 1:
                    if len(tv) == 0 or tv[0] != 0:
                        tv = [0] + tv
            ev = np.where((lens >= 3) & (cs > last) & (vals == 1))[0]
            if len(cs) > 1 and len(ev) > 0 and ev[0] > 0 and cs[ev[0] - 1] < last:
                tv.append(ev[0])
                if cs[ev[0]] > len(si) + 14:
                    lens[ev[0]] -= cs[ev[0]] - (len(si) + 14)
                    cs = np.cumsum(lens)
            nh = len(tv)
            hwn[y] = nh
            if nh > 0:
                hm, ha = [], []
                for r in tv:
                    i1 = 0 if r == 0 else cs[r - 1]
                    i2 = cs[r]
                    seg = ed[i1:i2]
                    hm.append(np.nanmean(seg)); ha.append(np.nanmin(seg) if ecf else np.nanmax(seg))
                hm = np.array(hm)
                if not np.all(np.isnan(hm)):
                    k = np.nanargmin(hm) if ecf else np.nanargmax(hm)
                    hwa[y] = ha[k]
                hwf[y] = lens[tv].sum()
        return hwa, hwf, hwn

    # ------------------------------------------------------------ the 44 indices + extras
    def annual(self, dtr_thr=15.0):
        A = self.amask
        r = {}
        tx, tn, pr, tm = self.tx, self.tn, self.pr, self.tm
        with np.errstate(invalid="ignore"):
            r["PRCPTOT"] = self.agg(np.where(pr >= 1, pr, 0), "sum") * A["pr"]
            r["R10mm"] = self.count(pr >= 10, "pr")
            r["R20mm"] = self.count(pr >= 20, "pr")
            r["RX1day"] = self.rxn(1)
            r["RX3day"] = self.rxn(3)
            r["RX5day"] = self.rxn(5)
            wet = self.count(pr >= 1, "pr")
            r["SDII"] = np.where(wet > 0, r["PRCPTOT"] / np.where(wet > 0, wet, 1), 0) * A["pr"]
            q95, q99 = self.thr[("pr", 0.95)], self.thr[("pr", 0.99)]
            r["R95p"] = self.agg(np.where(pr > q95, pr, 0), "sum") * A["pr"]
            r["R99p"] = self.agg(np.where(pr > q99, pr, 0), "sum") * A["pr"]
            r["R95pTOT"] = 100 * r["R95p"] / r["PRCPTOT"]
            r["R99pTOT"] = 100 * r["R99p"] / r["PRCPTOT"]
            r["CDD"] = self.spell_max((pr < 1) & ~np.isnan(pr), "pr")
            r["CWD"] = self.spell_max((pr >= 1) & ~np.isnan(pr), "pr")
            r["CSDI6"] = self.spell_days("tn", 0.1, "<", 6)
            r["CSDI3"] = self.spell_days("tn", 0.1, "<", 3)
            r["WSDI6"] = self.spell_days("tx", 0.9, ">", 6)
            r["WSDI3"] = self.spell_days("tx", 0.9, ">", 3)
            r["FD"] = self.count(tn < 0, "tn")
            r["ID"] = self.count(tx < 0, "tx")
            r["SU"] = self.count(tx > 25, "tx")
            r["TR"] = self.count(tn > 20, "tn")
            r["TNn"] = self.agg(tn, "min") * A["tn"]
            r["TNx"] = self.agg(tn, "max") * A["tn"]
            r["TXx"] = self.agg(tx, "max") * A["tx"]
            r["TXn"] = self.agg(tx, "min") * A["tx"]
            r["TXm"] = self.agg(tx, "mean") * A["tx"]
            r["TMm"] = self.agg(tm, "mean") * A["tx"] * A["tn"]
            r["TNm"] = self.agg(tn, "mean") * A["tn"]
            r["TN10p"] = self.pct_days("tn", 0.1, "<")
            r["TN90p"] = self.pct_days("tn", 0.9, ">")
            r["TX10p"] = self.pct_days("tx", 0.1, "<")
            r["TX90p"] = self.pct_days("tx", 0.9, ">")
            r["DTR"] = self.agg(tx - tn, "mean") * A["tm"]
            r["DTRgt15"] = self.count((tx - tn) > dtr_thr, "tm")
            r["TXb3TNb3"] = self.dual_events(0.05, "<", 3)          # Climpact txbdtnbd, n=3 (cold, events)
            r["TXge30"] = self.count(tx >= 30, "tx")
            r["TXgt50p"] = self.pct_days("tx", 0.5, ">")
            r["GDDgrow10"] = self.agg(np.where(tm > 10, tm - 10, 0), "sum") * A["tm"]
            r["TNltm20"] = self.count(tn < -20, "tn")
            r["GSL"] = self.gsl()
            hw = self.heatwaves()
            r.update({k: hw[k] for k in ("CWA", "CWF", "HWA", "HWF")})
            # printed-definition / sensitivity variants
            r["x_TXd3TNd3_warm_events"] = self.dual_events(0.95, ">", 3)
            r["x_DTRge15"] = self.count((tx - tn) >= dtr_thr, "tm")
            r["x_HWN"] = hw["HWN"]; r["x_CWN"] = hw["CWN"]
        df = pd.DataFrame(r, index=self.years)
        return df

    def monthly(self):
        M = self.mmask
        tx, tn, pr, tm = self.tx, self.tn, self.pr, self.tm
        r = {}
        with np.errstate(invalid="ignore"):
            r["PRCPTOT"] = self.agg(np.where(pr >= 1, pr, 0), "sum", "m") * M["pr"]
            r["R10mm"] = self.count(pr >= 10, "pr", "m")
            r["RX1day"] = self.rxn(1, "m")
            r["RX5day"] = self.rxn(5, "m")
            r["CDD"] = self.spell_max((pr < 1) & ~np.isnan(pr), "pr", "m")
            r["SU"] = self.count(tx > 25, "tx", "m")
            r["ID"] = self.count(tx < 0, "tx", "m")
            r["TXx"] = self.agg(tx, "max", "m") * M["tx"]
            r["TXn"] = self.agg(tx, "min", "m") * M["tx"]
            r["TMm"] = self.agg(tm, "mean", "m") * M["tx"] * M["tn"]
        idx = pd.MultiIndex.from_arrays([np.repeat(self.years, 12), np.tile(np.arange(1, 13), self.ny)], names=["year", "month"])
        return pd.DataFrame(r, index=idx)


INDICES = ["PRCPTOT", "R10mm", "R20mm", "RX1day", "RX3day", "RX5day", "SDII", "R95p", "R95pTOT", "R99p", "R99pTOT", "CDD", "CWD",
           "CSDI6", "CSDI3", "WSDI6", "WSDI3", "FD", "ID", "SU", "TR", "TNn", "TNx", "TXx", "TXn", "TXm", "TMm", "TNm", "TN10p", "TN90p",
           "TX10p", "TX90p", "DTR", "DTRgt15", "TXb3TNb3", "TXge30", "TXgt50p", "GDDgrow10", "TNltm20", "GSL", "CWA", "CWF", "HWA", "HWF"]
PREC = INDICES[:13]
TEMP = INDICES[13:]


def read_daily(path):
    d = pd.read_csv(path, sep=r"\s+", header=None, names=["year", "month", "day", "pr", "tx", "tn"])
    for c in ("pr", "tx", "tn"):
        d.loc[d[c] <= -99.8, c] = np.nan
    return d
