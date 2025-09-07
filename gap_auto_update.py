#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (Enhanced) See chat for detailed description.
import argparse, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def try_import(name):
    try: return __import__(name)
    except Exception: return None

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'path'; plt.rcParams['pdf.fonttype'] = 42

yf = try_import("yfinance")
ak = try_import("akshare")

def read_policy_dr007(csv_path: Path, start='2014-01-01') -> pd.DataFrame:
    import pandas as pd; import numpy as np
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    for enc in ['utf-8-sig','utf-8','gbk','gb2312']:
        try: df = pd.read_csv(csv_path, encoding=enc); break
        except Exception: continue
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
    out = pd.DataFrame(index=pd.date_range(start=start, end=datetime.today(), freq='M'))
    if '政策利率' in df.columns:
        out['policy'] = pd.to_numeric(df['政策利率'], errors='coerce').resample('M').mean()
    else: raise ValueError("CSV missing column: 政策利率")
    if 'DR007' in df.columns:
        out['dr007'] = pd.to_numeric(df['DR007'], errors='coerce').resample('M').mean()
    else: raise ValueError("CSV missing column: DR007")
    return out

def fetch_policy_dr007_via_akshare(start='2014-01-01') -> pd.DataFrame:
    import pandas as pd
    if ak is None: raise RuntimeError("AkShare not installed")
    dr_df = None
    for func_name in ["macro_china_repo_rate", "macro_china_dr_repo", "repo_rate"]:
        try:
            func = getattr(ak, func_name); tmp = func()
            tmp.columns = [str(c).strip() for c in tmp.columns]
            dcol = next((c for c in tmp.columns if any(k in c for k in ['日期','时间','date','Date'])), None)
            if dcol is None: continue
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce'); tmp = tmp.set_index(dcol).sort_index()
            cdr = next((c for c in tmp.columns if 'DR007' in c.upper() or 'DR 007' in c.upper()), None)
            if cdr is None: continue
            dr_df = tmp[[cdr]].rename(columns={cdr:'dr007'}); break
        except Exception: continue
    if dr_df is None: raise RuntimeError("AkShare: DR007 endpoint not found")
    dr_m = dr_df.resample('M').mean()
    pol = None
    for fn, hint in [("macro_china_omo_daily","操作利率"),("macro_china_omo","中标利率"),("macro_china_mlf","利率")]:
        try:
            func = getattr(ak, fn); tmp = func()
            tmp.columns = [str(c).strip() for c in tmp.columns]
            dcol = next((c for c in tmp.columns if any(k in c for k in ['日期','时间','date','Date'])), None)
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce'); tmp = tmp.set_index(dcol).sort_index()
            col = next((c for c in tmp.columns if hint in c), None)
            if col is None:
                num_cols = tmp.select_dtypes(include='number').columns
                if len(num_cols): col = num_cols[-1]
            pol = tmp[[col]].rename(columns={col:'policy'}).resample('M').mean(); break
        except Exception: continue
    if pol is None: raise RuntimeError("AkShare: policy rate endpoint not found")
    out = pd.DataFrame(index=pd.date_range(start=start, end=datetime.today(), freq='M'))
    out = out.join(pol, how='left').join(dr_m, how='left'); return out

def fetch_index_monthly(start='2014-01-01') -> pd.DataFrame:
    import pandas as pd
    if yf is None: raise RuntimeError("yfinance not installed. pip install yfinance")
    data = yf.download("399006.SZ", start=start, progress=False, auto_adjust=True)
    if data.empty: raise RuntimeError("Yahoo Finance returned no data for 399006.SZ")
    mclose = data['Close'].resample('M').last().dropna(); mret = mclose.pct_change()
    level = (1 + mret.fillna(0)).cumprod() * 100.0
    return pd.DataFrame({'close_m': mclose, 'ret_m': mret, 'level_base100': level})

def read_margin_ratio_csv(csv_path: Path, start='2014-01-01'):
    import pandas as pd
    if not csv_path or not Path(csv_path).exists(): return None
    for enc in ['utf-8-sig','utf-8','gbk','gb2312']:
        try: df = pd.read_csv(csv_path, encoding=enc); break
        except Exception: continue
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
    ratio_col = next((c for c in df.columns if any(k in str(c) for k in ['占比','ratio','MarginShare'])), None)
    if ratio_col is not None:
        s = pd.to_numeric(df[ratio_col], errors='coerce')
    else:
        mm = next((c for c in df.columns if '融' in str(c) and ('成交' in str(c) or '交易额' in str(c))), None)
        mkt = next((c for c in df.columns if '全市场' in str(c) and ('成交' in str(c) or '交易额' in str(c))), None)
        if mm is None or mkt is None: return None
        s = pd.to_numeric(df[mm], errors='coerce') / pd.to_numeric(df[mkt], errors='coerce')
    return s.resample('M').mean().rename('margin_ratio')

def fetch_margin_ratio_via_akshare(start='2014-01-01'):
    if ak is None: return None
    try:
        sse = szse = None
        for fn in ["stock_margin_sse","margin_sse"]:
            if hasattr(ak, fn):
                tmp = getattr(ak, fn)(); tmp.columns = [str(c).strip() for c in tmp.columns]
                dcol = next((c for c in tmp.columns if '日期' in c or 'date' in c.lower() or '时间' in c), None)
                if dcol is None: continue
                tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce'); tmp = tmp.set_index(dcol).sort_index()
                cand = [c for c in tmp.columns if ('成交' in c or '交易额' in c) and ('融资融券' in c or '两融' in c)]
                if cand:
                    if 'sse' in fn: sse = tmp[cand[0]]
                    else: szse = tmp[cand[0]]
        if sse is None or szse is None: return None
        mm = (sse.fillna(0) + szse.fillna(0)).to_frame('mm')
        # market turnover heuristic omitted for brevity; return None to avoid false data
        return None
    except Exception: return None

def read_pe_csv(csv_path: Path, start='2014-01-01'):
    import pandas as pd
    if not csv_path or not Path(csv_path).exists(): return None
    for enc in ['utf-8-sig','utf-8','gbk','gb2312']:
        try: df = pd.read_csv(csv_path, encoding=enc); break
        except Exception: continue
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
    col = next((c for c in df.columns if any(k in str(c).upper() for k in ['PE','PE_TTM','估值分位','PERCENTILE'])), None)
    if col is None: return None
    s = pd.to_numeric(df[col], errors='coerce').rename('pe_or_pct')
    return s.resample('M').last()

def zscore(x): return (x - x.mean()) / x.std(ddof=1)

def fwd_cum_simple(r, k):
    rv = r.values; out = np.full_like(rv, np.nan, dtype=float)
    for i in range(len(rv)-k):
        w = rv[i+1:i+1+k]; out[i] = np.nan if np.any(np.isnan(w)) else np.prod(1+w)-1.0
    return pd.Series(out, index=r.index)

def make_core_charts(panel, out_dir: Path):
    fig = plt.figure(figsize=(12,5), dpi=150); ax = plt.gca()
    ax.plot(panel.index, panel['gap_z'], label='Gap Z', linewidth=2.0, color='#F59E0B')
    ax.plot(panel.index, panel['index_z'], label='ChiNext Z', linewidth=2.0, color='#EF4444')
    ax.plot(panel.index, panel['gap_ma12_z'], label='Gap MA12 Z', linewidth=2.4, color='#1D4ED8')
    ax.set_title(f'Gap vs ChiNext (Z) + Gap MA12 | {panel.index.min():%Y-%m}–{panel.index.max():%Y-%m}')
    ax.set_ylabel('Z-score'); ax.set_xlabel('Date'); ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax.legend()
    plt.tight_layout()
    for ext in ['png','svg','pdf']: plt.savefig(out_dir / f'gap_vs_chinext_with_ma12.{ext}', bbox_inches='tight')
    plt.close(fig)
    roll = panel['gap'].rolling(12).corr(panel['ret_m'])
    fig2 = plt.figure(figsize=(12,5), dpi=150); ax2 = plt.gca()
    ax2.plot(roll.index, roll, label='Rolling corr (12m)', color='#F59E0B')
    ax2.axhline(0.0, linestyle='--', linewidth=1.0, color='#F59E0B', alpha=0.6, label='0 baseline')
    ax2.set_title('12m Rolling Correlation: Gap vs ChiNext monthly return')
    ax2.set_ylabel('Correlation (12m)'); ax2.set_xlabel('Date'); ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax2.legend()
    plt.tight_layout()
    for ext in ['png','svg','pdf']: plt.savefig(out_dir / f'rolling_corr_12m.{ext}', bbox_inches='tight')
    plt.close(fig2)

def make_factor_panel(panel, out_dir: Path):
    fig = plt.figure(figsize=(12,5), dpi=150); ax = plt.gca()
    if 'margin_ratio' in panel: ax.plot(panel.index, zscore(panel['margin_ratio']), label='Margin share (Z)', linewidth=2.0)
    if 'pe_percentile' in panel: ax.plot(panel.index, -zscore(panel['pe_percentile']), label='-Valuation pct (Z)', linewidth=2.0)
    ax.plot(panel.index, zscore(panel['rel_gap']), label='RelGap (Z)', linewidth=2.0)
    ax.set_title('Factor panel (standardized)')
    ax.set_ylabel('Z-score'); ax.set_xlabel('Date'); ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax.legend()
    plt.tight_layout()
    for ext in ['png','svg','pdf']: plt.savefig(out_dir / f'factor_panel_standardized.{ext}', bbox_inches='tight')
    plt.close(fig)

def make_corr_table(panel, out_dir: Path):
    import pandas as pd
    factors = [c for c in ['gap','rel_gap','ratio','margin_ratio','pe_percentile'] if c in panel.columns]
    rows = []
    for f in factors:
        for k in [0,3,6,12]:
            if k==0:
                dfk = panel[[f,'ret_m']].dropna(); corr = dfk[f].corr(dfk['ret_m'])
            else:
                fwd = fwd_cum_simple(panel['ret_m'], k); dfk = pd.concat([panel[f], fwd], axis=1).dropna(); corr = dfk.iloc[:,0].corr(dfk.iloc[:,1])
            rows.append({'factor': f, 'k_months': k, 'corr': float(corr)})
    pd.DataFrame(rows).pivot(index='factor', columns='k_months', values='corr').to_csv(out_dir / 'factor_corr_table.csv', encoding='utf-8-sig')

def fetch_index_monthly(start='2014-01-01'):  # redefine to keep top order
    import pandas as pd
    if yf is None: raise RuntimeError("yfinance not installed. pip install yfinance")
    data = yf.download("399006.SZ", start=start, progress=False, auto_adjust=True)
    if data.empty: raise RuntimeError("Yahoo Finance returned no data for 399006.SZ")
    mclose = data['Close'].resample('M').last().dropna(); mret = mclose.pct_change(); level = (1+mret.fillna(0)).cumprod()*100.0
    return pd.DataFrame({'close_m': mclose, 'ret_m': mret, 'level_base100': level})

def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--csv', required=False)
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--margin-csv', default=None); ap.add_argument('--pe-csv', default=None)
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    policy_dr = None
    if ak is not None:
        try: policy_dr = fetch_policy_dr007_via_akshare(start=a.start); print("[info] policy/dr007 via AkShare")
        except Exception as e: print(f"[warn] AkShare policy/dr007 failed: {e}")
    if policy_dr is None:
        if not a.csv: print("[error] Need --csv for policy & DR007 fallback"); sys.exit(2)
        policy_dr = read_policy_dr007(Path(a.csv), start=a.start); print("[info] policy/dr007 via CSV")
    index_m = fetch_index_monthly(start=a.start)
    panel = policy_dr.join(index_m, how='inner')
    panel['gap'] = panel['policy'] - panel['dr007']
    panel['gap_ma12'] = panel['gap'].rolling(12).mean()
    panel['rel_gap'] = panel['gap'] / panel['policy']
    panel['ratio'] = panel['dr007'] / panel['policy']
    # margin & valuation
    mr = fetch_margin_ratio_via_akshare(start=a.start)
    if mr is None and a.margin_csv: mr = read_margin_ratio_csv(Path(a.margin_csv), start=a.start)
    if mr is not None: panel = panel.join(mr, how='left')
    pe = read_pe_csv(Path(a.pe_csv), start=a.start) if a.pe_csv else None
    if pe is not None:
        panel['pe_percentile'] = pe if pe.max() <= 3.0 else pe.rank(pct=True)
    # standardize for chart
    def z(x): return (x - x.mean())/x.std(ddof=1)
    panel['gap_z'] = z(panel['gap']); panel['gap_ma12_z'] = z(panel['gap_ma12']); panel['index_z'] = z(panel['level_base100'])
    panel.to_csv(out_dir / 'gap_chinext_monthly_panel.csv', encoding='utf-8-sig')
    make_core_charts(panel, out_dir); make_factor_panel(panel, out_dir); make_corr_table(panel, out_dir)
    print(f"[done] Updated through {panel.index.max():%Y-%m}. Files saved to: {out_dir}")

if __name__ == '__main__':
    main()
