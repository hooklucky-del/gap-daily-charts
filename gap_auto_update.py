
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap vs ChiNext auto-updater (robust index fetch)
- Tries yfinance(399006.SZ) → ETF proxy(159915.SZ)
- Optional --index-csv fallback (columns: 日期 + Close 或 Return)
- Outputs charts (PNG/SVG/PDF) + factor table + panel CSV
"""
import argparse, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'path'; plt.rcParams['pdf.fonttype'] = 42

def try_import(name):
    try: return __import__(name)
    except Exception: return None

yf = try_import("yfinance")
ak = try_import("akshare")

# ------------ Loaders ------------
def read_policy_dr007(csv_path: Path, start='2014-01-01') -> pd.DataFrame:
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
    if ak is None: raise RuntimeError("AkShare not installed")
    # DR007 daily
    dr_df = None
    for func_name in ["macro_china_repo_rate", "macro_china_dr_repo", "repo_rate"]:
        try:
            func = getattr(ak, func_name); tmp = func()
            tmp.columns = [str(c).strip() for c in tmp.columns]
            dcol = next((c for c in tmp.columns if any(k in c for k in ['日期','时间','date','Date'])), None)
            if dcol is None: continue
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce'); tmp = tmp.set_index(dcol).sort_index()
            cdr = next((c for c in tmp.columns if 'DR007' in c.upper()), None)
            if cdr is None: continue
            dr_df = tmp[[cdr]].rename(columns={cdr:'dr007'}); break
        except Exception: continue
    if dr_df is None: raise RuntimeError("AkShare: DR007 endpoint not found")
    dr_m = dr_df.resample('M').mean()
    # Policy rate (prefer OMO 7d; fallback MLF)
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
    return out.join(pol, how='left').join(dr_m, how='left')

def read_index_csv(index_csv: Path) -> pd.DataFrame:
    for enc in ['utf-8-sig','utf-8','gbk','gb2312']:
        try: df = pd.read_csv(index_csv, encoding=enc); break
        except Exception: continue
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
    close_col = next((c for c in df.columns if str(c).lower() in ['close','收盘','收盘价']), None)
    ret_col   = next((c for c in df.columns if 'ret' in str(c).lower() or '收益' in str(c)), None)
    if close_col is not None:
        close_m = pd.to_numeric(df[close_col], errors='coerce').resample('M').last().dropna()
        ret_m = close_m.pct_change()
    elif ret_col is not None:
        ret_m = pd.to_numeric(df[ret_col], errors='coerce').resample('M').sum()
        if ret_m.abs().max() > 1.0: ret_m = ret_m/100.0
        close_m = (1+ret_m.fillna(0)).cumprod()*100.0
    else:
        raise ValueError("index CSV 需要 Close(收盘) 或 Return(收益) 列")
    level = (1+ret_m.fillna(0)).cumprod()*100.0
    return pd.concat([close_m.rename('close_m'), ret_m.rename('ret_m'), level.rename('level_base100')], axis=1)

def fetch_index_monthly(start='2014-01-01', index_csv: Path|None=None) -> pd.DataFrame:
    if index_csv is not None and Path(index_csv).exists():
        return read_index_csv(index_csv)
    yfinance = try_import("yfinance")
    if yfinance is None:
        raise RuntimeError("yfinance not installed; 请提供 --index-csv 回退")
    for ticker in ["399006.SZ", "159915.SZ"]:
        try:
            data = yfinance.download(ticker, start=start, progress=False, auto_adjust=True, threads=False)
            if data is None or data.empty or 'Close' not in data: continue
            mclose = data['Close'].resample('M').last().dropna()
            if len(mclose) < 2: continue
            mret = mclose.pct_change()
            level = (1+mret.fillna(0)).cumprod()*100.0
            return pd.concat([mclose.rename('close_m'),
                              mret.rename('ret_m'),
                              level.rename('level_base100')], axis=1)
        except Exception:
            continue
    raise RuntimeError("获取指数失败：yfinance 对 399006.SZ/159915.SZ 返回空。可在 data/ 放 index_monthly.csv 并用 --index-csv 指定。")

# ------------ Utils ------------
def zscore(x: pd.Series) -> pd.Series: return (x - x.mean()) / x.std(ddof=1)
def fwd_cum_simple(r: pd.Series, k: int) -> pd.Series:
    rv = r.values; out = np.full_like(rv, np.nan, dtype=float)
    for i in range(len(rv)-k):
        w = rv[i+1:i+1+k]; out[i] = np.nan if np.any(np.isnan(w)) else np.prod(1+w)-1.0
    return pd.Series(out, index=r.index)

# ------------ Charts ------------
def make_core_charts(panel: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12,5), dpi=150); ax = plt.gca()
    ax.plot(panel.index, zscore(panel['gap']), label='Gap Z', linewidth=2.0, color='#F59E0B')
    ax.plot(panel.index, zscore(panel['level_base100']), label='ChiNext Z', linewidth=2.0, color='#EF4444')
    ax.plot(panel.index, zscore(panel['gap'].rolling(12).mean()), label='Gap MA12 Z', linewidth=2.4, color='#1D4ED8')
    ax.set_title(f'Gap vs ChiNext (Z) + Gap MA12 | {panel.index.min():%Y-%m}–{panel.index.max():%Y-%m}')
    ax.set_ylabel('Z-score'); ax.set_xlabel('Date'); ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax.legend()
    for ext in ['png','svg','pdf']: fig.savefig(out_dir / f'gap_vs_chinext_with_ma12.{ext}', bbox_inches='tight')
    plt.close(fig)
    roll = panel['gap'].rolling(12).corr(panel['ret_m'])
    fig2 = plt.figure(figsize=(12,5), dpi=150); ax2 = plt.gca()
    ax2.plot(roll.index, roll, label='Rolling corr (12m)', color='#F59E0B')
    ax2.axhline(0.0, linestyle='--', linewidth=1.0, color='#F59E0B', alpha=0.6, label='0 baseline')
    ax2.set_title('12m Rolling Correlation: Gap vs ChiNext monthly return')
    ax2.set_ylabel('Correlation (12m)'); ax2.set_xlabel('Date'); ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax2.legend()
    for ext in ['png','svg','pdf']: fig2.savefig(out_dir / f'rolling_corr_12m.{ext}', bbox_inches='tight')
    plt.close(fig2)

def make_factor_panel(panel: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(12,5), dpi=150); ax = plt.gca()
    if 'margin_ratio' in panel: ax.plot(panel.index, zscore(panel['margin_ratio']), label='Margin share (Z)', linewidth=2.0)
    if 'pe_percentile' in panel: ax.plot(panel.index, -zscore(panel['pe_percentile']), label='-Valuation pct (Z)', linewidth=2.0)
    if 'gap' in panel and 'policy' in panel and 'dr007' in panel:
        ax.plot(panel.index, zscore((panel['policy']-panel['dr007'])/panel['policy']), label='RelGap (Z)', linewidth=2.0)
    ax.set_title('Factor panel (standardized)'); ax.set_ylabel('Z-score'); ax.set_xlabel('Date')
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax.legend()
    for ext in ['png','svg','pdf']: fig.savefig(out_dir / f'factor_panel_standardized.{ext}', bbox_inches='tight')
    plt.close(fig)

def make_corr_table(panel: pd.DataFrame, out_dir: Path):
    rows = []
    factors = [c for c in ['gap','rel_gap','ratio','margin_ratio','pe_percentile'] if c in panel.columns]
    for f in factors:
        for k in [0,3,6,12]:
            if k==0:
                dfk = panel[[f,'ret_m']].dropna(); corr = dfk[f].corr(dfk['ret_m'])
            else:
                fwd = fwd_cum_simple(panel['ret_m'], k); dfk = pd.concat([panel[f], fwd], axis=1).dropna(); corr = dfk.iloc[:,0].corr(dfk.iloc[:,1])
            rows.append({'factor': f, 'k_months': k, 'corr': float(corr)})
    pd.DataFrame(rows).pivot(index='factor', columns='k_months', values='corr').to_csv(out_dir / 'factor_corr_table.csv', encoding='utf-8-sig')

# ------------ Main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--csv', required=False, help='policy & DR007 CSV')
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--margin-csv', default=None)
    ap.add_argument('--pe-csv', default=None)
    ap.add_argument('--index-csv', default=None, help='index monthly fallback CSV')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # policy & dr007
    policy_dr = None
    if ak is not None:
        try:
            policy_dr = fetch_policy_dr007_via_akshare(start=a.start)
            print("[info] policy/dr007 via AkShare")
        except Exception as e:
            print(f"[warn] AkShare policy/dr007 failed: {e}")
    if policy_dr is None:
        if not a.csv: print("[error] Need --csv for policy & DR007 fallback"); sys.exit(2)
        policy_dr = read_policy_dr007(Path(a.csv), start=a.start)
        print("[info] policy/dr007 via CSV")

    # index
    index_m = fetch_index_monthly(start=a.start, index_csv=(Path(a.index_csv) if a.index_csv else None))

    # merge
    panel = policy_dr.join(index_m, how='inner')
    panel['gap'] = panel['policy'] - panel['dr007']
    panel['gap_ma12'] = panel['gap'].rolling(12).mean()
    panel['rel_gap'] = panel['gap'] / panel['policy']
    panel['ratio'] = panel['dr007'] / panel['policy']

    # save
    panel.to_csv(out / 'gap_chinext_monthly_panel.csv', encoding='utf-8-sig')

    # charts
    make_core_charts(panel, out)
    make_factor_panel(panel, out)
    make_corr_table(panel, out)

    print(f"[done] Updated through {panel.index.max():%Y-%m}. Files saved to: {out}")

if __name__ == '__main__':
    main()
