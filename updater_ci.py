#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gap 日更脚本（MA3 版，改进：
- 更强的 AkShare 端点回退（DR007 / 政策利率）；
- X 轴显示“月份”刻度与格式；
- 仍支持 CSV 回退与当月临时展期；
）
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)

# 画图基础设置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['pdf.fonttype'] = 42

DEBUG = os.getenv("DEBUG_FETCH", "0") == "1"


def dbg(msg: str):
    if DEBUG:
        try:
            print(f"[debug] {msg}")
        except Exception:
            pass


def try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


yf = try_import("yfinance")
ak = try_import("akshare")


def to_monthly_last(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    g = s.groupby(s.index.to_period('M')).last()
    g.index = g.index.to_timestamp('M')
    return g


def extend_to_current_month_locf(df: pd.DataFrame, cols=('policy', 'dr007')):
    if df is None or df.empty:
        return df, False
    cols_list = [*cols] if not isinstance(cols, list) else cols
    cols_list = [c for c in cols_list if c in df.columns]
    if not cols_list:
        print("[note] extend_to_current_month_locf: no target columns found; skipped")
        return df, False
    cur_m_end = pd.Timestamp(datetime.today()).to_period('M').to_timestamp('M')
    last_idx = df.index.max()
    if pd.isna(last_idx) or last_idx >= cur_m_end:
        return df, False
    dfx = df.copy()
    dfx.loc[cur_m_end, cols_list] = df[cols_list].iloc[-1].values
    dfx = dfx.sort_index()
    return dfx, True


# ----------- 在线获取政策/DR007：多端点 + 宽松列名匹配 -----------
def fetch_policy_dr007_online(start='2014-01-01') -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("AkShare not installed")

    # DR007 多端点
    dr = None
    dr_endpoints = [
        "macro_china_repo_rate",
        "macro_china_dr_repo",
        "repo_rate",
        "repo_rate_hist_em",     # Eastmoney 历史端点（部分版本有）
        "repo_rate_em",          # 有些版本可能有
        "repo_rate_ths",         # 可能的同花顺端点
        "bond_repo_rate_ths",    # 可能的同花顺债券回购端点
    ]
    for fname in dr_endpoints:
        if not hasattr(ak, fname):
            dbg(f"{fname}: not found")
            continue
        try:
            f = getattr(ak, fname)
            try:
                tmp = f()
            except TypeError:
                tmp = f(start_date=start.replace('-', ''), end_date=datetime.today().strftime("%Y%m%d"))
            tmp.columns = [str(c).strip() for c in tmp.columns]
            dbg(f"{fname}: columns={list(tmp.columns)}")
            try:
                dbg(f"{fname} head:\n{tmp.head(3).to_string(index=False)}")
            except Exception:
                pass
            dcol = next((c for c in tmp.columns if '日期' in c or 'date' in str(c).lower() or '时间' in c), None)
            cand = [c for c in tmp.columns
                    if any(k in str(c).upper() for k in ['DR007','7天','7D'])
                    and 'R007' not in str(c).upper()]
            if not dcol or not cand:
                continue
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            ser = pd.to_numeric(tmp[cand[0]], errors='coerce').dropna()
            if not ser.empty:
                dr = ser
                dbg(f"{fname}: chose {cand[0]} points={len(ser)}")
                break
        except Exception as e:
            dbg(f"{fname}: error={e}")

    if dr is None:
        raise RuntimeError("AkShare: DR007 endpoint(s) not usable")
    dr_m = to_monthly_last(dr).rename('dr007')

    # 政策利率多端点
    pol = None
    pol_candidates = [
        ("macro_china_omo_daily",  ["操作利率", "7天", "7D"]),
        ("macro_china_omo",        ["中标利率", "7天", "7D"]),
        ("macro_china_mlf",        ["利率", "中期借贷便利"]),
        ("macro_china_loan_prime_rate", ["1年", "一年", "LPR"]),  # 兜底，若找不到 OMO/MLF
    ]
    for fname, hints in pol_candidates:
        if not hasattr(ak, fname):
            dbg(f"{fname}: not found")
            continue
        try:
            f = getattr(ak, fname)
            tmp = f()
            tmp.columns = [str(c).strip() for c in tmp.columns]
            dbg(f"{fname}: columns={list(tmp.columns)}")
            try:
                dbg(f"{fname} head:\n{tmp.head(3).to_string(index=False)}")
            except Exception:
                pass
            dcol = next((c for c in tmp.columns if '日期' in c or 'date' in str(c).lower() or '时间' in c), None)
            if not dcol:
                continue
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            col = next((c for c in tmp.columns if any(h in str(c) for h in hints)), None)
            if col is None:
                num = tmp.select_dtypes(include='number').columns
                col = num[-1] if len(num) else None
            if col is None:
                continue
            pol = to_monthly_last(pd.to_numeric(tmp[col], errors='coerce')).rename('policy')
            dbg(f"{fname}: chose {col} points={len(pol.dropna())}")
            break
        except Exception as e:
            dbg(f"{fname}: error={e}")

    if pol is None:
        raise RuntimeError("AkShare: policy rate endpoint(s) not usable")

    rng = pd.date_range(start=start, end=datetime.today(), freq='M')
    out = pd.DataFrame(index=rng).join(pol, how='left').join(dr_m, how='left')
    print("[info] policy/dr007 via AkShare (robust)")
    return out


def read_policy_dr007_csv(csv_path: Path, start='2014-01-01') -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    if df.shape[0] == 0:
        raise ValueError("政策利率/DR007 CSV 为空")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
    if '政策利率' not in df.columns or 'DR007' not in df.columns:
        raise ValueError("CSV 需要包含列：政策利率、DR007")
    pol = pd.to_numeric(df['政策利率'], errors='coerce').resample('D').ffill().groupby(lambda x: x.to_period('M')).last()
    dr  = pd.to_numeric(df['DR007'],   errors='coerce').resample('D').ffill().groupby(lambda x: x.to_period('M')).last()
    pol.index = pol.index.to_timestamp('M')
    dr.index  = dr.index.to_timestamp('M')
    rng = pd.date_range(start=start, end=datetime.today(), freq='M')
    out = pd.DataFrame(index=rng)
    out['policy'] = pol
    out['dr007']  = dr
    print("[info] policy/dr007 via CSV")
    return out


def get_policy_dr007(start='2014-01-01', csv_path=Path("data/中国 DR007 政策利率.csv")) -> pd.DataFrame:
    try:
        return fetch_policy_dr007_online(start=start)
    except Exception as e_online:
        print(f"[warn] online policy/dr007 failed: {e_online}")
        if csv_path.exists():
            return read_policy_dr007_csv(csv_path, start=start)
        raise


# ----------- 创业板指数月频：多源兜底 -----------
def fetch_index_monthly(start='2014-01-01') -> pd.DataFrame:
    def from_close(close: pd.Series) -> pd.DataFrame:
        mclose = to_monthly_last(close).dropna()
        mret   = mclose.pct_change()
        level  = (1 + mret.fillna(0)).cumprod() * 100.0
        return pd.concat([
            mclose.rename('close_m'),
            mret.rename('ret_m'),
            level.rename('level_base100')
        ], axis=1)

    # 1) yfinance 399006.SZ
    if yf is not None:
        try:
            data = yf.download("399006.SZ", start=start, progress=False, auto_adjust=True, threads=False)
            if data is not None and not data.empty and 'Close' in data:
                print("[info] index via yfinance (399006.SZ)")
                return from_close(data['Close'])
        except Exception as e:
            print(f"[warn] yfinance 399006.SZ failed: {e}")

    # 2) AkShare 指数 399006
    if ak is not None and hasattr(ak, "index_zh_a_hist"):
        try:
            tmp = ak.index_zh_a_hist(symbol="399006", period="daily",
                                     start_date=start.replace('-', ''),
                                     end_date=datetime.today().strftime("%Y%m%d"))
            dcol = next((c for c in tmp.columns if '日期' in c or 'date' in str(c).lower()), None)
            ccol = next((c for c in tmp.columns if '收盘' in c or 'close' in str(c).lower()), None)
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            close = pd.to_numeric(tmp[ccol], errors='coerce').dropna()
            if not close.empty:
                print("[info] index via akshare index_zh_a_hist (399006)")
                return from_close(close)
        except Exception as e:
            print(f"[warn] akshare 399006 failed: {e}")

    # 3) yfinance 159915.SZ（ETF 代理）
    if yf is not None:
        try:
            data = yf.download("159915.SZ", start=start, progress=False, auto_adjust=True, threads=False)
            if data is not None and not data.empty and 'Close' in data:
                print("[info] index via yfinance (159915.SZ proxy)")
                return from_close(data['Close'])
        except Exception as e:
            print(f"[warn] yfinance 159915.SZ failed: {e}")

    # 4) AkShare ETF 159915
    if ak is not None and hasattr(ak, "fund_etf_hist_em"):
        try:
            tmp = ak.fund_etf_hist_em(symbol="159915", period="daily",
                                      start_date=start.replace('-', ''),
                                      end_date=datetime.today().strftime("%Y%m%d"))
            dcol = next((c for c in tmp.columns if '日期' in c or 'date' in str(c).lower()), None)
            ccol = next((c for c in tmp.columns if '收盘' in c or 'close' in str(c).lower()), None)
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            close = pd.to_numeric(tmp[ccol], errors='coerce').dropna()
            if not close.empty:
                print("[info] index via akshare fund_etf_hist_em (159915 proxy)")
                return from_close(close)
        except Exception as e:
            print(f"[warn] akshare ETF 159915 failed: {e}")

    raise RuntimeError("指数获取失败：yfinance 与 AkShare 均不可用")


def zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    std = x.std(ddof=1)
    return (x - x.mean()) / std if std and not np.isnan(std) else x * 0.0


def fwd_cum_simple(r: pd.Series, k: int) -> pd.Series:
    rv = r.values
    out = np.full_like(rv, np.nan, dtype=float)
    for i in range(len(rv)-k):
        w = rv[i+1:i+1+k]
        out[i] = np.nan if np.any(np.isnan(w)) else np.prod(1+w)-1.0
    return pd.Series(out, index=r.index)


def _format_month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   # 每 3 个月一个主刻度
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))   # 显示到月份
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))   # 每月一个次刻度
    for label in ax.get_xticklabels(which='major'):
        label.set_rotation(0)
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)


def make_core_charts(panel: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5), dpi=150); ax = plt.gca()
    ax.plot(panel.index, zscore(panel['gap']), label='Gap Z', linewidth=2.0, color='#F59E0B')
    ax.plot(panel.index, zscore(panel['level_base100']), label='ChiNext Z', linewidth=2.0, color='#EF4444')
    ax.plot(panel.index, zscore(panel['gap'].rolling(3).mean()), label='Gap MA3 Z', linewidth=2.4, color='#1D4ED8')
    ax.set_title(f"Gap vs ChiNext (Z) + Gap MA3 | {panel.index.min():%Y-%m}–{panel.index.max():%Y-%m}")
    ax.set_ylabel('Z-score'); ax.set_xlabel('Date')
    _format_month_axis(ax)
    ax.legend()
    for ext in ['png','svg','pdf']:
        fig.savefig(out_dir / f'gap_vs_chinext_with_ma3.{ext}', bbox_inches='tight')
    plt.close(fig)

    roll = panel['gap'].rolling(12).corr(panel['ret_m'])
    fig2 = plt.figure(figsize=(12,5), dpi=150); ax2 = plt.gca()
    ax2.plot(roll.index, roll, label='Rolling corr (12m)', color='#F59E0B')
    ax2.axhline(0.0, linestyle='--', linewidth=1.0, color='#F59E0B', alpha=0.6, label='0 baseline')
    ax2.set_title('12m Rolling Correlation: Gap vs ChiNext monthly return')
    ax2.set_ylabel('Correlation (12m)'); ax2.set_xlabel('Date')
    _format_month_axis(ax2)
    ax2.legend()
    for ext in ['png','svg','pdf']:
        fig2.savefig(out_dir / f'rolling_corr_12m.{ext}', bbox_inches='tight')
    plt.close(fig2)


def make_factor_panel(panel: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(12,5), dpi=150); ax = plt.gca()
    if 'margin_ratio' in panel:
        ax.plot(panel.index, zscore(panel['margin_ratio']), label='Margin share (Z)', linewidth=2.0)
    if 'pe_percentile' in panel:
        ax.plot(panel.index, -zscore(panel['pe_percentile']), label='-Valuation pct (Z)', linewidth=2.0)
    if {'policy','dr007'}.issubset(panel.columns):
        ax.plot(panel.index, zscore((panel['policy']-panel['dr007'])/panel['policy']), label='RelGap (Z)', linewidth=2.0)
    ax.set_title('Factor panel (standardized)'); ax.set_ylabel('Z-score'); ax.set_xlabel('Date')
    _format_month_axis(ax)
    ax.legend()
    for ext in ['png','svg','pdf']:
        fig.savefig(out_dir / f'factor_panel_standardized.{ext}', bbox_inches='tight')
    plt.close(fig)


def make_corr_table(panel: pd.DataFrame, out_dir: Path):
    rows = []
    factors = [c for c in ['gap','rel_gap','ratio','margin_ratio','pe_percentile'] if c in panel.columns]
    for f in factors:
        for k in [0,3,6,12]:
            if k==0:
                dfk = panel[[f,'ret_m']].dropna()
                corr = dfk[f].corr(dfk['ret_m'])
            else:
                fwd = fwd_cum_simple(panel['ret_m'], k)
                dfk = pd.concat([panel[f], fwd], axis=1).dropna()
                corr = dfk.iloc[:,0].corr(dfk.iloc[:,1])
            rows.append({'factor': f, 'k_months': k, 'corr': float(corr)})
    pd.DataFrame(rows).pivot(index='factor', columns='k_months', values='corr').to_csv(out_dir / 'factor_corr_table.csv', encoding='utf-8-sig')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--csv', default="data/中国 DR007 政策利率.csv")
    ap.add_argument('--start', default='2014-01-01')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    policy_dr = get_policy_dr007(start=a.start, csv_path=Path(a.csv))
    policy_dr, provisional = extend_to_current_month_locf(policy_dr, cols=['policy','dr007'])
    if provisional:
        print("[note] policy/dr007 for current month is provisional (LOCF from last month)")

    index_m = fetch_index_monthly(start=a.start)

    panel = policy_dr.join(index_m, how='inner')
    if panel.empty:
        panel = policy_dr.join(index_m, how='outer').dropna(subset=['policy','dr007','ret_m','level_base100'])
    if panel.empty:
        raise ValueError("合并后为空：请检查数据日期交集")

    panel['gap']      = panel['policy'] - panel['dr007']
    panel['gap_ma3']  = panel['gap'].rolling(3).mean()
    panel['rel_gap']  = panel['gap'] / panel['policy']
    panel['ratio']    = panel['dr007'] / panel['policy']

    panel.to_csv(out / 'gap_chinext_monthly_panel.csv', encoding='utf-8-sig')
    make_core_charts(panel, out)
    make_factor_panel(panel, out)
    make_corr_table(panel, out)
    print(f"[done] Updated through {panel.index.max():%Y-%m}. Files saved to: {out}")


if __name__ == '__main__':
    main()
