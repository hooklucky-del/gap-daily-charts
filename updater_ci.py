#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gap 日更脚本（MA3 版，带在线抓取 + CSV 回退 + 当月临时展期 + 调试开关）
- 指数：yfinance 399006.SZ → AkShare 指数 399006 → yfinance 159915.SZ → AkShare ETF 159915
- 政策/DR007：AkShare 多端点；失败回退 CSV；当月缺数据则用上月值临时展期（provisional）
- 输出：charts 与 CSV（gap_vs_chinext_with_ma3.*、rolling_corr_12m.*、
        factor_panel_standardized.*、gap_chinext_monthly_panel.csv、factor_corr_table.csv）
- 调试：环境变量 DEBUG_FETCH="1" 打印端点、列名、head(3)
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# 画图基础设置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']   # Actions 环境可用
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['pdf.fonttype'] = 42

# 调试开关：在工作流或本地设置环境变量 DEBUG_FETCH="1" 即可
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
    """将日序列聚合为“每月最新交易日的值”，索引为当月月末。"""
    if s is None or s.empty:
        return s
    g = s.groupby(s.index.to_period('M')).last()
    g.index = g.index.to_timestamp('M')
    return g


def extend_to_current_month_locf(df: pd.DataFrame, cols=('policy', 'dr007')):
    """若 df 只到上月，则把指定列用最后一个已知值 临时展期 到当月月末（provisional）。"""
    if df is None or df.empty:
        return df, False

    # 防御：把传入的列名转为 list，并过滤不存在的列
    cols_list = [*cols] if not isinstance(cols, list) else cols
    cols_list = [c for c in cols_list if c in df.columns]
    if len(cols_list) == 0:
        print("[note] extend_to_current_month_locf: no target columns found; skipped")
        return df, False

    cur_m_end = pd.Timestamp(datetime.today()).to_period('M').to_timestamp('M')
    last_idx = df.index.max()
    if pd.isna(last_idx) or last_idx >= cur_m_end:
        return df, False  # 已包含当月或索引异常

    extended = df.copy()
    extended.loc[cur_m_end, cols_list] = df[cols_list].iloc[-1].values
    extended = extended.sort_index()
    return extended, True


# ---------------- 政策利率 & DR007：在线优先，失败回退 CSV ----------------
def fetch_policy_dr007_online(start='2014-01-01') -> pd.DataFrame:
    """
    在线获取 DR007 与政策利率；兼容多端点与列名差异。
    开启 DEBUG 时会打印端点、列名与 head(3) 以便排查。
    返回按月末口径（当月取最新交易日值）。
    """
    if ak is None:
        raise RuntimeError("AkShare not installed")

    # ---- DR007：多端点尝试 + 宽松列名匹配 ----
    dr = None
    dr_endpoints = [
        "macro_china_repo_rate",   # 一些版本
        "macro_china_dr_repo",     # 另一些版本
        "repo_rate",               # 旧别名
        "repo_rate_hist_em",       # Eastmoney 历史端点（部分版本）
    ]
    for fname in dr_endpoints:
        if not hasattr(ak, fname):
            dbg(f"{fname}: not found in this akshare version")
            continue
        try:
            f = getattr(ak, fname)
            try:
                tmp = f()
            except TypeError:
                tmp = f(
                    start_date=start.replace('-', ''),
                    end_date=datetime.today().strftime("%Y%m%d")
                )
            tmp.columns = [str(c).strip() for c in tmp.columns]
            dbg(f"{fname}: columns={list(tmp.columns)}")
            try:
                dbg(f"{fname} head:\n{tmp.head(3).to_string(index=False)}")
            except Exception:
                pass

            dcol = next((c for c in tmp.columns if '日期' in c or 'date' in str(c).lower() or '时间' in c), None)
            cand_cols = [c for c in tmp.columns
                         if any(k in str(c).upper() for k in ['DR007', '7天', '7D'])
                         and 'R007' not in str(c).upper()]
            dbg(f"{fname}: dcol={dcol}, cand_cols={cand_cols[:3]}")
            if not dcol or not cand_cols:
                dbg(f"{fname}: no suitable columns; numeric={list(tmp.select_dtypes(include='number').columns)}")
                continue

            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            ser = pd.to_numeric(tmp[cand_cols[0]], errors='coerce').dropna()
            if not ser.empty:
                dr = ser
                dbg(f"{fname}: DR007 selected col={cand_cols[0]}, points={len(ser)}")
                break
        except Exception as e:
            dbg(f"{fname}: error={e}")

    if dr is None:
        raise RuntimeError("AkShare: DR007 endpoint(s) not usable")
    dr_m = to_monthly_last(dr).rename('dr007')

    # ---- 政策利率：OMO 7D → OMO中标 → MLF；宽松列名匹配 ----
    pol = None
    pol_candidates = [
        ("macro_china_omo_daily",  ["操作利率", "7天", "7D"]),
        ("macro_china_omo",        ["中标利率", "7天", "7D"]),
        ("macro_china_mlf",        ["利率", "中期借贷便利"]),
    ]
    for fname, hints in pol_candidates:
        if not hasattr(ak, fname):
            dbg(f"{fname}: not found in this akshare version")
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
                dbg(f"{fname}: no date column")
                continue
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()

            col = next((c for c in tmp.columns if any(h in str(c) for h in hints)), None)
            if col is None:
                num_cols = tmp.select_dtypes(include='number').columns
                col = num_cols[-1] if len(num_cols) else None
            if col is None:
                dbg(f"{fname}: no numeric column available")
                continue

            pol = to_monthly_last(pd.to_numeric(tmp[col], errors='coerce')).rename('policy')
            dbg(f"{fname}: policy selected col={col}, points={len(pol.dropna())}")
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
    """读取本地 CSV（第一列为日期，还需包含列：政策利率、DR007）。"""
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
    """优先在线，其次 CSV。"""
    try:
        return fetch_policy_dr007_online(start=start)
    except Exception as e_online:
        print(f"[warn] online policy/dr007 failed: {e_online}")
        if csv_path.exists():
            return read_policy_dr007_csv(csv_path, start=start)
        raise


# ---------------- 创业板指数：多源兜底 ----------------
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


def make_core_charts(panel: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5), dpi=150); ax = plt.gca()
    ax.plot(panel.index, zscore(panel['gap']), label='Gap Z', linewidth=2.0, color='#F59E0B')
    ax.plot(panel.index, zscore(panel['level_base100']), label='ChiNext Z', linewidth=2.0, color='#EF4444')
    ax.plot(panel.index, zscore(panel['gap'].rolling(3).mean()), label='Gap MA3 Z', linewidth=2.4, color='#1D4ED8')
    ax.set_title(f"Gap vs ChiNext (Z) + Gap MA3 | {panel.index.min():%Y-%m}–{panel.index.max():%Y-%m}")
    ax.set_ylabel('Z-score'); ax.set_xlabel('Date'); ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax.legend()
    for ext in ['png','svg','pdf']:
        fig.savefig(out_dir / f'gap_vs_chinext_with_ma3.{ext}', bbox_inches='tight')
    plt.close(fig)

    roll = panel['gap'].rolling(12).corr(panel['ret_m'])
    fig2 = plt.figure(figsize=(12,5), dpi=150); ax2 = plt.gca()
    ax2.plot(roll.index, roll, label='Rolling corr (12m)', color='#F59E0B')
    ax2.axhline(0.0, linestyle='--', linewidth=1.0, color='#F59E0B', alpha=0.6, label='0 baseline')
    ax2.set_title('12m Rolling Correlation: Gap vs ChiNext monthly return')
    ax2.set_ylabel('Correlation (12m)'); ax2.set_xlabel('Date')
    ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax2.legend()
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
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5); ax.legend()
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
    ap.add_argument('--csv', default="data/中国 DR007 政策利率.csv")  # CSV 作为回退
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--margin-csv', default=None)  # 预留：你的两融占比 CSV
    ap.add_argument('--pe-csv', default=None)      # 预留：你的估值 CSV
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 政策/DR007：在线优先，失败才用 CSV；必要时临时展期到当月
    policy_dr = get_policy_dr007(start=a.start, csv_path=Path(a.csv))
    policy_dr, provisional = extend_to_current_month_locf(policy_dr, cols=['policy','dr007'])
    if provisional:
        print("[note] policy/dr007 for current month is provisional (LOCF from last month)")

    # 指数：在线多源兜底
    index_m   = fetch_index_monthly(start=a.start)

    # 合并
    panel = policy_dr.join(index_m, how='inner')
    if panel.empty:
        panel = policy_dr.join(index_m, how='outer').dropna(subset=['policy','dr007','ret_m','level_base100'])
    if panel.empty:
        raise ValueError("合并后为空：请检查数据日期交集")

    # 指标（MA3）
    panel['gap']      = panel['policy'] - panel['dr007']
    panel['gap_ma3']  = panel['gap'].rolling(3).mean()
    panel['rel_gap']  = panel['gap'] / panel['policy']
    panel['ratio']    = panel['dr007'] / panel['policy']

    # 导出 & 图
    panel.to_csv(out / 'gap_chinext_monthly_panel.csv', encoding='utf-8-sig')
    make_core_charts(panel, out)
    make_factor_panel(panel, out)
    make_corr_table(panel, out)
    print(f"[done] Updated through {panel.index.max():%Y-%m}. Files saved to: {out}")


if __name__ == '__main__':
    main()
