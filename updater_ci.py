#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gap 日更脚本（MA3 版）— DR007 优先抓取中国货币网（英文页），失败回退 AkShare / CSV；时间轴为“月份”刻度。
- DR007：先尝试 https://www.chinamoney.com.cn/english/mdtqapprp/ （英文：Pledged Repo Avg Price & Rate）
         尝试解析页面内表格或脚本中的数据；若站点为 JS 渲染导致抓不到，则自动回退 AkShare，再次失败则回退 CSV。
- 政策利率：仍按前版（优先 PBoC LPR / AkShare；支持 CSV 回退）。
- 指数：399006 多源兜底。
"""

import os
import re
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

# ---- optional deps ----
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import akshare as ak
except Exception:
    ak = None

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

DEBUG = os.getenv("DEBUG_FETCH", "0") == "1"


def dbg(msg: str):
    if DEBUG:
        print(f"[debug] {msg}")


def to_monthly_last(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    g = s.groupby(s.index.to_period('M')).last()
    g.index = g.index.to_timestamp('M')
    return g


def extend_to_current_month_locf(df: pd.DataFrame, cols=('policy', 'dr007')):
    if df is None or df.empty:
        return df, False
    cols = [c for c in (list(cols) if not isinstance(cols, list) else cols) if c in df.columns]
    if not cols:
        return df, False
    cur_m_end = pd.Timestamp(datetime.today()).to_period('M').to_timestamp('M')
    last_idx = df.index.max()
    if pd.isna(last_idx) or last_idx >= cur_m_end:
        return df, False
    dfx = df.copy()
    dfx.loc[cur_m_end, cols] = df[cols].iloc[-1].values
    dfx = dfx.sort_index()
    return dfx, True


# ---------------------- PBoC LPR（政策利率）抓取（同前版） ----------------------
def fetch_lpr_series_from_pbc(list_url: str, start='2014-01-01', max_pages: int = 12) -> pd.Series:
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("需要 requests 与 bs4 才能抓取 PBoC LPR")

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36"
    })

    def parse_detail(url: str):
        r = sess.get(url, timeout=15)
        r.encoding = r.apparent_encoding or 'utf-8'
        html = r.text
        # 日期：优先匹配 “YYYY年M月D日”，退化到 “YYYY年M月”
        m_date = re.search(r'(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日', html)
        day = int(m_date.group(3)) if m_date else 1
        if not m_date:
            m_date = re.search(r'(20\d{2})年\s*(\d{1,2})月', html)
        if not m_date:
            raise ValueError("未能从公告页解析日期")
        year, month = int(m_date.group(1)), int(m_date.group(2))
        dt = pd.Timestamp(year=year, month=month, day=day)

        # 解析 1年期 LPR
        m_lpr = re.search(r'(?:1年期|一年期)\s*LPR\s*为\s*([0-9.]+)\s*%', html)
        if not m_lpr:
            nums = re.findall(r'([0-9.]+)\s*%', html)
            val = float(nums[0]) if nums else None
        else:
            val = float(m_lpr.group(1))
        if val is None:
            raise ValueError("未能解析 LPR 数值")
        return dt, val

    # 目录页翻页并收集详情链接（较宽松）
    items = []
    page_url = list_url
    for _ in range(max_pages):
        resp = sess.get(page_url, timeout=15)
        resp.encoding = resp.apparent_encoding or 'utf-8'
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, 'lxml')
        for a in soup.find_all('a', href=True):
            text = (a.get_text() or '').strip()
            if 'LPR' in text or '报价利率' in text:
                items.append(requests.compat.urljoin(page_url, a['href']))
        nxt = soup.find('a', string=re.compile(r'下一页|下页|>'))
        if nxt and nxt.get('href'):
            page_url = requests.compat.urljoin(page_url, nxt['href'])
        else:
            break
    # 解析详情
    recs = []
    for url in list(dict.fromkeys(items))[:240]:
        try:
            dt, val = parse_detail(url)
            recs.append((dt, val))
        except Exception as e:
            dbg(f"LPR detail parse fail: {url} -> {e}")
            continue

    if not recs:
        raise RuntimeError("PBoC LPR 页面解析为空")

    df = pd.DataFrame(recs, columns=['date','lpr1y'])
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    df['month_end'] = df['date'].dt.to_period('M').dt.to_timestamp('M')
    s = df.set_index('month_end')['lpr1y']
    s = s[s.index >= pd.to_datetime(start)]
    s = s.groupby(s.index).last()
    s.name = 'policy'
    return s


# ---------------------- ChinaMoney 英文页 DR007 抓取 ----------------------
def fetch_dr007_from_chinamoney_en(page_url: str, start='2014-01-01') -> pd.Series:
    """
    尝试从 https://www.chinamoney.com.cn/english/mdtqapprp/ 抓取 DR007 日频，并聚合到“每月最新交易日”
    说明：该页多为 JS 渲染。此函数先尝试直抓 HTML 中表格/脚本内 JSON；抓不到则抛异常交由上层回退。
    """
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("需要 requests 与 bs4 才能抓取 ChinaMoney 英文页")

    r = requests.get(page_url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36"
    })
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, 'lxml')

    # 1) 尝试直接解析页面表格（若有静态渲染）
    for table in soup.find_all('table'):
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        if not headers:
            # 有些表无 thead
            headers = [td.get_text(strip=True) for td in table.find('tr').find_all('td')] if table.find('tr') else []
        hdr = ''.join(headers).upper()
        if ('DR007' in hdr) or ('REPO' in hdr and '7' in hdr):
            dates, vals = [], []
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 2:
                    continue
                txt_row = [td.get_text(strip=True) for td in tds]
                row = ' '.join(txt_row)
                # 日期
                m_d = re.search(r'(20\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2})', row)
                if not m_d:
                    continue
                dt = pd.to_datetime(m_d.group(1), errors='coerce')
                # 数值（%）
                m_v = re.search(r'([0-9.]+)\s*%?', row)
                if dt is not None and not pd.isna(dt) and m_v:
                    dates.append(dt)
                    vals.append(float(m_v.group(1)))
            if dates:
                ser = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
                ser = ser[ser.index >= pd.to_datetime(start)]
                return to_monthly_last(ser).rename('dr007')

    # 2) 尝试解析脚本中的 JSON（常见形如 var option = {... series:[{name:'DR007', data:[['2025-09-05',1.44], ...]}]} ）
    m = re.search(r'(?s)\{[^{}]*"DR007"[^{}]*\}', html)
    if m:
        # 粗提取所有 [ 'YYYY-mm-dd', number ] 对
        arr = re.findall(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})"\s*,\s*([0-9.]+)', html)
        if arr:
            dates = pd.to_datetime([d for d, _ in arr], errors='coerce')
            vals = [float(v) for _, v in arr]
            ser = pd.Series(vals, index=dates).sort_index()
            ser = ser[ser.index >= pd.to_datetime(start)]
            return to_monthly_last(ser).rename('dr007')

    raise RuntimeError("ChinaMoney 英文页未解析到 DR007（可能为 JS 动态加载）")


# ---------------------- 组合获取（政策 + DR007） ----------------------
def get_policy_dr007(start='2014-01-01',
                     csv_path=Path("data/中国 DR007 政策利率.csv"),
                     lpr_url="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/index.html",
                     dr007_en_url="https://www.chinamoney.com.cn/english/mdtqapprp/") -> pd.DataFrame:
    # 政策：先抓 PBoC LPR，失败时 AkShare → CSV
    policy = None
    try:
        policy = fetch_lpr_series_from_pbc(lpr_url, start=start)
        print("[info] policy via PBoC LPR")
    except Exception as e:
        print(f"[warn] PBoC LPR 抓取失败：{e}")
        # 退回 AkShare
        if ak is not None:
            pol = None
            for fname, hints in [
                ("macro_china_omo_daily",  ["操作利率", "7天", "7D"]),
                ("macro_china_omo",        ["中标利率", "7天", "7D"]),
                ("macro_china_mlf",        ["利率", "中期借贷便利"]),
            ]:
                if hasattr(ak, fname):
                    try:
                        f = getattr(ak, fname)
                        tmp = f()
                        tmp.columns = [str(c).strip() for c in tmp.columns]
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
                        break
                    except Exception:
                        continue
            if pol is not None:
                policy = pol
                print("[info] policy via AkShare")
    if policy is None and Path(csv_path).exists():
        # CSV 回退（列：政策利率、DR007）
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception:
                continue
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
        policy = pd.to_numeric(df.get('政策利率'), errors='coerce').resample('D').ffill().groupby(lambda x: x.to_period('M')).last()
        policy.index = policy.index.to_timestamp('M')
        print("[info] policy via CSV")

    if policy is None:
        raise RuntimeError("policy rate not available via PBoC/AkShare/CSV")

    # DR007：ChinaMoney 英文页 → AkShare → CSV
    dr007 = None
    try:
        dr007 = fetch_dr007_from_chinamoney_en(dr007_en_url, start=start)
        print("[info] dr007 via ChinaMoney (EN)")
    except Exception as e:
        print(f"[warn] ChinaMoney EN DR007 抓取失败：{e}")
        # AkShare 多端点
        if ak is not None:
            for fname in ["macro_china_repo_rate","macro_china_dr_repo","repo_rate","repo_rate_hist_em","repo_rate_em","repo_rate_ths","bond_repo_rate_ths"]:
                if not hasattr(ak, fname):
                    continue
                try:
                    f = getattr(ak, fname)
                    try:
                        tmp = f()
                    except TypeError:
                        tmp = f(start_date=start.replace('-', ''), end_date=datetime.today().strftime("%Y%m%d"))
                    tmp.columns = [str(c).strip() for c in tmp.columns]
                    dcol = next((c for c in tmp.columns if '日期' in c or 'date' in str(c).lower() or '时间' in c), None)
                    cand = [c for c in tmp.columns if any(k in str(c).upper() for k in ['DR007','7天','7D']) and 'R007' not in str(c).upper()]
                    if not dcol or not cand:
                        continue
                    tmp[dcol] = pd.to_datetime(tmp[dcol], errors='coerce')
                    tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
                    ser = pd.to_numeric(tmp[cand[0]], errors='coerce').dropna()
                    if not ser.empty:
                        dr007 = to_monthly_last(ser).rename('dr007')
                        break
                except Exception:
                    continue
        if dr007 is None and Path(csv_path).exists():
            # CSV 回退
            for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    break
                except Exception:
                    continue
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).sort_values(by=date_col).set_index(date_col)
            dr = pd.to_numeric(df.get('DR007'), errors='coerce').resample('D').ffill().groupby(lambda x: x.to_period('M')).last()
            dr.index = dr.index.to_timestamp('M')
            dr007 = dr.rename('dr007')
            print("[info] dr007 via CSV")

    if dr007 is None:
        raise RuntimeError("DR007 not available via ChinaMoney/AkShare/CSV")

    rng = pd.date_range(start=start, end=datetime.today(), freq='M')
    out = pd.DataFrame(index=rng).join(policy.rename('policy'), how='left').join(dr007.rename('dr007'), how='left')
    return out


# ---------------------- 创业板指数（多源） ----------------------
def fetch_index_monthly(start='2014-01-01') -> pd.DataFrame:
    def from_close(close: pd.Series) -> pd.DataFrame:
        mclose = to_monthly_last(close).dropna()
        mret   = mclose.pct_change()
        level  = (1 + mret.fillna(0)).cumprod() * 100.0
        return pd.concat([
            mclose.rename('close_m'), mret.rename('ret_m'), level.rename('level_base100')
        ], axis=1)

    # yfinance 399006.SZ → akshare 399006 → yfinance 159915 → akshare 159915
    if yf is not None:
        try:
            data = yf.download("399006.SZ", start=start, progress=False, auto_adjust=True, threads=False)
            if data is not None and not data.empty and 'Close' in data:
                print("[info] index via yfinance (399006.SZ)")
                return from_close(data['Close'])
        except Exception as e:
            print(f"[warn] yfinance 399006.SZ failed: {e}")

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

    if yf is not None:
        try:
            data = yf.download("159915.SZ", start=start, progress=False, auto_adjust=True, threads=False)
            if data is not None and not data.empty and 'Close' in data:
                print("[info] index via yfinance (159915.SZ proxy)")
                return from_close(data['Close'])
        except Exception as e:
            print(f"[warn] yfinance 159915.SZ failed: {e}")

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
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)


def make_core_charts(panel: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5), dpi=150); ax = plt.gca()
    ax.plot(panel.index, zscore(panel['gap']), label='Gap Z', linewidth=2.0, color='#F59E0B')
    ax.plot(panel.index, zscore(panel['level_base100']), label='ChiNext Z', linewidth=2.0, color='#EF4444')
    ax.plot(panel.index, zscore(panel['gap'].rolling(3).mean()), label='Gap MA3 Z', linewidth=2.4, color='#1D4ED8')
    ax.set_title(f"Gap vs ChiNext (Z) + Gap MA3 | {panel.index.min():%Y-%m}–{panel.index.max():%Y-%m}")
    ax.set_ylabel('Z-score'); ax.set_xlabel('Date'); _format_month_axis(ax); ax.legend()
    for ext in ['png','svg','pdf']:
        fig.savefig(out_dir / f'gap_vs_chinext_with_ma3.{ext}', bbox_inches='tight')
    plt.close(fig)

    roll = panel['gap'].rolling(12).corr(panel['ret_m'])
    fig2 = plt.figure(figsize=(12,5), dpi=150); ax2 = plt.gca()
    ax2.plot(roll.index, roll, label='Rolling corr (12m)', color='#F59E0B')
    ax2.axhline(0.0, linestyle='--', linewidth=1.0, color='#F59E0B', alpha=0.6, label='0 baseline')
    ax2.set_title('12m Rolling Correlation: Gap vs ChiNext monthly return')
    ax2.set_ylabel('Correlation (12m)'); ax2.set_xlabel('Date'); _format_month_axis(ax2); ax2.legend()
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
    ax.set_title('Factor panel (standardized)'); ax.set_ylabel('Z-score'); ax.set_xlabel('Date'); _format_month_axis(ax); ax.legend()
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
    ap.add_argument('--lpr-url', default="https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/index.html",
                    help="PBoC LPR 目录页 URL")
    ap.add_argument('--dr007-en-url', default="https://www.chinamoney.com.cn/english/mdtqapprp/",
                    help="ChinaMoney 英文 DR007 页 URL")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    policy_dr = get_policy_dr007(start=a.start, csv_path=Path(a.csv),
                                 lpr_url=a.lpr_url, dr007_en_url=a.dr007_en_url)
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
