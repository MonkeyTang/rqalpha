# -*- coding: utf-8 -*-
"""
交易回测可视化工具（TradingView Lightweight Charts）
====================================================

用法
----
回测后自动启动（推荐）::

    rqalpha run -f strategies/my_strategy.py ... --view

命令行独立运行::

    python -m rqalpha.mod.rqalpha_mod_trade_viewer.viewer --trades /tmp/result.pkl

回测完成后直接传入 result dict::

    from rqalpha.mod.rqalpha_mod_trade_viewer.viewer import launch_viewer
    launch_viewer(result)

依赖::

    pip install flask pandas numpy
"""

from __future__ import annotations

import json
import argparse
from datetime import timezone, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ════════════════════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════════════════════

_HK_TZ = timezone(timedelta(hours=8))
_DEFAULT_CACHE_DIR = Path.home() / '.rqalpha' / 'futu_cache'

_BAR_DTYPE = np.dtype([
    ('datetime',       '<u8'),
    ('open',           '<f4'),
    ('high',           '<f4'),
    ('low',            '<f4'),
    ('close',          '<f4'),
    ('volume',         '<i8'),
    ('total_turnover', '<f8'),
])

# ════════════════════════════════════════════════════════════════════════════
#  数据层
# ════════════════════════════════════════════════════════════════════════════

def load_trades(source) -> pd.DataFrame:
    if isinstance(source, dict):
        df = source.get('trades', pd.DataFrame())
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        p = Path(source)
        if p.suffix == '.pkl':
            result = pd.read_pickle(p)
            df = result.get('trades', pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
        else:
            df = pd.read_csv(source)

    if df.empty:
        return df

    if df.index.name == 'datetime':
        df = df.reset_index(drop=True) if 'datetime' in df.columns else df.reset_index()

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['side'] = df['side'].str.upper()
    return df.sort_values('datetime').reset_index(drop=True)


def load_portfolio(source) -> Optional[pd.DataFrame]:
    """从 result dict 或 pkl 文件中提取 portfolio DataFrame。"""
    if isinstance(source, dict):
        pf = source.get('portfolio')
    else:
        p = Path(source)
        if p.suffix == '.pkl':
            result = pd.read_pickle(p)
            pf = result.get('portfolio') if isinstance(result, dict) else None
        else:
            return None

    if pf is None or not isinstance(pf, pd.DataFrame) or pf.empty:
        return None
    if 'unit_net_value' not in pf.columns:
        return None
    return pf


def decode_bar_array(bars) -> pd.DataFrame:
    """将 numpy 结构化数组（BAR_DTYPE）转换为 DataFrame。

    datetime 字段为 uint64，格式 YYYYMMDDHHMMSS。
    """
    dt_int = bars['datetime'].astype('i8')
    sec    = dt_int % 100;  dt_int //= 100
    minute = dt_int % 100;  dt_int //= 100
    hour   = dt_int % 100;  dt_int //= 100
    day    = dt_int % 100;  dt_int //= 100
    month  = dt_int % 100
    year   = dt_int // 100

    dts = pd.to_datetime({
        'year': year, 'month': month, 'day': day,
        'hour': hour, 'minute': minute, 'second': sec,
    })

    # 适配不同字段名：total_turnover 或 turnover
    if 'total_turnover' in bars.dtype.names:
        turnover = bars['total_turnover'].astype(float)
    elif 'turnover' in bars.dtype.names:
        turnover = bars['turnover'].astype(float)
    else:
        turnover = np.zeros(len(bars), dtype=float)

    return pd.DataFrame({
        'datetime': dts,
        'open':     bars['open'].astype(float),
        'high':     bars['high'].astype(float),
        'low':      bars['low'].astype(float),
        'close':    bars['close'].astype(float),
        'volume':   bars['volume'].astype(int),
        'turnover': turnover,
    })


def _load_npy_cache(
    code: str,
    frequency: str,
    start: str,
    end: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> Optional[pd.DataFrame]:
    safe = code.replace('.', '_')
    p = cache_dir / f"{safe}_{frequency}.npy"
    if not p.exists():
        return None
    try:
        bars = np.load(str(p), allow_pickle=False)
        if bars.dtype != _BAR_DTYPE or len(bars) == 0:
            return None
    except Exception:
        return None

    df = decode_bar_array(bars)
    t_start = pd.Timestamp(start)
    t_end   = pd.Timestamp(end) + pd.Timedelta(days=1)
    df = df[(df['datetime'] >= t_start) & (df['datetime'] < t_end)]
    return df if not df.empty else None


def fetch_klines(
    code: str,
    frequency: str,
    start: str,
    end: str,
    host: str = '127.0.0.1',
    port: int = 11112,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    cached = _load_npy_cache(code, frequency, start, end, cache_dir)
    if cached is not None:
        return cached

    try:
        from futu import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK
    except ImportError:
        return pd.DataFrame()

    ktype_map = {'15m': KLType.K_15M, '1h': KLType.K_60M, '1d': KLType.K_DAY}
    ktype = ktype_map.get(frequency, KLType.K_DAY)

    try:
        ctx = OpenQuoteContext(host=host, port=port)
        all_dfs: List[pd.DataFrame] = []
        page_key = None
        fields = [KL_FIELD.DATE_TIME, KL_FIELD.OPEN, KL_FIELD.HIGH,
                  KL_FIELD.LOW, KL_FIELD.CLOSE, KL_FIELD.TRADE_VOL]
        try:
            fields.append(KL_FIELD.TRADE_VAL)
        except AttributeError:
            pass
        while True:
            common = dict(code=code, ktype=ktype, autype=AuType.QFQ,
                          fields=fields, max_count=3000)
            if page_key is None:
                ret, df, page_key = ctx.request_history_kline(start=start, end=end, **common)
            else:
                ret, df, page_key = ctx.request_history_kline(page_req_key=page_key, **common)
            if ret != RET_OK or df is None or df.empty:
                break
            all_dfs.append(df)
            if page_key is None:
                break
        ctx.close()
        if not all_dfs:
            return pd.DataFrame()
        raw = pd.concat(all_dfs, ignore_index=True)
        raw['datetime'] = pd.to_datetime(raw['time_key'])
        for col in ('turnover', 'trade_val'):
            if col in raw.columns:
                raw['turnover'] = raw[col]
                break
        else:
            raw['turnover'] = 0.0
        return raw[['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']].copy()
    except Exception as e:
        print(f"[警告] 拉取 {code} {frequency} K线失败: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
#  时间转换
# ════════════════════════════════════════════════════════════════════════════

def _to_chart_time(dt: pd.Timestamp, freq: str):
    if freq == '1d':
        return dt.strftime('%Y-%m-%d')
    return int(dt.replace(tzinfo=_HK_TZ).timestamp())


def _snap_to_freq(dt: pd.Timestamp, freq: str) -> pd.Timestamp:
    if freq == '1h':
        return dt.replace(minute=0, second=0, microsecond=0)
    if freq == '15m':
        return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
    return dt


# ════════════════════════════════════════════════════════════════════════════
#  HTML 模板
# ════════════════════════════════════════════════════════════════════════════

_HTML = """\
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>交易回测可视化</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #1e1e2e; color: #cdd6f4;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      height: 100vh; display: flex; flex-direction: column; overflow: hidden;
    }
    /* ── 顶栏 ── */
    #topbar {
      background: #181825; border-bottom: 1px solid #45475a;
      height: 48px; display: flex; align-items: center;
      padding: 0 20px; flex-shrink: 0;
    }
    #topbar span { font-size: 15px; font-weight: 600; }
    /* ── 主体 ── */
    #body { display: flex; flex: 1; overflow: hidden; }
    /* ── 侧边栏 ── */
    #sidebar {
      width: 200px; min-width: 200px; background: #1e1e2e;
      border-right: 1px solid #45475a; overflow-y: auto; padding-top: 12px;
    }
    .sec-label {
      color: #6c7086; font-size: 10px; letter-spacing: 1.2px;
      padding: 0 20px 8px; text-transform: uppercase;
    }
    .stock-item {
      padding: 10px 20px; cursor: pointer;
      border-left: 3px solid transparent; user-select: none; line-height: 1.5;
    }
    .stock-item:hover { background: #313244; }
    .stock-item.active {
      background: #313244; border-left-color: #89b4fa;
      color: #89b4fa; font-weight: 600;
    }
    .stock-item .code { font-size: 13px; }
    .stock-item .name { color: #a6adc8; font-size: 12px; }
    .stock-item .cnt  { color: #6c7086; font-size: 11px; }
    .stock-item.active .name { color: #89b4facc; }
    .stock-item.active .cnt  { color: #89b4fa99; }
    /* ── 右侧面板 ── */
    #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
    #toolbar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 20px; height: 48px; flex-shrink: 0;
      border-bottom: 1px solid #45475a;
    }
    #stock-title { font-size: 14px; font-weight: 600; color: #cdd6f4; }
    .pbtn {
      background: none; border: 1px solid #45475a; color: #6c7086;
      padding: 4px 14px; border-radius: 4px; cursor: pointer;
      font-size: 12px; margin-left: 8px; transition: all .15s;
    }
    .pbtn:hover  { color: #cdd6f4; border-color: #cdd6f4; }
    .pbtn.active { background: #313244; color: #89b4fa; border-color: #89b4fa; }
    /* ── K线区域 ── */
    #panels { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    #chart-wrap { flex: 1; position: relative; overflow: hidden; min-height: 0; }
    #chart { position: absolute; inset: 0; }
    /* ── OHLCV 悬浮框 ── */
    #ohlcv {
      position: absolute; top: 8px; left: 12px; z-index: 5;
      display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap;
      font-size: 12px; color: #6c7086; pointer-events: none;
      background: #181825cc; padding: 3px 10px; border-radius: 4px;
    }
    #ohlcv b    { margin-left: 3px; font-weight: 600; color: #cdd6f4; }
    #ohlcv .up  { color: #a6e3a1; }
    #ohlcv .dn  { color: #f38ba8; }
    #ov-time    { color: #a6adc8; margin-right: 4px; }
    /* ── 加载遮罩 ── */
    #loading {
      display: none; position: absolute; inset: 0;
      background: #1e1e2ecc; align-items: center; justify-content: center;
      font-size: 14px; color: #6c7086; z-index: 10;
    }
    #loading.show { display: flex; }
    /* ── 净值面板 ── */
    #pf-panel {
      height: 220px; flex-shrink: 0;
      border-top: 1px solid #45475a;
      display: flex; flex-direction: column;
    }
    #pf-toolbar {
      display: flex; align-items: center; gap: 10px;
      padding: 0 12px; height: 38px; flex-shrink: 0;
      border-bottom: 1px solid #31324488;
      font-size: 12px; color: #6c7086;
    }
    #pf-toolbar .pf-label { font-weight: 600; color: #a6adc8; white-space: nowrap; }
    #pf-legend { flex: 1; display: flex; gap: 14px; }
    .pf-leg-item { display: flex; align-items: center; gap: 4px; }
    .pf-leg-dot  { width: 8px; height: 2px; border-radius: 1px; }
    #bm-input {
      background: #313244; border: 1px solid #45475a; color: #cdd6f4;
      padding: 2px 8px; border-radius: 4px; font-size: 11px; width: 120px;
      outline: none;
    }
    #bm-input::placeholder { color: #6c7086; }
    #bm-input:focus { border-color: #89b4fa; }
    #bm-btn {
      background: none; border: 1px solid #45475a; color: #6c7086;
      padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;
    }
    #bm-btn:hover { color: #cdd6f4; border-color: #cdd6f4; }
    #bm-clear {
      display: none; background: none; border: none;
      color: #f38ba8; cursor: pointer; font-size: 11px; padding: 0 4px;
    }
    #bm-clear:hover { color: #ff6b6b; }
    #pf-wrap { flex: 1; position: relative; overflow: hidden; }
    #pf-chart { position: absolute; inset: 0; }
  </style>
</head>
<body>
  <div id="topbar"><span>交易回测可视化</span></div>
  <div id="body">
    <div id="sidebar">
      <div class="sec-label">交易标的</div>
      <div id="stock-list"></div>
    </div>
    <div id="main">
      <div id="toolbar">
        <span id="stock-title"></span>
        <div>
          <button class="pbtn active" data-freq="15m">15 分钟</button>
          <button class="pbtn"        data-freq="1h">1 小时</button>
          <button class="pbtn"        data-freq="1d">日线</button>
        </div>
      </div>
      <div id="panels">
        <!-- K线图 -->
        <div id="chart-wrap">
          <div id="chart"></div>
          <div id="ohlcv">
            <span id="ov-time"></span>
            <span>开<b id="ov-o"></b></span>
            <span>高<b id="ov-h" class="up"></b></span>
            <span>低<b id="ov-l" class="dn"></b></span>
            <span>收<b id="ov-c"></b></span>
            <span>涨跌<b id="ov-chg"></b></span>
            <span>额<b id="ov-amt"></b></span>
            <span>量<b id="ov-v"></b></span>
          </div>
          <div id="loading">加载中…</div>
        </div>
        <!-- 净值曲线 -->
        <div id="pf-panel">
          <div id="pf-toolbar">
            <span class="pf-label">净值曲线</span>
            <div id="pf-legend"></div>
            <input  id="bm-input"  placeholder="基准代码，如 HK.800000" />
            <button id="bm-btn">添加基准</button>
            <button id="bm-clear">✕ 移除基准</button>
          </div>
          <div id="pf-wrap"><div id="pf-chart"></div></div>
        </div>
      </div>
    </div>
  </div>

<script>
const STOCKS = __STOCKS__;
let chart, candleSeries, volSeries;
let pfChart, pfPortSeries, pfBmSeries;
let curCode = null, curFreq = '15m';
let curKlines = [];   // 当前 K 线数组，供涨跌幅计算

// ════════════════════════════════════════════════════════════════════════════
//  工具函数
// ════════════════════════════════════════════════════════════════════════════
const fmt2  = v => v.toFixed(2);
const fmtPct = v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
const fmtAmt = v => {
  if (v >= 1e8)  return (v/1e8).toFixed(2) + '亿';
  if (v >= 1e4)  return (v/1e4).toFixed(0) + '万';
  return String(Math.round(v));
};

// ════════════════════════════════════════════════════════════════════════════
//  K 线图初始化
// ════════════════════════════════════════════════════════════════════════════
function initChart() {
  const el = document.getElementById('chart');
  chart = LightweightCharts.createChart(el, {
    layout: { background: { color: '#181825' }, textColor: '#cdd6f4' },
    grid:   { vertLines: { color: '#31324455' }, horzLines: { color: '#31324455' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#45475a' },
    timeScale: {
      borderColor: '#45475a', timeVisible: true, secondsVisible: false,
      timezone: 'Asia/Hong_Kong',
    },
  });
  new ResizeObserver(() =>
    chart.applyOptions({ width: el.clientWidth, height: el.clientHeight })
  ).observe(el);

  // OHLCV 悬浮信息
  const ovTime = document.getElementById('ov-time');
  const ovO    = document.getElementById('ov-o');
  const ovH    = document.getElementById('ov-h');
  const ovL    = document.getElementById('ov-l');
  const ovC    = document.getElementById('ov-c');
  const ovChg  = document.getElementById('ov-chg');
  const ovAmt  = document.getElementById('ov-amt');
  const ovV    = document.getElementById('ov-v');

  chart.subscribeCrosshairMove(param => {
    if (!candleSeries || !param.time || !param.seriesData) return;
    const bar = param.seriesData.get(candleSeries);
    if (!bar) return;

    // 时间显示
    let timeStr = '';
    if (typeof param.time === 'string') {
      timeStr = param.time;
    } else {
      timeStr = new Date(param.time * 1000).toLocaleString('zh-CN', {
        timeZone: 'Asia/Hong_Kong',
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', hour12: false,
      });
    }

    // 涨跌幅：用上一根 K 线收盘价
    const idx = curKlines.findIndex(b => b.time === param.time);
    const prevClose = idx > 0 ? curKlines[idx - 1].close : bar.open;
    const chgPct = prevClose ? (bar.close - prevClose) / prevClose * 100 : 0;
    const isUp = bar.close >= prevClose;

    // 成交额
    const kBar = idx >= 0 ? curKlines[idx] : null;
    const amt  = kBar ? (kBar.turnover ?? 0) : 0;

    // 成交量
    const volBar = volSeries ? param.seriesData.get(volSeries) : null;
    const vol    = volBar ? (volBar.value ?? 0) : 0;

    ovTime.textContent = timeStr;
    ovO.textContent    = fmt2(bar.open);
    ovH.textContent    = fmt2(bar.high);
    ovL.textContent    = fmt2(bar.low);
    ovC.textContent    = fmt2(bar.close);
    ovC.className      = isUp ? 'up' : 'dn';
    ovChg.textContent  = fmtPct(chgPct);
    ovChg.className    = isUp ? 'up' : 'dn';
    ovAmt.textContent  = amt > 0 ? fmtAmt(amt) : '—';
    ovV.textContent    = fmtAmt(vol);
  });
}

// ════════════════════════════════════════════════════════════════════════════
//  净值图初始化
// ════════════════════════════════════════════════════════════════════════════
function initPfChart() {
  const el = document.getElementById('pf-chart');
  pfChart = LightweightCharts.createChart(el, {
    layout: { background: { color: '#181825' }, textColor: '#6c7086' },
    grid:   { vertLines: { color: '#31324433' }, horzLines: { color: '#31324433' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#45475a', scaleMargins: { top: 0.08, bottom: 0.08 } },
    timeScale: { borderColor: '#45475a', timeVisible: false },
    handleScroll: true,
    handleScale: true,
  });
  new ResizeObserver(() =>
    pfChart.applyOptions({ width: el.clientWidth, height: el.clientHeight })
  ).observe(el);
}

async function loadPfChart() {
  const resp = await fetch('/api/portfolio');
  const data = await resp.json();
  if (!data.length) return;

  if (pfPortSeries) { pfChart.removeSeries(pfPortSeries); pfPortSeries = null; }

  pfPortSeries = pfChart.addLineSeries({
    color: '#89b4fa', lineWidth: 2, priceLineVisible: false,
    lastValueVisible: true,
  });
  pfPortSeries.setData(data);
  pfChart.timeScale().fitContent();
  updatePfLegend();
}

// ════════════════════════════════════════════════════════════════════════════
//  基准
// ════════════════════════════════════════════════════════════════════════════
async function addBenchmark(code) {
  // 取净值曲线日期范围
  const pfResp = await fetch('/api/portfolio');
  const pfData = await pfResp.json();
  if (!pfData.length) return;
  const start = pfData[0].time;
  const end   = pfData[pfData.length - 1].time;

  const resp = await fetch(`/api/klines?code=${encodeURIComponent(code)}&freq=1d&start=${start}&end=${end}&ignore_trades=1`);
  const klines = await resp.json();
  if (!klines.length) { alert(`未能获取 ${code} 数据`); return; }

  // 归一化：以第一根有效日期为基准
  const base = klines[0].close;
  if (!base) return;
  const normalized = klines.map(b => ({ time: b.time, value: b.close / base }));

  if (pfBmSeries) { pfChart.removeSeries(pfBmSeries); pfBmSeries = null; }
  pfBmSeries = pfChart.addLineSeries({
    color: '#f9e2af', lineWidth: 1.5, lineStyle: 1 /* dashed */,
    priceLineVisible: false, lastValueVisible: true,
  });
  pfBmSeries.setData(normalized);

  document.getElementById('bm-clear').style.display = 'inline';
  updatePfLegend(code);
}

function removeBenchmark() {
  if (pfBmSeries) { pfChart.removeSeries(pfBmSeries); pfBmSeries = null; }
  document.getElementById('bm-clear').style.display = 'none';
  updatePfLegend();
}

function updatePfLegend(bmCode) {
  const leg = document.getElementById('pf-legend');
  leg.innerHTML = `
    <span class="pf-leg-item">
      <span class="pf-leg-dot" style="background:#89b4fa"></span>
      <span style="color:#89b4fa">策略净值</span>
    </span>
    ${bmCode ? `<span class="pf-leg-item">
      <span class="pf-leg-dot" style="background:#f9e2af"></span>
      <span style="color:#f9e2af">${bmCode}</span>
    </span>` : ''}
  `;
}

// ════════════════════════════════════════════════════════════════════════════
//  K 线加载
// ════════════════════════════════════════════════════════════════════════════
async function loadChart(code, freq) {
  document.getElementById('loading').classList.add('show');
  if (candleSeries) { chart.removeSeries(candleSeries); candleSeries = null; }
  if (volSeries)    { chart.removeSeries(volSeries);    volSeries    = null; }

  try {
    const [kr, tr] = await Promise.all([
      fetch(`/api/klines?code=${encodeURIComponent(code)}&freq=${freq}`),
      fetch(`/api/trades?code=${encodeURIComponent(code)}&freq=${freq}`),
    ]);
    const klines = await kr.json();
    const trades = await tr.json();

    if (!klines.length) {
      document.getElementById('stock-title').textContent = `${code}  ·  无 K 线数据`;
      return;
    }
    curKlines = klines;   // 保存供涨跌幅计算

    // 成交量
    volSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol', color: '#a6e3a140',
    });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volSeries.setData(klines.map(b => ({
      time: b.time, value: b.volume,
      color: b.close >= b.open ? '#a6e3a155' : '#f38ba855',
    })));

    // 蜡烛图
    candleSeries = chart.addCandlestickSeries({
      upColor: '#a6e3a1', downColor: '#f38ba8',
      borderUpColor: '#a6e3a1', borderDownColor: '#f38ba8',
      wickUpColor: '#a6e3a1', wickDownColor: '#f38ba8',
    });
    candleSeries.setData(klines);

    // 买卖标记
    const markers = trades
      .map(t => ({
        time:     t.time,
        position: t.side === 'BUY' ? 'belowBar' : 'aboveBar',
        color:    t.side === 'BUY' ? '#a6e3a1'  : '#f38ba8',
        shape:    t.side === 'BUY' ? 'arrowUp'  : 'arrowDown',
        text:     `${t.side === 'BUY' ? '买' : '卖'} ${t.qty}股 @${t.price}`,
        size: 1.5,
      }))
      .sort((a, b) => (a.time > b.time ? 1 : a.time < b.time ? -1 : 0));
    candleSeries.setMarkers(markers);

    chart.timeScale().fitContent();

    const info = STOCKS.find(s => s.code === code) || {};
    const nameStr = info.name ? `${info.name}  ` : '';
    document.getElementById('stock-title').textContent =
      `${nameStr}${code}  ·  买入 ${info.buys ?? 0} 笔  卖出 ${info.sells ?? 0} 笔`;
  } finally {
    document.getElementById('loading').classList.remove('show');
  }
}

// ════════════════════════════════════════════════════════════════════════════
//  侧边栏 & 控件
// ════════════════════════════════════════════════════════════════════════════
function buildSidebar() {
  const list = document.getElementById('stock-list');
  STOCKS.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'stock-item' + (i === 0 ? ' active' : '');
    div.innerHTML = `<div class="code">${s.code}</div>`
      + (s.name ? `<div class="name">${s.name}</div>` : '')
      + `<div class="cnt">${s.total} 笔</div>`;
    div.onclick = () => {
      document.querySelectorAll('.stock-item').forEach(el => el.classList.remove('active'));
      div.classList.add('active');
      curCode = s.code;
      loadChart(curCode, curFreq);
    };
    list.appendChild(div);
  });
}

document.querySelectorAll('.pbtn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.pbtn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    curFreq = btn.dataset.freq;
    if (curCode) loadChart(curCode, curFreq);
  };
});

document.getElementById('bm-btn').onclick = () => {
  const code = document.getElementById('bm-input').value.trim();
  if (code) addBenchmark(code);
};
document.getElementById('bm-input').onkeydown = e => {
  if (e.key === 'Enter') document.getElementById('bm-btn').click();
};
document.getElementById('bm-clear').onclick = removeBenchmark;

// ════════════════════════════════════════════════════════════════════════════
//  启动
// ════════════════════════════════════════════════════════════════════════════
initChart();
initPfChart();
buildSidebar();
loadPfChart();
updatePfLegend();
if (STOCKS.length) {
  curCode = STOCKS[0].code;
  loadChart(curCode, curFreq);
}
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════════════
#  Flask 应用
# ════════════════════════════════════════════════════════════════════════════

def build_app(
    trades: pd.DataFrame,
    portfolio: Optional[pd.DataFrame],
    klines_cache: Optional[dict] = None,
    futu_host: str = '127.0.0.1',
    futu_port: int = 11112,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> Flask:
    """构建 Flask 应用。

    Parameters
    ----------
    trades:       交易记录 DataFrame
    portfolio:    净值 DataFrame（index 为日期，含 unit_net_value 列）
    klines_cache: 预先拉取的 K 线数据，格式 {code: {freq: df}}
                  df 含 datetime/open/high/low/close/volume/turnover 列
                  优先于 Futu API 和磁盘缓存
    futu_host:    Futu OpenD 地址（klines_cache 缺失时兜底）
    futu_port:    Futu OpenD 端口
    cache_dir:    磁盘缓存目录
    """
    app = Flask(__name__)

    stocks_info: list = []
    if not trades.empty:
        for code, grp in trades.groupby('order_book_id'):
            buys  = int((grp['side'] == 'BUY').sum())
            sells = int((grp['side'] == 'SELL').sum())
            name  = grp['symbol'].iloc[0] if 'symbol' in grp.columns else ''
            stocks_info.append({'code': code, 'name': name, 'buys': buys, 'sells': sells, 'total': buys + sells})
        stocks_info.sort(key=lambda x: -x['total'])

    _page = _HTML.replace('__STOCKS__', json.dumps(stocks_info, ensure_ascii=False))

    @app.route('/')
    def index():
        return _page

    @app.route('/api/klines')
    def api_klines():
        code           = request.args.get('code', '')
        freq           = request.args.get('freq', '15m')
        ignore_tr      = request.args.get('ignore_trades', '0') == '1'
        start_override = request.args.get('start')
        end_override   = request.args.get('end')

        if ignore_tr or trades.empty:
            if not start_override or not end_override:
                return jsonify([])
            start, end = start_override, end_override
        else:
            stock_trades = trades[trades['order_book_id'] == code]
            if stock_trades.empty:
                return jsonify([])
            t_min = stock_trades['datetime'].min()
            t_max = stock_trades['datetime'].max()
            pad   = pd.Timedelta(days=10 if freq == '1d' else 3)
            start = start_override or (t_min - pad).strftime('%Y-%m-%d')
            end   = end_override   or (t_max + pad).strftime('%Y-%m-%d')

        # 优先使用预缓存数据
        df = None
        if klines_cache and code in klines_cache and freq in klines_cache[code]:
            cached_df = klines_cache[code][freq]
            if not cached_df.empty and 'datetime' in cached_df.columns:
                t_start = pd.Timestamp(start)
                t_end   = pd.Timestamp(end) + pd.Timedelta(days=1)
                df = cached_df[(cached_df['datetime'] >= t_start) & (cached_df['datetime'] < t_end)].copy()
                if df.empty:
                    df = None

        if df is None:
            df = fetch_klines(code, freq, start, end, futu_host, futu_port, cache_dir)

        if df is None or df.empty:
            return jsonify([])

        result = [
            {
                'time':     _to_chart_time(row['datetime'], freq),
                'open':     float(row['open']),
                'high':     float(row['high']),
                'low':      float(row['low']),
                'close':    float(row['close']),
                'volume':   int(row['volume']),
                'turnover': float(row.get('turnover', 0)),
            }
            for _, row in df.iterrows()
        ]
        return jsonify(result)

    @app.route('/api/trades')
    def api_trades():
        code = request.args.get('code', '')
        freq = request.args.get('freq', '15m')

        stock_trades = trades[trades['order_book_id'] == code] if not trades.empty else pd.DataFrame()
        if stock_trades.empty:
            return jsonify([])

        result = [
            {
                'time':  _to_chart_time(_snap_to_freq(row['datetime'], freq), freq),
                'side':  row['side'],
                'price': float(row['last_price']),
                'qty':   int(row['last_quantity']),
            }
            for _, row in stock_trades.iterrows()
        ]
        return jsonify(result)

    @app.route('/api/portfolio')
    def api_portfolio():
        if portfolio is None or portfolio.empty:
            return jsonify([])
        result = []
        for dt, row in portfolio.iterrows():
            result.append({
                'time':  dt.strftime('%Y-%m-%d'),
                'value': round(float(row['unit_net_value']), 6),
            })
        return jsonify(result)

    return app


# ════════════════════════════════════════════════════════════════════════════
#  公开入口
# ════════════════════════════════════════════════════════════════════════════

def _free_port(port: int) -> None:
    """Kill any process currently listening on port so the new server can bind."""
    import os
    import signal
    import subprocess
    import time
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True, text=True,
        )
        pids = [p for p in result.stdout.strip().split('\n') if p.strip()]
        if pids:
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(0.4)  # brief pause for OS to release the port
    except Exception:
        pass


def launch_viewer(
    trades_source,
    klines_cache: Optional[dict] = None,
    futu_host: str = '127.0.0.1',
    futu_port: int = 11112,
    server_port: int = 8050,
    cache_dir: str = str(_DEFAULT_CACHE_DIR),
    debug: bool = False,
):
    """启动可视化界面。

    Parameters
    ----------
    trades_source: result dict、DataFrame、.pkl 路径或 .csv 路径
    klines_cache:  预先拉取的 K 线数据，格式 {code: {freq: df}}
    futu_host:     Futu OpenD 地址（klines_cache 缺失时兜底）
    futu_port:     Futu OpenD 端口
    server_port:   本地 HTTP 端口
    cache_dir:     磁盘缓存目录
    debug:         Flask debug 模式
    """
    trades    = load_trades(trades_source)
    portfolio = load_portfolio(trades_source)

    if trades.empty:
        print("[警告] 未找到交易记录，将显示空界面。")
    else:
        print(f"[info] 共 {trades['order_book_id'].nunique()} 只标的，{len(trades)} 笔交易")

    if portfolio is not None:
        print(f"[info] 净值曲线 {len(portfolio)} 个交易日，"
              f"最终净值 {portfolio['unit_net_value'].iloc[-1]:.4f}")

    _free_port(server_port)
    app = build_app(trades, portfolio, klines_cache, futu_host, futu_port, Path(cache_dir))

    url = f"http://127.0.0.1:{server_port}"
    print(f"\n可视化界面已启动 → {url}\n")

    import threading
    import webbrowser
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host='127.0.0.1', port=server_port, debug=debug, use_reloader=False)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RQAlpha 交易回测可视化')
    parser.add_argument('--trades',      required=True,           help='result.pkl 或 trades.csv 路径')
    parser.add_argument('--host',        default='127.0.0.1',     help='Futu OpenD 地址')
    parser.add_argument('--port',        type=int, default=11112, help='Futu OpenD 端口')
    parser.add_argument('--server-port', type=int, default=8050,  help='本地 HTTP 端口')
    parser.add_argument('--debug',       action='store_true')
    args = parser.parse_args()

    launch_viewer(args.trades, futu_host=args.host, futu_port=args.port,
                  server_port=args.server_port, debug=args.debug)
