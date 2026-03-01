# -*- coding: utf-8 -*-
"""
富途数据源 (FutuDataSource)

通过 OpenD 进程的 Futu API 提供港股历史 K 线数据，实现 AbstractDataSource 接口，
支持 '1d'（日线）和 '15m'（15 分钟线）两种频率。

数据通过内存缓存避免重复拉取；首次访问时按 (stock_code, ktype, date_range) 拉取
并以结构化 numpy 数组形式存储。
"""

import time
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from rqalpha.const import INSTRUMENT_TYPE, MARKET, TRADING_CALENDAR_TYPE
from rqalpha.interface import AbstractDataSource, ExchangeRate
from rqalpha.model.instrument import Instrument
from rqalpha.utils.datetime_func import convert_dt_to_int, convert_date_to_int
from rqalpha.utils.logger import system_log

try:
    from futu import (
        OpenQuoteContext, KLType, AuType, Market, SecurityType, RET_OK,
        KL_FIELD,
    )
    _FUTU_AVAILABLE = True
except ImportError:
    _FUTU_AVAILABLE = False

# ── numpy 结构化数组的 dtype，与 RQAlpha 内置 BaseDataSource 保持兼容 ───────────
BAR_DTYPE = np.dtype([
    ('datetime', '<u8'),        # YYYYMMDDHHMMSS 编码的 uint64
    ('open',     '<f4'),
    ('high',     '<f4'),
    ('low',      '<f4'),
    ('close',    '<f4'),
    ('volume',   '<i8'),
    ('total_turnover', '<f8'),
])

# 港股 HK_STOCK 交易日历缓存（DatetimeIndex）
_HK_CALENDAR_CACHE: Optional[pd.DatetimeIndex] = None
_HK_CALENDAR_LOCK = threading.Lock()


def _futu_ktype(frequency: str) -> "KLType":
    """将 RQAlpha frequency 字符串映射到 Futu KLType。"""
    if not _FUTU_AVAILABLE:
        raise RuntimeError("futu-api 未安装，请执行 pip install futu-api")
    mapping = {
        '1d':  KLType.K_DAY,
        '15m': KLType.K_15M,
        '1m':  KLType.K_1M,
    }
    if frequency not in mapping:
        raise NotImplementedError(f"FutuDataSource 不支持 frequency='{frequency}'")
    return mapping[frequency]


def _parse_time_key(time_key: str) -> int:
    """把富途 time_key ('2024-01-02 09:45:00') 解析为 RQAlpha uint64 编码。"""
    dt = datetime.strptime(time_key, '%Y-%m-%d %H:%M:%S')
    return int(convert_dt_to_int(dt))


def _df_to_bars(df: pd.DataFrame, frequency: str) -> np.ndarray:
    """把富途返回的 DataFrame 转换为 numpy 结构化数组。"""
    n = len(df)
    bars = np.empty(n, dtype=BAR_DTYPE)
    if frequency == '1d':
        # 日线的 time_key 格式为 '2024-01-02'
        for i, tk in enumerate(df['time_key']):
            dt = datetime.strptime(tk[:10], '%Y-%m-%d')
            bars['datetime'][i] = np.uint64(convert_date_to_int(dt))
    else:
        for i, tk in enumerate(df['time_key']):
            bars['datetime'][i] = np.uint64(_parse_time_key(tk))

    bars['open']  = df['open'].values.astype(np.float32)
    bars['high']  = df['high'].values.astype(np.float32)
    bars['low']   = df['low'].values.astype(np.float32)
    bars['close'] = df['close'].values.astype(np.float32)
    bars['volume'] = df['volume'].values.astype(np.int64)
    bars['total_turnover'] = df['turnover'].values.astype(np.float64) if 'turnover' in df.columns \
        else np.zeros(n, dtype=np.float64)
    return bars


def _bars_last_date(bars: np.ndarray) -> date:
    """从 bars 数组的最后一根 K 线提取日期。
    datetime 编码为 YYYYMMDDHHMMSS（14 位整数）。
    """
    dt_int = int(bars['datetime'][-1])
    d = dt_int // 1000000          # 去掉 HHMMSS → YYYYMMDD
    return date(d // 10000, (d % 10000) // 100, d % 100)


class FutuDataSource(AbstractDataSource):
    """
    使用富途 API 提供港股历史数据。

    缓存策略（三级）：
    1. 运行期内存缓存（per-process，最快）
    2. 磁盘持久化缓存（~/.rqalpha/futu_cache/<code>_<freq>.npy）
    3. Futu OpenD API（仅拉取磁盘缓存缺失或过期的增量部分）

    好处：
    - 重复回测无需重新拉取历史数据
    - 可视化工具在 OpenD 未运行时也能读取已缓存的 K 线
    """

    def __init__(self, host: str, port: int, api_delay: float,
                 cache_dir: str = '~/.rqalpha/futu_cache'):
        if not _FUTU_AVAILABLE:
            raise RuntimeError(
                "请先安装富途 API：pip install futu-api\n"
                "并确保 OpenD 已在本机运行"
            )
        self._host = host
        self._port = port
        self._api_delay = api_delay
        self._quote_ctx: Optional[OpenQuoteContext] = None
        self._lock = threading.Lock()

        # {(order_book_id, frequency): np.ndarray}
        self._bar_cache: Dict = {}
        # {order_book_id: Instrument}
        self._instruments: Dict[str, Instrument] = {}
        # HK 交易日历
        self._hk_calendar: Optional[pd.DatetimeIndex] = None

        # 磁盘缓存目录（空字符串 = 不使用磁盘缓存）
        self._cache_dir: Optional[Path] = None
        if cache_dir:
            self._cache_dir = Path(cache_dir).expanduser()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            system_log.info(f"[Futu] K线磁盘缓存目录: {self._cache_dir}")

    # ── 连接管理 ──────────────────────────────────────────────────────────────

    def _get_ctx(self) -> "OpenQuoteContext":
        if self._quote_ctx is None:
            self._quote_ctx = OpenQuoteContext(host=self._host, port=self._port)
        return self._quote_ctx

    def close(self):
        if self._quote_ctx is not None:
            self._quote_ctx.close()
            self._quote_ctx = None

    # ── 磁盘缓存 ─────────────────────────────────────────────────────────────

    def _cache_path(self, order_book_id: str, frequency: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        safe = order_book_id.replace('.', '_')
        return self._cache_dir / f"{safe}_{frequency}.npy"

    def _load_disk_cache(self, order_book_id: str, frequency: str) -> Optional[np.ndarray]:
        p = self._cache_path(order_book_id, frequency)
        if p is None or not p.exists():
            return None
        try:
            bars = np.load(str(p), allow_pickle=False)
            if bars.dtype == BAR_DTYPE and len(bars) > 0:
                return bars
        except Exception as e:
            system_log.warning(f"[Futu Cache] 读取失败 {order_book_id}/{frequency}: {e}")
        return None

    def _save_disk_cache(self, order_book_id: str, frequency: str, bars: np.ndarray):
        p = self._cache_path(order_book_id, frequency)
        if p is None or len(bars) == 0:
            return
        try:
            np.save(str(p), bars)
        except Exception as e:
            system_log.warning(f"[Futu Cache] 写入失败 {order_book_id}/{frequency}: {e}")

    # ── 内部拉取工具 ──────────────────────────────────────────────────────────

    def _fetch_kline(
        self,
        stock_code: str,
        start: str,
        end: str,
        ktype: "KLType",
        max_count: int = 3000,
    ) -> Optional[pd.DataFrame]:
        """调用富途 request_history_kline，返回 DataFrame 或 None。"""
        ctx = self._get_ctx()
        all_dfs = []
        page_req_key = None
        while True:
            if page_req_key is None:
                ret, df, page_req_key = ctx.request_history_kline(
                    stock_code,
                    start=start,
                    end=end,
                    ktype=ktype,
                    autype=AuType.QFQ,
                    fields=[
                        KL_FIELD.DATE_TIME,
                        KL_FIELD.OPEN,
                        KL_FIELD.HIGH,
                        KL_FIELD.LOW,
                        KL_FIELD.CLOSE,
                        KL_FIELD.TRADE_VOL,
                        KL_FIELD.TRADE_VAL,
                    ],
                    max_count=max_count,
                )
            else:
                ret, df, page_req_key = ctx.request_history_kline(
                    stock_code,
                    ktype=ktype,
                    autype=AuType.QFQ,
                    fields=[
                        KL_FIELD.DATE_TIME,
                        KL_FIELD.OPEN,
                        KL_FIELD.HIGH,
                        KL_FIELD.LOW,
                        KL_FIELD.CLOSE,
                        KL_FIELD.TRADE_VOL,
                        KL_FIELD.TRADE_VAL,
                    ],
                    max_count=max_count,
                    page_req_key=page_req_key,
                )
            time.sleep(self._api_delay)
            if ret != RET_OK:
                system_log.warning(f"[Futu] 拉取 {stock_code} K 线失败: {df}")
                return None
            if df is not None and not df.empty:
                all_dfs.append(df)
            if page_req_key is None:
                break
        if not all_dfs:
            return None
        return pd.concat(all_dfs, ignore_index=True)

    def _ensure_bars(self, order_book_id: str, frequency: str) -> Optional[np.ndarray]:
        """
        三级缓存策略：
        1. 命中内存缓存 → 直接返回
        2. 读磁盘缓存 → 仅增量拉取 (last_date+1 → today) → 合并 → 回写磁盘
        3. 磁盘无缓存 → 全量拉取 (2015-01-01 → today) → 写磁盘
        """
        key = (order_book_id, frequency)
        if key in self._bar_cache:
            return self._bar_cache[key]

        with self._lock:
            if key in self._bar_cache:
                return self._bar_cache[key]

            today = datetime.now().strftime('%Y-%m-%d')
            ktype = _futu_ktype(frequency)

            # ── 1. 读磁盘缓存 ──────────────────────────────────────────────
            cached = self._load_disk_cache(order_book_id, frequency)

            if cached is not None and len(cached) > 0:
                # 若缓存文件今天已经写过，跳过拉取（避免周末/节假日无数据时反复请求）
                p = self._cache_path(order_book_id, frequency)
                if p and p.exists():
                    mtime_date = date.fromtimestamp(p.stat().st_mtime)
                    if mtime_date >= date.today():
                        self._bar_cache[key] = cached
                        return cached

                last_date = _bars_last_date(cached)
                fetch_start_date = last_date + timedelta(days=1)
                fetch_start = fetch_start_date.strftime('%Y-%m-%d')

                if fetch_start > today:
                    # 缓存已是最新（bar 日期维度）
                    self._bar_cache[key] = cached
                    return cached

                # ── 2. 增量拉取 ────────────────────────────────────────────
                system_log.info(
                    f"[Futu Cache] {order_book_id}/{frequency} "
                    f"增量拉取 {fetch_start} → {today}"
                )
                df = self._fetch_kline(order_book_id, fetch_start, today, ktype)
                if df is not None and not df.empty:
                    new_bars = _df_to_bars(df, frequency)
                    merged = np.concatenate([cached, new_bars])
                    # 去重（以 datetime 为唯一键）
                    _, idx = np.unique(merged['datetime'], return_index=True)
                    bars = merged[idx]
                else:
                    bars = cached   # 无新数据，直接用缓存
            else:
                # ── 3. 全量拉取 ────────────────────────────────────────────
                system_log.info(
                    f"[Futu Cache] {order_book_id}/{frequency} 全量拉取 (首次)"
                )
                df = self._fetch_kline(order_book_id, '2015-01-01', today, ktype)
                if df is None or df.empty:
                    bars = np.empty(0, dtype=BAR_DTYPE)
                else:
                    bars = _df_to_bars(df, frequency)

            # 写回磁盘缓存
            self._save_disk_cache(order_book_id, frequency, bars)
            self._bar_cache[key] = bars
            return bars

    # ── 预加载 ────────────────────────────────────────────────────────────────

    def prefetch_stocks(self, stock_codes: List[str], frequency: str = '15m'):
        """在策略初始化阶段批量预拉取数据，减少 handle_bar 延迟。"""
        for code in stock_codes:
            self._ensure_bars(code, frequency)
            self._ensure_bars(code, '1d')

    # ── AbstractDataSource 接口实现 ───────────────────────────────────────────

    def available_data_range(self, frequency: str):
        return date(2015, 1, 1), date.today()

    def get_trading_calendars(self):
        """返回港股交易日历。从 Futu API 拉取后缓存。
        CN_STOCK key 也返回 HK 日历，确保 RQAlpha 内部的默认日历查询（_adjust_start_date 等）正常工作。
        """
        if self._hk_calendar is None:
            self._hk_calendar = self._fetch_hk_calendar()
        return {
            TRADING_CALENDAR_TYPE.HK_STOCK: self._hk_calendar,
            TRADING_CALENDAR_TYPE.CN_STOCK:  self._hk_calendar,  # RQAlpha 内部默认查询 CN_STOCK
        }

    def _fetch_hk_calendar(self) -> pd.DatetimeIndex:
        ctx = self._get_ctx()
        start = '2015-01-01'
        end = datetime.now().strftime('%Y-%m-%d')
        ret, data = ctx.request_trading_days(market=Market.HK, start=start, end=end)
        time.sleep(self._api_delay)
        if ret != RET_OK or not data:
            # 退化：生成连续工作日（粗略估计）
            system_log.warning("[Futu] 无法拉取港股交易日历，使用工作日近似")
            return pd.bdate_range(start, end)
        # data 为 list[dict]，每项含 'time' 键（'YYYY-MM-DD' 字符串）
        dates = pd.to_datetime([item['time'][:10] for item in data])
        return pd.DatetimeIndex(sorted(dates))

    def get_hk_trading_dates(self, start: date, end: date) -> pd.DatetimeIndex:
        """供 FutuEventSource 调用的交易日接口。"""
        cal = self.get_trading_calendars()[TRADING_CALENDAR_TYPE.HK_STOCK]
        mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
        return cal[mask]

    def get_instruments(
        self,
        id_or_syms: Optional[Iterable[str]] = None,
        types=None,
    ) -> Iterable[Instrument]:
        if not self._instruments:
            self._load_instruments()
        result = list(self._instruments.values())
        if id_or_syms is not None:
            codes = set(id_or_syms)
            result = [ins for ins in result if ins.order_book_id in codes or ins.symbol in codes]
        if types is not None:
            type_set = set(types)
            result = [ins for ins in result if ins.type in type_set]
        return result

    def _load_instruments(self):
        ctx = self._get_ctx()
        ret, df = ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK)
        time.sleep(self._api_delay)
        if ret != RET_OK or df is None or df.empty:
            system_log.warning("[Futu] 无法拉取港股基本信息")
            return
        for _, row in df.iterrows():
            code = row['code']
            lot_size = int(row.get('lot_size', 100)) if row.get('lot_size', 0) > 0 else 100
            listing_date = str(row.get('listing_date', '1990-01-01'))[:10]
            ins = Instrument(
                {
                    'order_book_id': code,
                    'symbol': row.get('name', code),
                    'abbrev_symbol': code,
                    'type': INSTRUMENT_TYPE.CS.value,
                    'listed_date': listing_date,
                    'de_listed_date': '2999-12-31',
                    'exchange': 'XHKG',
                    'board_lot': lot_size,
                    'round_lot': lot_size,
                    'board_type': '',   # 非 KSH，确保 round_lot 返回 board_lot 值
                    'market_tplus': 0,  # 港股当日可买卖（T+0 买卖，T+2 结算）
                },
                market=MARKET.HK,
            )
            self._instruments[code] = ins

    def get_new_stocks(self, days: int = 90, min_history_days: int = 5) -> List[str]:
        """返回最近 days 天内上市、且已有至少 min_history_days 个交易日历史的港股代码列表。"""
        if not self._instruments:
            self._load_instruments()
        today = date.today()
        cutoff_listed  = today - timedelta(days=days)          # 上市不早于此日期
        cutoff_history = today - timedelta(days=min_history_days)  # 已有足够历史
        result = []
        for ins in self._instruments.values():
            try:
                ld = date.fromisoformat(str(ins.listed_date)[:10])
            except Exception:
                continue
            if cutoff_listed <= ld <= cutoff_history:
                result.append(ins.order_book_id)
        result.sort()
        return result

    def get_new_stocks_for_date(self, target_date: date) -> List[str]:
        """返回指定日期上市的港股代码列表（供实盘 before_trading 调用）。"""
        if not self._instruments:
            self._load_instruments()
        result = []
        for ins in self._instruments.values():
            try:
                ld = date.fromisoformat(str(ins.listed_date)[:10])
            except Exception:
                continue
            if ld == target_date:
                result.append(ins.order_book_id)
        return sorted(result)

    def get_instrument(self, order_book_id: str) -> Optional[Instrument]:
        if not self._instruments:
            self._load_instruments()
        return self._instruments.get(order_book_id)

    def get_bar(self, instrument: Instrument, dt, frequency: str):
        """返回 dt 那根 K 线的单行数据（numpy void），供 BarObject 使用。"""
        bars = self._ensure_bars(instrument.order_book_id, frequency)
        if bars is None or len(bars) == 0:
            return None
        if frequency == '1d':
            dt_int = np.uint64(convert_date_to_int(dt))
        else:
            dt_int = np.uint64(convert_dt_to_int(dt))
        pos = bars['datetime'].searchsorted(dt_int)
        if pos >= len(bars) or bars['datetime'][pos] != dt_int:
            return None
        return bars[pos]

    def history_bars(
        self,
        instrument: Instrument,
        bar_count: Optional[int],
        frequency: str,
        fields,
        dt: datetime,
        skip_suspended: bool = True,
        include_now: bool = False,
        adjust_type: str = 'pre',
        adjust_orig=None,
    ) -> Optional[np.ndarray]:
        """返回截至 dt（不含）最近 bar_count 根 K 线。"""
        bars = self._ensure_bars(instrument.order_book_id, frequency)
        if bars is None or len(bars) == 0:
            return np.empty(0, dtype=BAR_DTYPE) if fields is None else np.array([])

        if frequency == '1d':
            dt_int = np.uint64(convert_date_to_int(dt))
        else:
            dt_int = np.uint64(convert_dt_to_int(dt))

        # 找到 dt 右侧边界（不含当前 bar，除非 include_now）
        side = 'right' if include_now else 'left'
        i = bars['datetime'].searchsorted(dt_int, side=side)

        if bar_count is None:
            left = 0
        else:
            left = max(0, i - bar_count)
        bars = bars[left:i]

        if fields is None:
            return bars
        return bars[fields]

    def is_suspended(self, order_book_id: str, dates) -> List[bool]:
        return [False] * len(dates)

    def is_st_stock(self, order_book_id: str, dates) -> List[bool]:
        return [False] * len(dates)

    def get_dividend(self, instrument: Instrument):
        return None

    def get_split(self, instrument: Instrument):
        return None

    def get_exchange_rate(self, trading_date, local: MARKET, settlement: MARKET = MARKET.CN) -> ExchangeRate:
        """港股账户以 HKD 计价，回测中汇率统一视为 1:1，避免依赖外部汇率数据源。"""
        return ExchangeRate(
            bid_reference=1.0, ask_reference=1.0,
            bid_settlement_sh=1.0, ask_settlement_sh=1.0,
            bid_settlement_sz=1.0, ask_settlement_sz=1.0,
        )

    def get_yield_curve(self, start_date, end_date, tenor=None):
        return None
