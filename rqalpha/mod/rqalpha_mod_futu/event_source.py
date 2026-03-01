# -*- coding: utf-8 -*-
"""
港股实盘事件源 (FutuLiveEventSource)

通过富途 OpenD 订阅实时 K 线回调，将到来的新 bar 转为 RQAlpha 事件推入队列，
主事件循环从队列中 yield 事件，驱动策略运行。

回测事件源由 sys_simulation (market=hk) 提供，无需在此实现。

港股交易时段（HKEX）
    上午盘：09:30 – 12:00
    下午盘：13:00 – 16:00
"""

import queue
import time
from datetime import datetime, time as dtime

from rqalpha.core.events import Event, EVENT
from rqalpha.interface import AbstractEventSource
from rqalpha.utils.logger import system_log

try:
    from futu import (
        OpenQuoteContext, SubType, RET_OK,
        CurKlineHandlerBase, KLType,
    )
    _FUTU_AVAILABLE = True
except ImportError:
    _FUTU_AVAILABLE = False

_HK_BEFORE_TRADING_TIME = dtime(8, 30)
_HK_AFTER_TRADING_TIME = dtime(16, 30)


class _BarHandler(CurKlineHandlerBase if _FUTU_AVAILABLE else object):
    """富途实时 K 线回调 → 将新 bar 写入事件队列。"""

    def __init__(self, event_queue: queue.Queue):
        if _FUTU_AVAILABLE:
            super().__init__()
        self._queue = event_queue

    def on_recv_rsp(self, rsp_str):
        ret, df = super().on_recv_rsp(rsp_str)  # type: ignore[misc]
        if ret != RET_OK:
            return ret, df
        # 仅处理 15m K 线（防止订阅多种周期时混淆）
        for _, row in df.iterrows():
            if row.get('k_type') not in (KLType.K_15M, 'K_15M'):
                continue
            try:
                dt = datetime.strptime(row['time_key'], '%Y-%m-%d %H:%M:%S')
            except Exception:
                continue
            # 把事件放入队列，主循环会 yield 出去
            self._queue.put(Event(EVENT.BAR, calendar_dt=dt, trading_dt=dt,
                                   code=row.get('code', '')))
        return ret, df


class FutuLiveEventSource(AbstractEventSource):
    """
    港股实盘事件源（基于富途 OpenD 实时 K 线订阅）。

    运行方式：
    1. 策略 init 后，通过 set_subscribe_codes() 告知要订阅的股票列表。
    2. events() 在主线程中运行；当富途回调线程推入新 bar 时自动 yield。
    3. 每个交易日开始前推送 BEFORE_TRADING，结束后推送 AFTER_TRADING。
    """

    def __init__(self, env, futu_data_source):
        if not _FUTU_AVAILABLE:
            raise RuntimeError("请先安装富途 API：pip install futu-api")
        self._env = env
        self._data_source = futu_data_source
        self._quote_ctx = None
        self._event_queue: queue.Queue = queue.Queue()
        self._subscribed_codes: list = []
        self._running = False

    def set_subscribe_codes(self, codes: list):
        self._subscribed_codes = codes

    def _subscribe(self):
        ctx = self._data_source._get_ctx()
        if self._subscribed_codes:
            ret, err = ctx.subscribe(self._subscribed_codes, [SubType.K_15M])
            if ret != RET_OK:
                system_log.warning(f"[Futu] 订阅实时行情失败: {err}")
        handler = _BarHandler(self._event_queue)
        ctx.set_handler(handler)
        self._quote_ctx = ctx

    def _unsubscribe(self):
        if self._quote_ctx and self._subscribed_codes:
            self._quote_ctx.unsubscribe(self._subscribed_codes, [SubType.K_15M])

    def events(self, start_date, end_date, frequency):
        """实盘模式：阻塞等待实时事件，按港股时间推送 BEFORE/AFTER_TRADING。"""
        self._subscribe()
        self._running = True
        trading_dates = self._data_source.get_hk_trading_dates(start_date, end_date)

        for day in trading_dates:
            trade_date = day.to_pydatetime().date() if hasattr(day, 'to_pydatetime') else day
            dt_before = datetime.combine(trade_date, _HK_BEFORE_TRADING_TIME)
            dt_after  = datetime.combine(trade_date, _HK_AFTER_TRADING_TIME)

            # 等待直到开盘前
            _wait_until(dt_before)
            yield Event(EVENT.BEFORE_TRADING, calendar_dt=dt_before, trading_dt=dt_before)

            # 等到本日 AFTER_TRADING 时间，从队列里取 BAR 事件
            while datetime.now() < dt_after:
                try:
                    event = self._event_queue.get(timeout=1)
                    if event.calendar_dt.date() == trade_date:
                        yield event
                except queue.Empty:
                    continue

            yield Event(EVENT.AFTER_TRADING, calendar_dt=dt_after, trading_dt=dt_after)

        self._unsubscribe()
        self._running = False


def _wait_until(target: datetime):
    """阻塞直到系统时间到达 target。"""
    now = datetime.now()
    if now < target:
        time.sleep((target - now).total_seconds())
