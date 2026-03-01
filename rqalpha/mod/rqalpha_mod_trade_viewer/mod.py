# -*- coding: utf-8 -*-

import datetime
from typing import Dict

import pandas as pd

from rqalpha.interface import AbstractMod
from rqalpha.core.events import EVENT
from rqalpha.utils.logger import system_log


class TradeViewerMod(AbstractMod):
    def __init__(self):
        self._env = None
        self._config = None
        self._trades = []      # list of dicts
        self._pf_records = []  # list of dicts

    def start_up(self, env, mod_config):
        self._env = env
        self._config = mod_config

        if not getattr(mod_config, 'auto_view', False):
            return

        env.event_bus.add_listener(EVENT.TRADE, self._on_trade)
        env.event_bus.prepend_listener(EVENT.POST_SETTLEMENT, self._on_settlement)

    def _on_trade(self, event):
        trade = event.trade
        try:
            instr = self._env.data_proxy.get_active_instrument(
                trade.order_book_id, trade.trading_datetime
            )
            symbol = instr.symbol if instr else ''
        except Exception:
            symbol = ''

        self._trades.append({
            'datetime':      trade.datetime,
            'order_book_id': trade.order_book_id,
            'symbol':        symbol,
            'side':          trade.side.name,
            'last_price':    trade.last_price,
            'last_quantity': trade.last_quantity,
        })

    def _on_settlement(self, _event):
        date      = self._env.calendar_dt.date()
        portfolio = self._env.portfolio
        self._pf_records.append({
            'date':          pd.Timestamp(date),
            'unit_net_value': float(portfolio.unit_net_value),
        })

    def tear_down(self, code, *args):
        auto_view = getattr(self._config, 'auto_view', False)
        print(f"[trade_viewer] tear_down: auto_view={auto_view}", flush=True)
        if not auto_view:
            return

        from .viewer import launch_viewer, decode_bar_array

        # ── 构建 trades DataFrame ──────────────────────────────────────────
        trades_df = pd.DataFrame(self._trades) if self._trades else pd.DataFrame()

        # ── 构建 portfolio DataFrame ───────────────────────────────────────
        portfolio_df = None
        if self._pf_records:
            portfolio_df = pd.DataFrame(self._pf_records).set_index('date')
            portfolio_df.index = pd.DatetimeIndex(portfolio_df.index)
            portfolio_df.index.name = 'date'

        # ── 预拉取 K 线 ───────────────────────────────────────────────────
        klines_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

        if not trades_df.empty:
            freq = self._env.config.base.frequency
            end_dt = pd.Timestamp(self._env.config.base.end_date) + pd.Timedelta(days=1)

            # 计算安全的 bar_count 上限（避免数据源不支持 None）
            start_dt = pd.Timestamp(self._env.config.base.start_date)
            days = max((end_dt - start_dt).days + 30, 30)
            bars_per_day = {'1d': 1, '1h': 7, '15m': 22}.get(freq, 22)
            bar_count = days * bars_per_day

            for order_book_id in trades_df['order_book_id'].unique():
                for fetch_freq, fetch_count in [(freq, bar_count), ('1d', days + 30)]:
                    # 避免重复拉取
                    if order_book_id in klines_cache and fetch_freq in klines_cache[order_book_id]:
                        continue
                    try:
                        bars = self._env.data_proxy.history_bars(
                            order_book_id,
                            fetch_count,
                            fetch_freq,
                            ['datetime', 'open', 'high', 'low', 'close', 'volume', 'total_turnover'],
                            end_dt,
                            skip_suspended=False,
                        )
                        if bars is not None and len(bars) > 0:
                            df = decode_bar_array(bars)
                            klines_cache.setdefault(order_book_id, {})[fetch_freq] = df
                    except Exception as e:
                        system_log.warning(
                            f"[trade_viewer] 拉取 {order_book_id} {fetch_freq} K线失败: {e}"
                        )

        result = {}
        if not trades_df.empty:
            result['trades'] = trades_df
        if portfolio_df is not None:
            result['portfolio'] = portfolio_df

        server_port = getattr(self._config, 'server_port', 8050)
        futu_host   = getattr(self._config, 'futu_host', '127.0.0.1')
        futu_port   = getattr(self._config, 'futu_port', 11112)

        system_log.info(f"[trade_viewer] 回测完成，启动可视化界面 → http://127.0.0.1:{server_port}")
        try:
            launch_viewer(
                result,
                klines_cache=klines_cache,
                futu_host=futu_host,
                futu_port=futu_port,
                server_port=server_port,
            )
        except Exception:
            import traceback
            traceback.print_exc()
