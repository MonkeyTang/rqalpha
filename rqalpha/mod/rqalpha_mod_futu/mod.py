# -*- coding: utf-8 -*-
"""
富途 Mod (FutuMod)

根据运行类型（回测 / 实盘）分别装配数据源、事件源和券商。

  回测 (run_type=b / p):
    data_source  → FutuDataSource   (从富途 API 拉取历史 K 线)
    event_source → sys_simulation   (market=hk，由 SimulationEventSource 提供港股时段)
    broker       → SimulationBroker (sys_simulation，保持不变)

  实盘 (run_type=r):
    data_source  → FutuDataSource
    event_source → FutuLiveEventSource  (实时 K 线订阅)
    broker       → FutuBroker           (富途下单 / 撤单)
"""

from rqalpha.const import RUN_TYPE, INSTRUMENT_TYPE, MARKET
from rqalpha.interface import AbstractMod, AbstractTransactionCostDecider, TransactionCost
from rqalpha.data.bar_dict_price_board import BarDictPriceBoard
from rqalpha.utils.logger import system_log

from .data_source import FutuDataSource
from .event_source import FutuLiveEventSource


class _ZeroCostDecider(AbstractTransactionCostDecider):
    """港股回测中费率置零，避免依赖外部费率数据源。"""
    def calc(self, args):
        return TransactionCost.zero()


class FutuMod(AbstractMod):

    def __init__(self):
        self._data_source: FutuDataSource = None
        self._broker = None

    def start_up(self, env, mod_config):
        system_log.info("[FutuMod] 启动，连接 OpenD {}:{}".format(
            mod_config.host, mod_config.port
        ))

        # ── 1. 数据源（回测 + 实盘均使用）────────────────────────────────────
        self._data_source = FutuDataSource(
            host=mod_config.host,
            port=mod_config.port,
            api_delay=mod_config.api_delay,
            cache_dir=getattr(mod_config, 'cache_dir', '~/.rqalpha/futu_cache'),
        )
        env.set_data_source(self._data_source)

        # PriceBoard：使用 bar 字典维护最新价，沿用默认实现
        if not hasattr(env, 'price_board') or env.price_board is None:
            env.price_board = BarDictPriceBoard()

        # ── 注册港股交易费率（零费率，避免依赖外部手续费数据）──────────────────
        _zero = _ZeroCostDecider()
        env.set_transaction_cost_decider(INSTRUMENT_TYPE.CS, _zero, market=MARKET.HK)

        # ── 2. 事件源 ─────────────────────────────────────────────────────────
        run_type = env.config.base.run_type
        if run_type in (RUN_TYPE.BACKTEST, RUN_TYPE.PAPER_TRADING):
            system_log.info("[FutuMod] 回测模式：事件源由 sys_simulation (market=hk) 提供")
        else:
            # 实盘
            live_source = FutuLiveEventSource(env, self._data_source)
            env.set_event_source(live_source)
            system_log.info("[FutuMod] 实盘模式：使用富途实时 K 线事件源")

            # ── 3. Broker（仅实盘）──────────────────────────────────────────
            from .broker import FutuBroker
            self._broker = FutuBroker(env, mod_config)
            env.set_broker(self._broker)
            self._broker.start_up()
            system_log.info("[FutuMod] 实盘 Broker 已启动")

    def tear_down(self, code, exception=None):
        if self._broker is not None:
            self._broker.tear_down()
        if self._data_source is not None:
            self._data_source.close()
        system_log.info("[FutuMod] 已关闭")
