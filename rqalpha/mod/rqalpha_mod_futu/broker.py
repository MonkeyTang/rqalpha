# -*- coding: utf-8 -*-
"""
富途实盘券商 (FutuBroker)

实现 AbstractBroker 接口，通过富途 OpenSecTradeContext 下单、撤单、查询持仓。
仅在 run_type=r（实盘）时启用；回测模式继续使用 sys_simulation 的 SimulationBroker。

注意：
- 富途实盘需要 OpenD 运行且已通过实名认证
- trade_env = 'SIMULATE' 时使用富途模拟盘（不占用真实资金）
- 订单匹配通过交易所完成，RQAlpha 侧通过事件总线广播成交/状态变更
"""

import threading
import time
from typing import Dict, List, Optional

from rqalpha.const import ORDER_STATUS, SIDE
from rqalpha.core.events import EVENT, Event
from rqalpha.interface import AbstractBroker
from rqalpha.model.order import Order
from rqalpha.utils.logger import system_log

try:
    from futu import (
        OpenSecTradeContext, TrdEnv, TrdMarket, OrderType, TrdSide,
        RET_OK, ModifyOrderOp,
    )
    _FUTU_AVAILABLE = True
except ImportError:
    _FUTU_AVAILABLE = False


def _to_trd_env(trade_env_str: str) -> "TrdEnv":
    return TrdEnv.REAL if trade_env_str.upper() == 'REAL' else TrdEnv.SIMULATE


def _to_trd_side(side: SIDE) -> "TrdSide":
    return TrdSide.BUY if side == SIDE.BUY else TrdSide.SELL


class FutuBroker(AbstractBroker):
    """
    富途实盘 Broker。

    下单后启动轮询线程监控订单状态，当订单成交时通过 EventBus 推送 EVENT.TRADE。
    """

    POLL_INTERVAL = 2.0  # 秒，订单状态轮询间隔

    def __init__(self, env, mod_config):
        if not _FUTU_AVAILABLE:
            raise RuntimeError("请先安装富途 API：pip install futu-api")
        self._env = env
        self._trade_env = _to_trd_env(mod_config.trade_env)
        self._password = str(mod_config.trade_password)
        self._account_id = mod_config.trade_account

        self._trd_ctx: Optional[OpenSecTradeContext] = None
        self._lock = threading.Lock()
        # {rqalpha_order_id: futu_order_id}
        self._order_map: Dict[int, str] = {}
        # {rqalpha_order_id: Order}
        self._open_orders: Dict[int, Order] = {}

        self._poll_thread: Optional[threading.Thread] = None
        self._running = False

    # ── 连接管理 ──────────────────────────────────────────────────────────────

    def _get_trd_ctx(self) -> "OpenSecTradeContext":
        if self._trd_ctx is None:
            self._trd_ctx = OpenSecTradeContext(
                filter_trdmarket=TrdMarket.HK,
                host=self._env.config.mod.futu.host,
                port=self._env.config.mod.futu.port,
                is_encrypt=False,
            )
            if self._password:
                self._trd_ctx.unlock_trade(self._password)
        return self._trd_ctx

    def start_up(self):
        """启动订单轮询线程。"""
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_orders, daemon=True, name='FutuBrokerPoller'
        )
        self._poll_thread.start()

    def tear_down(self):
        self._running = False
        if self._trd_ctx:
            self._trd_ctx.close()

    # ── AbstractBroker 接口 ───────────────────────────────────────────────────

    def submit_order(self, order: Order):
        if order.side == SIDE.BUY:
            trd_side = TrdSide.BUY
        else:
            trd_side = TrdSide.SELL

        ctx = self._get_trd_ctx()
        ret, data = ctx.place_order(
            price=float(order.price) if order.price else 0,
            qty=int(order.quantity),
            code=order.order_book_id,
            trd_side=trd_side,
            order_type=OrderType.MARKET if order.price is None else OrderType.NORMAL,
            trd_env=self._trade_env,
            acc_id=self._account_id,
        )
        time.sleep(0.1)
        if ret != RET_OK:
            system_log.error(f"[FutuBroker] 下单失败 {order.order_book_id}: {data}")
            order._status = ORDER_STATUS.REJECTED
            self._env.event_bus.publish_event(Event(EVENT.ORDER_REJECTED, order=order))
            return

        futu_order_id = str(data['orderid'].iloc[0])
        with self._lock:
            self._order_map[order.order_id] = futu_order_id
            self._open_orders[order.order_id] = order

        order._status = ORDER_STATUS.ACTIVE
        self._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, order=order))
        system_log.info(f"[FutuBroker] 下单成功 {order.order_book_id} qty={order.quantity} futu_id={futu_order_id}")

    def cancel_order(self, order: Order):
        futu_id = self._order_map.get(order.order_id)
        if futu_id is None:
            return
        ctx = self._get_trd_ctx()
        ret, data = ctx.modify_order(
            modify_order_op=ModifyOrderOp.CANCEL,
            order_id=futu_id,
            qty=0,
            price=0,
            trd_env=self._trade_env,
            acc_id=self._account_id,
        )
        time.sleep(0.1)
        if ret != RET_OK:
            system_log.warning(f"[FutuBroker] 撤单失败 {order.order_book_id}: {data}")
        else:
            order._status = ORDER_STATUS.CANCELLED
            self._env.event_bus.publish_event(Event(EVENT.ORDER_CANCELLATION_PASS, order=order))

    def get_open_orders(self, order_book_id=None) -> List[Order]:
        with self._lock:
            orders = list(self._open_orders.values())
        if order_book_id:
            orders = [o for o in orders if o.order_book_id == order_book_id]
        return orders

    # ── 订单状态轮询 ──────────────────────────────────────────────────────────

    def _poll_orders(self):
        while self._running:
            try:
                self._check_order_status()
            except Exception as e:
                system_log.warning(f"[FutuBroker] 轮询异常: {e}")
            time.sleep(self.POLL_INTERVAL)

    def _check_order_status(self):
        with self._lock:
            if not self._open_orders:
                return
            order_ids = list(self._order_map.items())  # [(rq_id, futu_id)]

        ctx = self._get_trd_ctx()
        ret, df = ctx.order_list_query(
            order_id='',
            trd_env=self._trade_env,
            acc_id=self._account_id,
            refresh_cache=True,
        )
        time.sleep(0.1)
        if ret != RET_OK or df is None or df.empty:
            return

        futu_id_to_row = {str(r['orderid']): r for _, r in df.iterrows()}
        for rq_id, futu_id in order_ids:
            row = futu_id_to_row.get(futu_id)
            if row is None:
                continue
            order_status = str(row.get('order_status', ''))
            order = self._open_orders.get(rq_id)
            if order is None:
                continue

            # 富途状态 -> RQAlpha 事件
            if 'FILLED_ALL' in order_status or order_status == '3':
                filled_qty  = int(row.get('dealt_qty', order.quantity))
                filled_price = float(row.get('dealt_avg_price', row.get('price', 0)))
                self._on_trade(order, filled_qty, filled_price)
            elif 'CANCELLED' in order_status or order_status in ('6', '7', '8', '9'):
                order._status = ORDER_STATUS.CANCELLED
                self._env.event_bus.publish_event(Event(EVENT.ORDER_CANCELLATION_PASS, order=order))
                with self._lock:
                    self._open_orders.pop(rq_id, None)

    def _on_trade(self, order: Order, filled_qty: int, filled_price: float):
        from rqalpha.model.trade import Trade
        trade = Trade.__from_create__(
            order_id=order.order_id,
            price=filled_price,
            amount=filled_qty,
            side=order.side,
            position_effect=order.position_effect,
            order_book_id=order.order_book_id,
            frozen_price=order.frozen_price,
        )
        order._status = ORDER_STATUS.FILLED
        self._env.event_bus.publish_event(Event(EVENT.TRADE, trade=trade, order=order))
        with self._lock:
            self._open_orders.pop(order.order_id, None)
