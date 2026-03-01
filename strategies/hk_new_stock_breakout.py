# -*- coding: utf-8 -*-
"""
港股新股突破策略 (RQAlpha 版)
=====================================

策略逻辑
--------
1. 初始化阶段（init）：
   - 回测模式：预加载整个回测时间段内所有新上市港股，缓存为 {date: [codes]} 日历。
   - 实盘模式：不做预加载，每日 before_trading 调用 API 获取当日新股。

2. 每日开盘前（before_trading）：
   - 回测模式：从缓存日历取出今日新上市股票。
   - 实盘模式：调用 FutuDataSource.get_new_stocks_for_date(today) 获取当日新股。
   - 对新发现的股票执行订阅并加入 active_codes。

3. 首日首根 K 线（handle_bar）：
   - 对尚未记录 first_bars 的股票，捕获当根 K 线的 high/low 作为 first_high/first_low。
   - 首根 K 线当根不入场。

4. 买入信号（三阶段确认）：
   - 阶段 1：收盘价第一次突破 first_high（记录，不买入）
   - 阶段 2：回踩确认，RSI(RSI_PERIOD) < RSI_THRESHOLD（基于 15m K 线）
   - 阶段 3：回踩后再次突破 first_high 且价格 > 15m MA5 → 买入

5. 卖出信号（满足任一条件即卖出）：
   - 跌破 max(first_low, 买入价 × (1 - STOP_LOSS_PCT))
   - 跌破日线 MA5（可选，取决于 STOP_LOSS_METHOD）
   - 日线 MA5 死叉 MA20 → 卖出并加入黑名单

使用方式
--------
回测：
    rqalpha run -f hk_new_stock_breakout.py \\
        -s 2024-01-01 -e 2024-12-31 \\
        -fq 15m -a stock 1000000 \\
        --config strategy_config.yml

strategy_config.yml 示例::

    mod:
      futu:
        enabled: true
        host: "127.0.0.1"
        port: 11112

    extra:
      context_vars:
        stop_loss_method: 1   # 1=首日低/固定比例, 2=MA5, 3=两者都用
"""

from __future__ import annotations

import numpy as np
from rqalpha.apis import *  # noqa: F401,F403 — injects logger, history_bars, order_shares, subscribe, etc.

# ── 策略参数 ────────────────────────────────────────────────────────────────
RSI_PERIOD        = 5      # RSI 计算周期（基于 15m K 线）
RSI_THRESHOLD     = 70     # 回踩阈值：RSI 低于此值视为完成回踩
MA5_PERIOD_15M    = 5      # 15m K 线 MA5 周期
MA5_PERIOD_DAILY  = 5      # 日线 MA5 周期
MA20_PERIOD_DAILY = 20     # 日线 MA20 周期
STOP_LOSS_PCT     = 0.10   # 固定止损比例 10%
LOT_SIZE          = 100    # 港股每手股数（若 Instrument 提供 board_lot 则使用 board_lot）
PER_STOCK_CAPITAL = 100000 # 每只股票固定动用资金（港币）
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
#  指标计算工具
# ════════════════════════════════════════════════════════════════════════════

def _calc_rsi(closes: np.ndarray, period: int) -> float:
    """返回最后一个 RSI 值；数据不足时返回 50（中性）。"""
    if len(closes) < period + 1:
        return 50.0
    delta = np.diff(closes.astype(float))
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _calc_ma(closes: np.ndarray, period: int) -> float:
    """返回最后 period 根 K 线的简单均线；数据不足时返回 NaN。"""
    if len(closes) < period:
        return float('nan')
    return float(np.mean(closes[-period:]))


# ════════════════════════════════════════════════════════════════════════════
#  RQAlpha 策略入口
# ════════════════════════════════════════════════════════════════════════════

def init(context):
    """
    策略初始化：
    - 回测模式：从数据源拉取所有合约，构建回测期内新股上市日历 {date: [codes]}。
    - 实盘模式：不做预加载，每日 before_trading 调用 API 获取当日新股。
    """
    context.stop_loss_method = int(getattr(context, 'stop_loss_method', 1))

    # 内部状态
    context.first_bars   = {}    # {code: {'high': float, 'low': float}}
    context.buy_state    = {}    # {code: {'first_breakout': bool, 'pullback_done': bool}}
    context.blacklist    = set() # 死叉黑名单
    context.active_codes = set() # 当前已订阅、正在追踪的股票集合

    from rqalpha.environment import Environment
    from rqalpha.const import RUN_TYPE
    env = Environment.get_instance()
    ds  = env.data_source
    run_type = env.config.base.run_type
    context.is_backtest = run_type in (RUN_TYPE.BACKTEST, RUN_TYPE.PAPER_TRADING)

    if context.is_backtest:
        start = env.config.base.start_date
        end   = env.config.base.end_date
        if hasattr(start, 'date'): start = start.date()
        if hasattr(end,   'date'): end   = end.date()
        context.new_stock_calendar = _build_new_stock_calendar(ds, start, end)
        total = sum(len(v) for v in context.new_stock_calendar.values())
        logger.info(f"回测模式：发现回测期内新股 {total} 只，分布在 {len(context.new_stock_calendar)} 个交易日")
    else:
        context.new_stock_calendar = {}
        logger.info("实盘模式：每日 before_trading 调用 API 获取当日新股")

    logger.info("策略初始化完成，等待每日新股上市触发订阅")


def _build_new_stock_calendar(ds, start_date, end_date) -> dict:
    """
    遍历数据源所有合约，将回测期间内每日新上市股票聚合为 {date: [codes]} 字典。
    仅供回测模式的 init 调用，用作缓存。
    """
    from datetime import date as date_type
    instruments = list(ds.get_instruments())
    calendar = {}
    for ins in instruments:
        try:
            ld = date_type.fromisoformat(str(ins.listed_date)[:10])
        except Exception:
            continue
        if start_date <= ld <= end_date:
            calendar.setdefault(ld, []).append(ins.order_book_id)
    for codes in calendar.values():
        codes.sort()
    return calendar


# ── 每日开盘前 ──────────────────────────────────────────────────────────────

def before_trading(context):
    """
    每日开盘前触发：
    - 从缓存（回测）或 API（实盘）获取今日新上市港股。
    - 对新股执行订阅、预拉取历史数据，并加入 active_codes。
    """
    today = context.now.date()
    from rqalpha.environment import Environment
    ds = Environment.get_instance().data_source

    if context.is_backtest:
        new_today = context.new_stock_calendar.get(today, [])
    else:
        new_today = ds.get_new_stocks_for_date(today) if hasattr(ds, 'get_new_stocks_for_date') else []

    for code in new_today:
        if code in context.active_codes or code in context.blacklist:
            continue
        context.active_codes.add(code)
        subscribe(code)
        if hasattr(ds, 'prefetch_stocks'):
            try:
                ds.prefetch_stocks([code], frequency='15m')
            except Exception as e:
                logger.warning(f"预拉取 {code} 失败（非致命）: {e}")
        logger.info(f"[新股上市] {code} 今日上市，已订阅")

    if new_today:
        logger.info(f"今日新股 {len(new_today)} 只: {new_today}")


# ── 主循环 ─────────────────────────────────────────────────────────────────

def handle_bar(context, bar_dict):
    for code in list(context.active_codes):
        if code in context.blacklist:
            continue
        if code not in bar_dict:
            continue

        bar = bar_dict[code]
        cur_close = float(bar.close)

        # 捕获上市首日第一根 K 线的高低价，本根不入场
        if code not in context.first_bars:
            context.first_bars[code] = {
                'high': float(bar.high),
                'low':  float(bar.low),
            }
            logger.info(
                f"[首日首根K线] {code} high={bar.high:.2f} low={bar.low:.2f}"
            )
            continue

        first_high = context.first_bars[code]['high']
        first_low  = context.first_bars[code]['low']

        # 15m 数据：买入三阶段所需
        bars_15m = history_bars(code, MA5_PERIOD_15M + RSI_PERIOD + 5, '15m', 'close')
        if bars_15m is None or len(bars_15m) < RSI_PERIOD + 1:
            continue

        rsi     = _calc_rsi(bars_15m, RSI_PERIOD)
        ma5_15m = _calc_ma(bars_15m,  MA5_PERIOD_15M)

        # ── 已持仓：检查卖出信号（需要日线数据）────────────────────────────
        position = context.portfolio.positions.get(code)
        if position and position.quantity > 0:
            bars_daily = history_bars(code, MA20_PERIOD_DAILY + 5, '1d', 'close')
            if bars_daily is None or len(bars_daily) < 2:
                continue  # 日线不足，暂不检查卖出
            ma5_d       = _calc_ma(bars_daily, MA5_PERIOD_DAILY)
            ma20_d      = _calc_ma(bars_daily, MA20_PERIOD_DAILY)
            prev_ma5_d  = _calc_ma(bars_daily[:-1], MA5_PERIOD_DAILY)
            prev_ma20_d = _calc_ma(bars_daily[:-1], MA20_PERIOD_DAILY)
            sold, reason = _check_sell(
                context, code, cur_close,
                first_low, position.avg_price,
                ma5_d, ma20_d, prev_ma5_d, prev_ma20_d,
            )
            if sold:
                result = order_target_percent(code, 0)
                if result is not None:
                    logger.info(f"[卖出] {code} @ {cur_close:.2f}，原因: {reason}")
                    context.buy_state[code] = {'first_breakout': False, 'pullback_done': False}
                    if "死叉" in reason:
                        context.blacklist.add(code)
                        logger.info(f"{code} 加入黑名单（死叉）")
        else:
            # ── 未持仓：检查买入信号（仅需 15m 数据）───────────────────────
            _check_buy(context, code, cur_close, first_high, rsi, ma5_15m)


def after_trading(context):
    p = context.portfolio
    logger.info(
        f"[日结] 净值={p.total_value:,.0f}  现金={p.cash:,.0f}  "
        f"收益率={p.total_returns*100:.2f}%  活跃股票={len(context.active_codes)}"
    )


def _check_buy(context, code, cur_close, first_high, rsi, ma5_15m):
    if code not in context.buy_state:
        context.buy_state[code] = {'first_breakout': False, 'pullback_done': False}

    state = context.buy_state[code]

    # 阶段 1：第一次突破 first_high
    if not state['first_breakout']:
        if cur_close > first_high:
            state['first_breakout'] = True
            logger.info(f"[阶段1] {code} 首次突破 {first_high:.2f}")
        return

    # 阶段 2：RSI 回踩确认
    if not state['pullback_done']:
        if rsi < RSI_THRESHOLD:
            state['pullback_done'] = True
            logger.info(f"[阶段2] {code} 回踩完成 RSI={rsi:.1f}")
        return

    # 阶段 3：再次突破 first_high 且 > MA5(15m) → 买入
    if cur_close > first_high and (np.isnan(ma5_15m) or cur_close > ma5_15m):
        lot = LOT_SIZE
        try:
            from rqalpha.environment import Environment
            ins_list = Environment.get_instance().data_proxy.get_instrument_history(code)
            ins = ins_list[0] if ins_list else None
            lot = getattr(ins, 'board_lot', LOT_SIZE) or LOT_SIZE
        except Exception:
            pass
        shares = int(PER_STOCK_CAPITAL / cur_close / lot) * lot
        if shares > 0:
            order_shares(code, shares)
            logger.info(f"[买入] {code} {shares}股 @ {cur_close:.2f} (阶段3突破)")
            context.buy_state[code] = {'first_breakout': False, 'pullback_done': False}


def _check_sell(context, code, cur_close, first_low, buy_price,
                ma5_d, ma20_d, prev_ma5_d, prev_ma20_d):
    """返回 (should_sell: bool, reason: str)。"""
    method = context.stop_loss_method

    # 方法 1 / 3：跌破 max(首日最低价, 买入价 × (1 - STOP_LOSS_PCT))
    if method in (1, 3):
        stop = max(first_low, buy_price * (1 - STOP_LOSS_PCT))
        if cur_close < stop:
            reason = "跌破首日最低价" if stop == first_low else f"跌破固定止损价 {stop:.2f}"
            return True, reason

    # 方法 2 / 3：跌破日线 MA5
    if method in (2, 3):
        if not np.isnan(ma5_d) and cur_close < ma5_d:
            return True, "跌破日线 MA5"

    # 日线死叉（MA5 下穿 MA20），无论 stop_loss_method
    if (not np.isnan(ma5_d) and not np.isnan(ma20_d)
            and not np.isnan(prev_ma5_d) and not np.isnan(prev_ma20_d)):
        if prev_ma5_d >= prev_ma20_d and ma5_d < ma20_d:
            return True, "日线 MA5 死叉 MA20"

    return False, ""
