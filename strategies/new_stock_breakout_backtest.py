# -*- coding: utf-8 -*-
"""
港股新股突破策略回测系统
使用富途API实现

策略逻辑：
1. 获取近一年上市的港股新股
2. 记录上市首日第一根15分钟K线的最高价和最低价
3. 买入信号（三阶段确认，仅限前BUY_DAY_LIMIT个交易日）：
   - 时间限制：只在上市后前BUY_DAY_LIMIT个交易日内允许买入
   - 阶段1：收盘价第一次突破首日最高价（记录，不买入）
   - 阶段2：回踩确认（15分钟K线RSI(RSI_PERIOD) < RSI_THRESHOLD）
   - 阶段3：回踩后再次突破首日最高价 且 突破15分钟K线MA5 → 买入
4. 卖出信号（满足任一条件即卖出）：
   - 跌破max(首日第一根K线最低价, 买入价*(1-STOP_LOSS_PCT))
   - 跌破日K线MA5
   - 日K线MA5下穿MA20（死叉）
5. 特殊规则：
   - 因死叉卖出的股票加入黑名单，不再交易
   - 盈亏比 = 总盈利 / 总亏损
6. 指标说明：
   - RSI：基于15分钟K线，周期=RSI_PERIOD，阈值=RSI_THRESHOLD
   - MA5/MA20：基于日K线，周期=MA5_PERIOD_DAILY/MA20_PERIOD_DAILY
   - 15分钟MA5：基于15分钟K线，周期=MA5_PERIOD_15M

所有配置参数见文件开头的配置区域，可根据需要调整。
"""

from futu import *
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import time
import logging
import os
import argparse


# ============================================================================
# 策略配置参数
# ============================================================================

# API连接配置
API_HOST = '127.0.0.1'
API_PORT = 11112
API_DELAY = 0.6  # API调用后延迟时间（秒），避免频率限制
STOCK_DELAY = 0.5  # 股票之间的延迟时间（秒）

# 资金配置
INITIAL_CAPITAL = 100000  # 初始资金（港币）

# 买入信号配置
RSI_PERIOD = 5  # RSI周期（基于15分钟K线）
RSI_THRESHOLD = 70  # RSI回踩阈值，低于此值确认回踩
MA5_PERIOD_15M = 5  # 15分钟K线MA5周期
BUY_DAY_LIMIT = 1000  # 只在上市后前N个交易日内允许买入

# 卖出信号配置
MA5_PERIOD_DAILY = 5  # 日K线MA5周期
MA20_PERIOD_DAILY = 20  # 日K线MA20周期
STOP_LOSS_PCT = 0.10  # 止损百分比（10% = 0.10）

# 回测配置
BACKTEST_DAYS = 90  # 回测天数（从今天往前推）
LOT_SIZE = 100  # 每手股数


class NewStockBreakoutBacktest:
    """新股突破策略回测系统"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, stop_loss_method: int = 1):
        """
        初始化回测系统

        参数：
            initial_capital: 初始资金
            stop_loss_method: 止损方法
                1 - 跌破max(首日最低价, 买入价-STOP_LOSS_PCT%)
                2 - 跌破5日均线
                3 - 两者都用（先触发哪个就执行）

        注意：无论选择哪种止损方法，5日均线下穿20日均线（死叉）都会触发卖出
        """
        self.quote_ctx = OpenQuoteContext(host=API_HOST, port=API_PORT)
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.stop_loss_method = stop_loss_method

        # 持仓记录
        self.positions: Dict[str, Dict] = {}  # {股票代码: {name, shares, cost, first_high, first_low}}

        # 交易记录
        self.trades: List[Dict] = []

        # 死叉黑名单：因死叉卖出的股票，不再交易
        self.death_cross_blacklist: set = set()

        # 买入状态跟踪：{股票代码: {first_breakout: bool, pullback_done: bool}}
        self.buy_state: Dict[str, Dict] = {}

        # 性能指标
        self.total_trades = 0
        self.win_trades = 0
        self.total_pnl = 0.0
        self.total_profit = 0.0  # 总盈利
        self.total_loss = 0.0    # 总亏损

        # 设置日志
        self.setup_logger()

    def __del__(self):
        """析构函数，关闭连接"""
        if hasattr(self, 'quote_ctx'):
            self.quote_ctx.close()

    def setup_logger(self):
        """设置日志记录器"""
        # 创建日志目录
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'backtest_{timestamp}.log')

        # 配置日志
        self.logger = logging.getLogger('NewStockBacktest')
        self.logger.setLevel(logging.INFO)

        # 移除已有的处理器
        self.logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        print(f"日志文件: {log_file}")

    def log_print(self, message: str):
        """同时打印到控制台和记录到日志"""
        print(message)
        self.logger.info(message)

    def get_new_stocks(self, days: int = BACKTEST_DAYS) -> List[str]:
        """
        获取近期上市的新股列表

        参数：
            days: 获取过去多少天上市的新股
        返回：
            新股代码列表
        """
        # 获取港股股票列表
        ret, data = self.quote_ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK)
        if ret != RET_OK:
            print(f"获取港股列表失败: {data}")
            return []

        all_stocks = data

        # 筛选新股
        cutoff_date = datetime.now() - timedelta(days=days)
        all_stocks['listing_date'] = pd.to_datetime(all_stocks['listing_date'])
        new_stocks = all_stocks[all_stocks['listing_date'] > cutoff_date]

        self.log_print(f"找到 {len(new_stocks)} 只近{days}天上市的港股新股")
        return new_stocks['code'].tolist()

    def get_first_day_kline(self, stock_code: str, list_date: str) -> Optional[Dict]:
        """
        获取上市首日第一根15分钟K线数据

        参数：
            stock_code: 股票代码
            list_date: 上市日期 (YYYY-MM-DD)
        返回：
            {high: 最高价, low: 最低价, close: 收盘价} 或 None
        """
        # 转换日期格式
        list_datetime = datetime.strptime(list_date, '%Y-%m-%d')

        # 获取上市首日的15分钟K线数据
        ret, data, page_req_key = self.quote_ctx.request_history_kline(
            stock_code,
            start=list_date,
            end=list_date,
            ktype=KLType.K_15M,
            autype=AuType.QFQ,
            max_count=100
        )

        # API调用后延迟，避免频率限制
        time.sleep(API_DELAY)

        if ret != RET_OK:
            print(f"获取 {stock_code} 上市首日K线失败: {data}")
            return None

        if data.empty:
            print(f"{stock_code} 上市首日无K线数据")
            return None

        # 获取第一根15分钟K线
        first_kline = data.iloc[0]

        return {
            'high': first_kline['high'],
            'low': first_kline['low'],
            'close': first_kline['close'],
            'time_key': first_kline['time_key']
        }

    def get_kline_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        ktype: KLType = KLType.K_15M
    ) -> Optional[pd.DataFrame]:
        """
        获取K线数据

        参数：
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            ktype: K线类型
        返回：
            K线数据DataFrame
        """
        ret, data, page_req_key = self.quote_ctx.request_history_kline(
            stock_code,
            start=start_date,
            end=end_date,
            ktype=ktype,
            autype=AuType.QFQ,
            max_count=1000
        )

        # API调用后延迟，避免频率限制
        time.sleep(API_DELAY)

        if ret != RET_OK:
            print(f"获取 {stock_code} K线数据失败: {data}")
            return None

        return data

    def calculate_ma5(self, df: pd.DataFrame) -> pd.Series:
        """
        计算5日均线（使用日K线数据）

        参数：
            df: 日K线数据
        返回：
            5日均线Series
        """
        return df['close'].rolling(window=5).mean()

    def calculate_ma20(self, df: pd.DataFrame) -> pd.Series:
        """
        计算20日均线（使用日K线数据）

        参数：
            df: 日K线数据
        返回：
            20日均线Series
        """
        return df['close'].rolling(window=20).mean()

    def calculate_rsi(self, df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        """
        计算RSI指标（使用日K线数据）

        参数：
            df: 日K线数据
            period: RSI周期，默认14天
        返回：
            RSI Series
        """
        # 计算价格变动
        delta = df['close'].diff()

        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 计算平均涨跌幅
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def check_buy_signal(
        self,
        stock_code: str,
        kline: pd.Series,
        first_high: float,
        rsi: Optional[float] = None,
        ma5_15m: Optional[float] = None
    ) -> bool:
        """
        检查买入信号：第一次突破 → 回踩（RSI < 70）→ 再次突破才买入

        参数：
            stock_code: 股票代码
            kline: 当前K线数据
            first_high: 首日第一根K线最高价
            rsi: 当前RSI值
            ma5_15m: 当前15分钟K线的5日均线
        返回：
            True表示产生买入信号
        """
        # 初始化该股票的买入状态
        if stock_code not in self.buy_state:
            self.buy_state[stock_code] = {
                'first_breakout': False,
                'pullback_done': False
            }

        state = self.buy_state[stock_code]
        current_price = kline['close']

        # 阶段1：检测第一次突破
        if not state['first_breakout']:
            if current_price > first_high:
                state['first_breakout'] = True
                self.log_print(f">>> {stock_code} 第一次突破 @ HK${current_price:.2f}，等待回踩...")
            return False

        # 阶段2：检测回踩（RSI < RSI_THRESHOLD）
        if state['first_breakout'] and not state['pullback_done']:
            if rsi is not None and rsi < RSI_THRESHOLD:
                state['pullback_done'] = True
                self.log_print(f">>> {stock_code} 回踩完成（RSI={rsi:.2f} < {RSI_THRESHOLD}），等待再次突破...")
            return False

        # 阶段3：回踩后再次突破首日最高价 且 突破15分钟MA5
        if state['first_breakout'] and state['pullback_done']:
            if current_price > first_high:
                # 必须同时突破15分钟MA5
                if ma5_15m is not None and current_price > ma5_15m:
                    return True
                elif ma5_15m is None:
                    # 如果MA5还未计算出来，仍然允许买入（兼容早期数据）
                    return True

        return False

    def check_sell_signal(
        self,
        stock_code: str,
        kline: pd.Series,
        first_low: float,
        buy_price: float,
        ma5: Optional[float] = None,
        ma20: Optional[float] = None,
        prev_ma5: Optional[float] = None,
        prev_ma20: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        检查卖出信号

        参数：
            stock_code: 股票代码
            kline: 当前K线数据
            first_low: 首日第一根K线最低价
            buy_price: 买入价格
            ma5: 当前5日均线值
            ma20: 当前20日均线值
            prev_ma5: 前一日5日均线值
            prev_ma20: 前一日20日均线值
        返回：
            (是否卖出, 卖出原因)
        """
        # 方法1：跌破首日最低价和买入价-STOP_LOSS_PCT%的高者
        if self.stop_loss_method in [1, 3]:
            # 计算止损价：取首日最低价和买入价-STOP_LOSS_PCT%中的较高值
            stop_loss_price = max(first_low, buy_price * (1 - STOP_LOSS_PCT))

            if kline['close'] < stop_loss_price:
                # 判断触发原因
                if stop_loss_price == first_low:
                    return True, "跌破首日最低价"
                else:
                    return True, f"跌破买入价-{STOP_LOSS_PCT*100:.0f}% (止损价:{stop_loss_price:.2f})"

        # 方法2：跌破5日均线
        if self.stop_loss_method in [2, 3]:
            if ma5 is not None and kline['close'] < ma5:
                return True, "跌破5日均线"

        # 新增：5日均线下穿20日均线（死叉）
        if ma5 is not None and ma20 is not None and prev_ma5 is not None and prev_ma20 is not None:
            # 前一日：5日均线 >= 20日均线（金叉或持平）
            # 当前日：5日均线 < 20日均线（死叉）
            if prev_ma5 >= prev_ma20 and ma5 < ma20:
                return True, "5日均线下穿20日均线(死叉)"

        return False, ""

    def execute_buy(
        self,
        stock_code: str,
        stock_name: str,
        price: float,
        first_high: float,
        first_low: float,
        date: str
    ) -> bool:
        """
        执行买入操作

        参数：
            stock_code: 股票代码
            stock_name: 股票名称
            price: 买入价格
            first_high: 首日最高价
            first_low: 首日最低价
            date: 交易日期
        返回：
            是否成功买入
        """
        # 检查是否在死叉黑名单中
        if stock_code in self.death_cross_blacklist:
            return False

        # 检查是否已持仓
        if stock_code in self.positions:
            return False

        # 每只股票固定使用INITIAL_CAPITAL资金
        fixed_amount = INITIAL_CAPITAL

        # 计算买入数量（港股最小LOT_SIZE股为一手）
        shares = int(fixed_amount / price / LOT_SIZE) * LOT_SIZE

        if shares == 0:
            print(f"股价过高，无法买入 {stock_code} {stock_name}")
            return False

        cost = shares * price

        # 更新持仓
        self.positions[stock_code] = {
            'name': stock_name,
            'shares': shares,
            'cost': price,
            'first_high': first_high,
            'first_low': first_low,
            'buy_date': date
        }

        # 更新现金
        self.cash -= cost

        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock_code,
            'name': stock_name,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'amount': cost,
            'cash': self.cash
        })

        msg = f"[{date}] 买入 {stock_code} {stock_name}: {shares}股 @ HK${price:.2f}, 成本 HK${cost:.2f}"
        self.log_print(msg)
        return True

    def execute_sell(
        self,
        stock_code: str,
        price: float,
        reason: str,
        date: str
    ) -> bool:
        """
        执行卖出操作

        参数：
            stock_code: 股票代码
            price: 卖出价格
            reason: 卖出原因
            date: 交易日期
        返回：
            是否成功卖出
        """
        if stock_code not in self.positions:
            return False

        position = self.positions[stock_code]
        stock_name = position['name']
        shares = position['shares']
        cost = position['cost']

        # 计算收益
        revenue = shares * price
        pnl = revenue - (shares * cost)
        pnl_pct = (price / cost - 1) * 100

        # 更新现金
        self.cash += revenue

        # 记录交易
        self.trades.append({
            'date': date,
            'stock': stock_code,
            'name': stock_name,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'amount': revenue,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'cash': self.cash
        })

        # 更新统计
        self.total_trades += 1
        if pnl > 0:
            self.win_trades += 1
            self.total_profit += pnl  # 累计盈利
        else:
            self.total_loss += abs(pnl)  # 累计亏损（取绝对值）
        self.total_pnl += pnl

        # 如果是因为死叉卖出，加入黑名单，永不再交易
        if "死叉" in reason:
            self.death_cross_blacklist.add(stock_code)
            self.log_print(f">>> {stock_code} {stock_name} 因死叉卖出，加入黑名单，不再交易")

        # 删除持仓
        del self.positions[stock_code]

        # 重置买入状态（允许未来重新交易，除非在黑名单中）
        if stock_code in self.buy_state:
            self.buy_state[stock_code] = {
                'first_breakout': False,
                'pullback_done': False
            }

        msg = f"[{date}] 卖出 {stock_code} {stock_name}: {shares}股 @ HK${price:.2f}, 盈亏 HK${pnl:.2f} ({pnl_pct:+.2f}%), 原因: {reason}"
        self.log_print(msg)
        return True

    def backtest_single_stock(
        self,
        stock_code: str,
        stock_name: str,
        list_date: str,
        end_date: str
    ) -> None:
        """
        回测单只股票

        参数：
            stock_code: 股票代码
            stock_name: 股票名称
            list_date: 上市日期
            end_date: 回测结束日期
        """
        self.log_print(f"\n{'='*60}")
        self.log_print(f"开始回测: {stock_code} {stock_name}, 上市日期: {list_date}")
        self.log_print(f"{'='*60}")

        # 获取上市首日第一根15分钟K线
        first_kline = self.get_first_day_kline(stock_code, list_date)
        if first_kline is None:
            print(f"无法获取 {stock_code} 首日K线，跳过")
            return

        first_high = first_kline['high']
        first_low = first_kline['low']
        first_time = first_kline['time_key']  # 第一根K线的时间
        self.log_print(f"首日第一根15分钟K线 ({first_time}): 最高 HK${first_high:.2f}, 最低 HK${first_low:.2f}")

        # 从上市当天开始获取K线（不是次日！）
        # 这样可以捕捉上市首日的突破机会
        start_date = list_date

        # 获取15分钟K线数据（从上市当天开始）
        kline_15m = self.get_kline_data(stock_code, start_date, end_date, KLType.K_15M)
        if kline_15m is None or kline_15m.empty:
            print(f"无法获取 {stock_code} K线数据，跳过")
            return

        # 获取日K线数据（用于计算均线）
        kline_daily = self.get_kline_data(stock_code, list_date, end_date, KLType.K_DAY)
        if kline_daily is None or kline_daily.empty:
            print(f"无法获取 {stock_code} 日K线数据，跳过")
            return

        # 计算日K线的5日和20日均线
        kline_daily['ma5'] = self.calculate_ma5(kline_daily)
        kline_daily['ma20'] = self.calculate_ma20(kline_daily)

        # 在15分钟K线上计算RSI（周期=RSI_PERIOD）和MA5
        kline_15m['rsi'] = self.calculate_rsi(kline_15m, period=RSI_PERIOD)
        kline_15m['ma5_15m'] = self.calculate_ma5(kline_15m)

        # 创建日期到日K线均线的映射
        ma5_dict = dict(zip(kline_daily['time_key'].str[:10], kline_daily['ma5']))
        ma20_dict = dict(zip(kline_daily['time_key'].str[:10], kline_daily['ma20']))

        # 创建time_key到15分钟K线指标的映射
        rsi_dict = dict(zip(kline_15m['time_key'], kline_15m['rsi']))
        ma5_15m_dict = dict(zip(kline_15m['time_key'], kline_15m['ma5_15m']))

        # 创建日期索引映射，用于获取前一日数据
        kline_daily['date'] = kline_daily['time_key'].str[:10]
        date_to_idx = {date: idx for idx, date in enumerate(kline_daily['date'].tolist())}

        # 回测主循环
        for idx, row in kline_15m.iterrows():
            # 跳过第一根K线（已经用来设置参考价格）
            if row['time_key'] == first_time:
                continue

            current_date = row['time_key'][:10]  # 提取日期部分
            current_time_key = row['time_key']  # 完整的时间键
            current_price = row['close']

            # 获取当日的日K线均线值
            ma5 = ma5_dict.get(current_date, None)
            ma20 = ma20_dict.get(current_date, None)

            # 获取当前15分钟K线的指标值
            rsi = rsi_dict.get(current_time_key, None)
            ma5_15m = ma5_15m_dict.get(current_time_key, None)

            # 获取前一日的均线值（用于判断死叉）
            prev_ma5 = None
            prev_ma20 = None
            current_idx = date_to_idx.get(current_date, None)
            if current_idx is not None and current_idx > 0:
                prev_date = kline_daily.iloc[current_idx - 1]['date']
                prev_ma5 = ma5_dict.get(prev_date, None)
                prev_ma20 = ma20_dict.get(prev_date, None)

            # 检查买入信号（仅在前5个交易日内）
            if stock_code not in self.positions:
                # 计算当前是上市后第几个交易日
                trading_day_num = current_idx + 1 if current_idx is not None else None

                # 只在前BUY_DAY_LIMIT个交易日内允许买入
                if trading_day_num is not None and trading_day_num <= BUY_DAY_LIMIT:
                    if self.check_buy_signal(stock_code, row, first_high, rsi, ma5_15m):
                        self.execute_buy(stock_code, stock_name, current_price, first_high, first_low, row['time_key'])
                elif trading_day_num is not None and trading_day_num == BUY_DAY_LIMIT + 1:
                    # 第BUY_DAY_LIMIT+1个交易日，记录一次提示
                    if stock_code not in self.buy_state or not self.buy_state[stock_code].get('day_limit_logged', False):
                        self.log_print(f">>> {stock_code} 已超过前{BUY_DAY_LIMIT}个交易日，不再尝试买入")
                        if stock_code not in self.buy_state:
                            self.buy_state[stock_code] = {}
                        self.buy_state[stock_code]['day_limit_logged'] = True

            # 检查卖出信号
            else:
                # 获取买入价
                buy_price = self.positions[stock_code]['cost']

                should_sell, reason = self.check_sell_signal(
                    stock_code, row, first_low, buy_price, ma5, ma20, prev_ma5, prev_ma20
                )
                if should_sell:
                    self.execute_sell(stock_code, current_price, reason, row['time_key'])

        # 如果回测结束时仍持仓，按最后价格平仓
        if stock_code in self.positions:
            last_price = kline_15m.iloc[-1]['close']
            last_date = kline_15m.iloc[-1]['time_key']
            self.execute_sell(stock_code, last_price, "回测结束平仓", last_date)

    def run_backtest(self, days: int = BACKTEST_DAYS, end_date: Optional[str] = None, target_stocks: Optional[List[str]] = None) -> None:
        """
        运行回测

        参数：
            days: 获取过去多少天上市的新股
            end_date: 回测结束日期，默认为今天
            target_stocks: 指定要回测的股票代码列表（可选），例如 ['HK.09988', 'HK.09618']
                          如果不指定，则回测所有找到的新股
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # 确定要回测的股票列表
        if target_stocks:
            # 如果指定了目标股票，使用指定列表
            new_stocks = target_stocks
            self.log_print(f"\n{'#'*60}")
            self.log_print(f"港股新股突破策略回测系统")
            self.log_print(f"初始资金: HK${self.initial_capital:,.2f}")
            self.log_print(f"止损方法: {['跌破max(首日最低价,买入价-10%)', '跌破5日均线', '两者都用'][self.stop_loss_method - 1]}")
            self.log_print(f"回测模式: 指定股票回测")
            self.log_print(f"目标股票: {', '.join(target_stocks)}")
            self.log_print(f"回测截止: {end_date}")
            self.log_print(f"{'#'*60}\n")
        else:
            # 否则获取所有新股
            self.log_print(f"\n{'#'*60}")
            self.log_print(f"港股新股突破策略回测系统")
            self.log_print(f"初始资金: HK${self.initial_capital:,.2f}")
            self.log_print(f"止损方法: {['跌破max(首日最低价,买入价-10%)', '跌破5日均线', '两者都用'][self.stop_loss_method - 1]}")
            self.log_print(f"回测区间: {days}天内上市的港股新股，截止 {end_date}")
            self.log_print(f"{'#'*60}\n")

            # 获取新股列表
            new_stocks = self.get_new_stocks(days)

        if not new_stocks:
            print("没有找到新股，回测结束")
            return

        # 获取港股基本信息以获得上市日期
        ret, stock_info = self.quote_ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK)
        if ret != RET_OK:
            print(f"获取港股基本信息失败: {stock_info}")
            return

        # 对每只新股进行回测
        for stock_code in new_stocks:  # 回测全部新股
            stock_data = stock_info[stock_info['code'] == stock_code]
            if stock_data.empty:
                continue

            list_date = stock_data.iloc[0]['listing_date'][:10]
            stock_name = stock_data.iloc[0]['name']

            self.backtest_single_stock(stock_code, stock_name, list_date, end_date)

            # 延时，避免API请求过快
            time.sleep(STOCK_DELAY)

        # 输出回测结果
        self.print_results()

    def print_results(self) -> None:
        """打印回测结果"""
        final_value = self.cash + sum(
            pos['shares'] * pos['cost'] for pos in self.positions.values()
        )
        total_return = (final_value / self.initial_capital - 1) * 100
        win_rate = (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        self.log_print(f"\n{'='*60}")
        self.log_print(f"港股回测结果统计")
        self.log_print(f"{'='*60}")
        self.log_print(f"初始资金:     HK${self.initial_capital:,.2f}")
        self.log_print(f"最终资金:     HK${self.cash:,.2f}")
        self.log_print(f"持仓市值:     HK${final_value - self.cash:,.2f}")
        self.log_print(f"总资产:       HK${final_value:,.2f}")
        self.log_print(f"总收益:       HK${final_value - self.initial_capital:,.2f}")
        self.log_print(f"收益率:       {total_return:+.2f}%")
        self.log_print(f"总交易次数:   {self.total_trades}")
        self.log_print(f"盈利次数:     {self.win_trades}")
        self.log_print(f"亏损次数:     {self.total_trades - self.win_trades}")
        self.log_print(f"胜率:         {win_rate:.2f}%")
        self.log_print(f"累计盈亏:     HK${self.total_pnl:,.2f}")
        self.log_print(f"总盈利:       HK${self.total_profit:,.2f}")
        self.log_print(f"总亏损:       HK${self.total_loss:,.2f}")

        # 计算盈亏比
        if self.total_loss > 0:
            profit_factor = self.total_profit / self.total_loss
            self.log_print(f"盈亏比:       {profit_factor:.2f}")
        else:
            self.log_print(f"盈亏比:       N/A (无亏损)")

        # 平均盈亏
        if self.win_trades > 0:
            avg_profit = self.total_profit / self.win_trades
            self.log_print(f"平均盈利:     HK${avg_profit:,.2f}")
        if self.total_trades - self.win_trades > 0:
            avg_loss = self.total_loss / (self.total_trades - self.win_trades)
            self.log_print(f"平均亏损:     HK${avg_loss:,.2f}")

        # 死叉黑名单统计
        if self.death_cross_blacklist:
            self.log_print(f"死叉黑名单:   {len(self.death_cross_blacklist)}只股票")

        self.log_print(f"{'='*60}\n")

        # 打印交易明细
        if self.trades:
            self.log_print("交易明细:")
            trades_df = pd.DataFrame(self.trades)
            trades_str = trades_df.to_string(index=False)
            self.log_print(trades_str)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='港股新股突破策略回测系统')
    parser.add_argument(
        '--stocks',
        type=str,
        help='指定要回测的股票代码，多个股票用逗号分隔，例如: 09988,09618 或 HK.09988,HK.09618'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='回测天数（从今天往前推），默认90天'
    )
    parser.add_argument(
        '--stop-loss-method',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='止损方法: 1-跌破max(首日最低价,买入价-10%%), 2-跌破5日均线, 3-两者都用，默认1'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='初始资金（港币），默认100000'
    )

    args = parser.parse_args()

    # 解析目标股票列表
    target_stocks = None
    if args.stocks:
        target_stocks = []
        for s in args.stocks.split(','):
            stock = s.strip()
            # 如果股票代码不包含"HK."前缀，自动添加
            if not stock.startswith('HK.'):
                stock = f'HK.{stock}'
            target_stocks.append(stock)
        print(f"指定回测股票: {', '.join(target_stocks)}")

    # 创建回测实例
    backtest = NewStockBreakoutBacktest(
        initial_capital=args.capital,
        stop_loss_method=args.stop_loss_method
    )

    try:
        # 运行回测
        backtest.run_backtest(days=args.days, target_stocks=target_stocks)
    except Exception as e:
        print(f"回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        del backtest


if __name__ == '__main__':
    main()
