# -*- coding: utf-8 -*-
# 版权所有 2019 深圳米筐科技有限公司（下称“米筐科技”）
#
# 除非遵守当前许可，否则不得使用本软件。
#
#     * 非商业用途（非商业用途指个人出于非商业目的使用本软件，或者高校、研究所等非营利机构出于教育、科研等目的使用本软件）：
#         遵守 Apache License 2.0（下称“Apache 2.0 许可”），
#         您可以在以下位置获得 Apache 2.0 许可的副本：http://www.apache.org/licenses/LICENSE-2.0。
#         除非法律有要求或以书面形式达成协议，否则本软件分发时需保持当前许可“原样”不变，且不得附加任何条件。
#
#     * 商业用途（商业用途指个人出于任何商业目的使用本软件，或者法人或其他组织出于任何目的使用本软件）：
#         未经米筐科技授权，任何个人不得出于任何商业目的使用本软件（包括但不限于向第三方提供、销售、出租、出借、转让本软件、
#         本软件的衍生产品、引用或借鉴了本软件功能或源代码的产品或服务），任何法人或其他组织不得出于任何目的使用本软件，
#         否则米筐科技有权追究相应的知识产权侵权责任。
#         在此前提下，对本软件的使用同样需要遵守 Apache 2.0 许可，Apache 2.0 许可与本许可冲突之处，以本许可为准。
#         详细的授权流程，请联系 public@ricequant.com 获取。

from datetime import timedelta, datetime, time

from rqalpha.environment import Environment
from rqalpha.interface import AbstractEventSource
from rqalpha.core.events import Event, EVENT
from rqalpha.utils.exception import patch_user_exc
from rqalpha.utils.datetime_func import convert_int_to_datetime
from rqalpha.const import DEFAULT_ACCOUNT_TYPE, INSTRUMENT_TYPE
from rqalpha.utils.i18n import gettext as _



class SimulationEventSource(AbstractEventSource):
    def __init__(self, env, market='cn'):
        # type: (Environment, str) -> None
        self._env = env
        self._config = env.config
        self._universe_changed = False
        self._env.event_bus.add_listener(EVENT.POST_UNIVERSE_CHANGED, self._on_universe_changed)
        self._market = market

        if market == 'hk':
            self._get_day_bar_dt = lambda date: date.replace(hour=16, minute=0)
            self._get_after_trading_dt = lambda date: date.replace(hour=16, minute=30)
        else:
            self._get_day_bar_dt = lambda date: date.replace(hour=15, minute=0)
            self._get_after_trading_dt = lambda date: date.replace(hour=15, minute=30)

    def _on_universe_changed(self, _):
        self._universe_changed = True

    def _get_trading_dates(self, start_date, end_date):
        """根据市场类型获取交易日历。HK 市场优先使用数据源的 HK 日历。"""
        if self._market == 'hk':
            ds = getattr(self._env, 'data_source', None)
            if ds is not None and hasattr(ds, 'get_hk_trading_dates'):
                return ds.get_hk_trading_dates(start_date, end_date)
        return self._env.data_proxy.get_trading_dates(start_date, end_date)

    def _get_universe(self):
        universe = self._env.get_universe()
        if len(universe) == 0 and DEFAULT_ACCOUNT_TYPE.STOCK.name not in self._config.base.accounts:
            raise patch_user_exc(RuntimeError(_(
                "Current universe is empty. Please use subscribe function before trade"
            )), force=True)
        return universe

    # [BEGIN] minute event helper
    def _get_stock_trading_minutes(self, trading_date, step_minutes=1):
        """返回当日股票交易时段的时间点集合，支持 step_minutes 步长（如 15）。
        HK：09:30-12:00, 13:00-16:00；CN：09:31-11:30, 13:01-15:00。
        """
        delta = timedelta(minutes=step_minutes)
        if self._market == 'hk':
            sessions = [(time(9, 30), time(12, 0)), (time(13, 0), time(16, 0))]
        else:
            sessions = [(time(9, 31), time(11, 30)), (time(13, 1), time(15, 0))]

        trading_minutes = set()
        for start_t, end_t in sessions:
            current_dt = datetime.combine(trading_date, start_t)
            if step_minutes > 1:
                current_dt += delta   # 首根 bar 结束时刻 = 开盘 + step
            end_dt = datetime.combine(trading_date, end_t)
            while current_dt <= end_dt:
                trading_minutes.add(current_dt)
                current_dt += delta
        return trading_minutes

    def _get_future_trading_minutes(self, trading_date):
        trading_minutes = set()
        universe = self._get_universe()
        for order_book_id in universe:
            if self._env.get_account_type(order_book_id) == DEFAULT_ACCOUNT_TYPE.STOCK:
                continue
            trading_minutes.update(self._env.data_proxy.get_trading_minutes_for(order_book_id, trading_date))
        return set([convert_int_to_datetime(minute) for minute in trading_minutes])

    def _get_trading_minutes(self, trading_date, step_minutes=1):
        trading_minutes = set()
        for account_type in self._config.base.accounts:
            if account_type == DEFAULT_ACCOUNT_TYPE.STOCK:
                trading_minutes = trading_minutes.union(
                    self._get_stock_trading_minutes(trading_date, step_minutes))
            elif account_type == DEFAULT_ACCOUNT_TYPE.FUTURE:
                trading_minutes = trading_minutes.union(self._get_future_trading_minutes(trading_date))
        return sorted(list(trading_minutes))
    # [END] minute event helper

    def events(self, start_date, end_date, frequency):
        trading_dates = self._get_trading_dates(start_date, end_date)
        if frequency == "1d":
            # 根据起始日期和结束日期，获取所有的交易日，然后再循环获取每一个交易日
            for day in trading_dates:
                date = day.to_pydatetime()
                dt_before_trading = date.replace(hour=0, minute=0)

                dt_bar = self._get_day_bar_dt(date)
                dt_after_trading = self._get_after_trading_dt(date)

                yield Event(EVENT.BEFORE_TRADING, calendar_dt=dt_before_trading, trading_dt=dt_before_trading)
                yield Event(EVENT.OPEN_AUCTION, calendar_dt=dt_before_trading, trading_dt=dt_before_trading)
                yield Event(EVENT.BAR, calendar_dt=dt_bar, trading_dt=dt_bar)
                yield Event(EVENT.AFTER_TRADING, calendar_dt=dt_after_trading, trading_dt=dt_after_trading)
        elif frequency in ('1m', '15m'):
            step_minutes = 15 if frequency == '15m' else 1
            for day in trading_dates:
                before_trading_flag = True
                date = day.to_pydatetime()
                last_dt = None
                done = False

                dt_before_day_trading = date.replace(hour=8, minute=30)

                while True:
                    if done:
                        break
                    exit_loop = True
                    trading_minutes = self._get_trading_minutes(date, step_minutes)
                    for calendar_dt in trading_minutes:
                        if last_dt is not None and calendar_dt < last_dt:
                            continue

                        if calendar_dt < dt_before_day_trading:
                            trading_dt = calendar_dt.replace(year=date.year, month=date.month, day=date.day)
                        else:
                            trading_dt = calendar_dt
                        if before_trading_flag:
                            before_trading_flag = False
                            yield Event(
                                EVENT.BEFORE_TRADING,
                                calendar_dt=calendar_dt - timedelta(minutes=30),
                                trading_dt=trading_dt - timedelta(minutes=30)
                            )
                            yield Event(
                                EVENT.OPEN_AUCTION,
                                calendar_dt=calendar_dt - timedelta(minutes=3),
                                trading_dt=trading_dt - timedelta(minutes=3),
                            )
                        if self._universe_changed:
                            self._universe_changed = False
                            last_dt = calendar_dt
                            exit_loop = False
                            break
                        # yield handle bar
                        yield Event(EVENT.BAR, calendar_dt=calendar_dt, trading_dt=trading_dt)
                    if exit_loop:
                        done = True

                dt = self._get_after_trading_dt(date)
                yield Event(EVENT.AFTER_TRADING, calendar_dt=dt, trading_dt=dt)
        elif frequency == "tick":
            data_proxy = self._env.data_proxy
            for day in trading_dates:
                date = day.to_pydatetime()
                last_tick = None
                last_dt = None
                dt_before_day_trading = date.replace(hour=8, minute=30)
                while True:
                    for tick in data_proxy.get_merge_ticks(self._get_universe(), date, last_dt):
                        # find before trading time

                        calendar_dt = tick.datetime

                        if calendar_dt < dt_before_day_trading:
                            trading_dt = calendar_dt.replace(year=date.year, month=date.month, day=date.day)
                        else:
                            trading_dt = calendar_dt

                        if last_tick is None:
                            last_tick = tick

                            """
                            这里区分时间主要是为了对其之前，之前对获取tick数据的时间有限制，期货的盘前时间是20:30，股票是09:00。
                            在解除获取tick数据的限制后，股票的tick的开始时间是09:15，而期货则是20:59
                            """
                            if self._env.get_instrument(tick.order_book_id).type == INSTRUMENT_TYPE.FUTURE:
                                yield Event(
                                    EVENT.BEFORE_TRADING,
                                    calendar_dt=calendar_dt - timedelta(minutes=30),
                                    trading_dt=trading_dt - timedelta(minutes=30),
                                )
                            else:
                                yield Event(
                                    EVENT.BEFORE_TRADING,
                                    calendar_dt=calendar_dt - timedelta(minutes=15),
                                    trading_dt=trading_dt - timedelta(minutes=15),
                                )

                        if self._universe_changed:
                            self._universe_changed = False
                            break

                        last_dt = calendar_dt
                        yield Event(EVENT.TICK, calendar_dt=calendar_dt, trading_dt=trading_dt, tick=tick)

                    else:
                        break

                dt = self._get_after_trading_dt(date)
                yield Event(EVENT.AFTER_TRADING, calendar_dt=dt, trading_dt=dt)
        else:
            raise NotImplementedError(_("Frequency {} is not support.").format(frequency))
