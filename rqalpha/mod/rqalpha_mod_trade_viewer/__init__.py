# -*- coding: utf-8 -*-

import click

from rqalpha.utils.i18n import gettext as _
from rqalpha import inject_run_param

__config__ = {
    # 回测结束后是否自动打开可视化界面
    "auto_view": False,
    # 可视化服务监听端口
    "server_port": 8050,
    # Futu OpenD 地址（klines_cache 缺失时兜底拉取 K 线）
    "futu_host": "127.0.0.1",
    # Futu OpenD 端口
    "futu_port": 11112,
    # 模块优先级：比 futu(200) 高，确保 tear_down 时数据源仍可用
    "priority": 250,
}


def load_mod():
    from .mod import TradeViewerMod
    return TradeViewerMod()


inject_run_param(click.Option(
    ('-V', '--view', 'mod__trade_viewer__auto_view'),
    is_flag=True, default=False,
    help=_("[trade_viewer] launch visualization after backtest"),
))
