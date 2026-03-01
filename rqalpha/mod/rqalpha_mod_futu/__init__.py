from .mod import FutuMod

__config__ = {
    "enabled": False,
    # 优先级：200 确保在 sys_simulation(100) 之后启动，从而覆盖其事件源
    "priority": 200,
    # OpenD 连接配置
    "host": "127.0.0.1",
    "port": 11112,
    # API 调用间隔（秒），防止超出富途频率限制
    "api_delay": 0.3,
    # K 线磁盘缓存目录（空字符串表示不缓存）
    # 缓存后，每次只增量拉取最新数据，可视化工具也可离线使用
    "cache_dir": "~/.rqalpha/futu_cache",
    # 实盘交易配置（run_type=r 时生效）
    "trade_env": "SIMULATE",   # "REAL" 或 "SIMULATE"（模拟盘）
    "trade_password": "",      # 交易密码（实盘需要）
    "trade_account": 0,        # 账户 ID，0 表示自动选取第一个账户
}


def load_mod():
    return FutuMod()
