# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable):**
```bash
pip install -e .
```

**Run a backtest:**
```bash
rqalpha run -f path/to/strategy.py -s 2015-01-01 -e 2016-01-01 -a stock 1000000
```

**Run all tests:**
```bash
pytest tests
```

**Run a single test:**
```bash
pytest tests/integration_tests/test_backtest_results/test_s_buy_and_hold.py
pytest tests/unittest/test_config.py::TestConfig::test_method
```

**Build docs:**
```bash
make -C docs html
```

**Mod management:**
```bash
rqalpha mod list
rqalpha mod enable <mod_name>
rqalpha mod disable <mod_name>
```

## Architecture

RQAlpha is an event-driven algorithmic trading framework supporting backtesting and live trading of stocks and futures in Chinese markets.

### Runtime Flow

`main.py:run()` bootstraps everything:
1. Creates an `Environment` singleton (global service registry)
2. Loads and starts Mods via `ModHandler`
3. Sets up data source, price board, data proxy
4. Loads the strategy file and builds a `scope` dict with all API functions
5. Calls `Executor.run()`, which drives the main event loop

### Core Components

- **`environment.py`** — Singleton (`Environment._env`) that holds all runtime services: `data_proxy`, `data_source`, `price_board`, `event_source`, `broker`, `portfolio`, `mod_dict`. Mods attach their implementations here during `start_up`.

- **`core/events.py`** — `EventBus` dispatches `Event` objects keyed by `EVENT` enum values. System listeners run first; user listeners run after. A listener returning `True` stops further propagation.

- **`core/executor.py`** — Main loop: iterates over events from `env.event_source` and routes `BAR`, `TICK`, `OPEN_AUCTION`, `BEFORE_TRADING`, and `AFTER_TRADING` events, publishing them through the event bus.

- **`core/strategy.py`** — Connects user strategy callbacks (`init`, `before_trading`, `handle_bar`, `handle_tick`, `open_auction`, `after_trading`) to `EventBus` listeners. Each callback has pre/post hook phases for mods.

- **`interface.py`** — Abstract base classes for all extension points: `AbstractDataSource`, `AbstractBroker`, `AbstractEventSource`, `AbstractPriceBoard`, `AbstractMod`, `AbstractPosition`, etc. New integrations must implement these.

- **`data/data_proxy.py`** — Thin caching layer over `AbstractDataSource`; used throughout the codebase for all market data queries.

### Mod System

Mods are the primary extension mechanism. Each mod is a Python package exposing a `load_mod()` function returning an `AbstractMod` subclass. Mods register services on `Environment` in `start_up()` and clean up in `tear_down()`.

Built-in system mods (under `rqalpha/mod/`):

| Mod | Purpose |
|-----|---------|
| `sys_accounts` | Stock/futures position models and order APIs |
| `sys_simulation` | Simulation broker, matching engine, and event source for backtesting |
| `sys_analyser` | Records trades/positions daily, computes risk metrics, outputs CSV/plots |
| `sys_transaction_cost` | Stock and futures fee/tax calculation |
| `sys_risk` | Pre-order risk validation |
| `sys_scheduler` | Scheduled callbacks (run at fixed intervals) |
| `sys_progress` | Console progress output |
| `trade_viewer` | Post-backtest visualization (Flask + TradingView Lightweight Charts); `--view` / `-V` flag |

### Portfolio / Accounts

`portfolio/` contains the `Portfolio` class aggregating multiple `Account` objects. Account types are registered by `sys_accounts`. `AbstractPosition` implementations (in `sys_accounts`) represent instrument-level holdings.

### Strategy API

`rqalpha/apis/` exposes trading functions to user strategies (e.g., `order_shares`, `order_value`, `get_price`). `user_module.py` defines the namespace injected into strategy files. `api_rqdatac.py` provides RQData-backed APIs when rqdatac is available.

### Configuration

Default config is in `rqalpha/config.yml`. Mod-specific defaults are in `rqalpha/mod_config.yml`. Config is a two-level namespace (`config.base.*`, `config.extra.*`, `config.mod.<mod_name>.*`) represented as `RqAttrDict` objects.

## Testing

Integration tests live under `tests/integration_tests/` and use the `run_and_assert_result` pytest fixture. This fixture:
1. Runs a full backtest using `rqalpha.run_func(**kwargs)` with a data bundle at `~/.rqalpha/bundle/`
2. Serializes the result and compares it against a golden file in `outs/<testcase_name>.txt`

To add a new integration test, create a test function that calls `run_and_assert_result(config=..., init=..., handle_bar=...)`, then generate the golden file by running the test with `--update-result` (or deleting the `.txt` file so it gets created fresh).

Unit tests in `tests/unittest/` test individual modules directly without needing a bundle.

**Integration tests require a data bundle at `~/.rqalpha/bundle/`.** If the bundle is absent, integration tests will fail.

## Futu Mod (港股 / HK Markets)

`rqalpha/mod/rqalpha_mod_futu/` adds HK-market support via the Futu OpenD API:

| File | Role |
| --- | --- |
| `data_source.py` | `FutuDataSource` — fetches 15m/1d bars; caches as numpy structured arrays |
| `event_source.py` | `FutuSimulationEventSource` (backtest, HK hours) + `FutuLiveEventSource` (realtime) |
| `broker.py` | `FutuBroker` — live order submission via `OpenSecTradeContext` |
| `mod.py` | `FutuMod` — wires data source, event source, broker based on `run_type` |

**HK sessions**: 09:30–12:00 and 13:00–16:00 (22 × 15m bars/day). The mod aliases the HK calendar under `CN_STOCK` so RQAlpha's internal calendar lookups work. **order_book_id format**: Futu-native `HK.XXXXX` (e.g. `HK.09988`).

**Enable**: add to your strategy config YAML:

```yaml
mod:
  futu:
    enabled: true
    host: "127.0.0.1"
    port: 11112
```

**Run a HK backtest** (OpenD must be running):

```bash
rqalpha run -f strategies/hk_new_stock_breakout.py \
    -s 2024-01-01 -e 2024-12-31 -fq 15m -a stock 1000000 \
    --config strategies/hk_breakout_config.yml --view
```

**Run live**: add `-rt r` and set `trade_env: REAL` in config.

## Strategies

User strategies live in `strategies/`. Add new `.py` strategy files and optional `.yml` config files here.

| File | Description |
| --- | --- |
| `hk_new_stock_breakout.py` | HK new stock 3-stage breakout strategy |
| `hk_breakout_config.yml` | Config for the HK breakout strategy |
| `new_stock_breakout_backtest.py` | A-share new stock breakout backtest |

**Visualization**: run any backtest with `--view` to auto-launch the interactive chart viewer at `http://127.0.0.1:8050` after completion. The viewer source is at `rqalpha/mod/rqalpha_mod_trade_viewer/viewer.py`.

## Key Entry Points

- **CLI**: `rqalpha/__main__.py` → `rqalpha/cmds/run.py`
- **Programmatic**: `rqalpha.run_func(config, init, handle_bar, ...)` (re-exported from `rqalpha/__init__.py`)
- **Core bootstrap**: `rqalpha/main.py:run()`
