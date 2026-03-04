"""
Soltrade-Plus
A modular wrapper around nmweaver/soltrade that adds persistent state,
dashboarding, backtesting, optimization, and risk tooling without
modifying the soltrade core.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_SOLTRADE_IMPORT_ERROR: Optional[Exception] = None
try:
    from soltrade.config import config as soltrade_config
    from soltrade.trading import perform_analysis, fetch_candlestick
    from soltrade.wallet import find_balance
    from soltrade.transactions import market
    from soltrade.log import log_general, log_transaction
    SOLTRADE_AVAILABLE = True
except Exception as exc:
    SOLTRADE_AVAILABLE = False
    _SOLTRADE_IMPORT_ERROR = exc
    soltrade_config = None
    perform_analysis = None
    fetch_candlestick = None
    find_balance = None
    market = None
    log_general = None
    log_transaction = None


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class PositionSnapshot:
    is_open: bool
    stop_loss: float
    take_profit: float


def read_position_file(path: str) -> PositionSnapshot:
    if not os.path.exists(path):
        return PositionSnapshot(False, 0.0, 0.0)
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return PositionSnapshot(
        bool(data.get("is_open", False)),
        float(data.get("sl", 0.0)),
        float(data.get("tp", 0.0)),
    )


def write_position_file(path: str, is_open: bool, stop_loss: float, take_profit: float) -> None:
    payload = {"is_open": bool(is_open), "sl": float(stop_loss), "tp": float(take_profit)}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def update_position_state(path: str, is_open: bool, stop_loss: float, take_profit: float) -> None:
    if SOLTRADE_AVAILABLE and market is not None and path == "position.json":
        market().update_position(is_open, stop_loss, take_profit)
        return
    write_position_file(path, is_open, stop_loss, take_profit)

class SQLiteStateStore:
    def __init__(self, path: str) -> None:
        _ensure_parent_dir(path)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                strategy_id TEXT PRIMARY KEY,
                symbol TEXT,
                is_open INTEGER,
                entry_price REAL,
                entry_time TEXT,
                size REAL,
                stop_loss REAL,
                take_profit REAL,
                last_price REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                updated_at TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                symbol TEXT,
                side TEXT,
                price REAL,
                size REAL,
                pnl REAL,
                pnl_pct REAL,
                txid TEXT,
                note TEXT,
                timestamp TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                equity REAL,
                timestamp TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self.conn.commit()

    def set_setting(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def upsert_position(self, payload: Dict[str, Any]) -> None:
        if "strategy_id" not in payload:
            raise ValueError("strategy_id is required")
        columns = ", ".join(payload.keys())
        placeholders = ", ".join(["?"] * len(payload))
        updates = ", ".join([f"{key}=excluded.{key}" for key in payload.keys() if key != "strategy_id"])
        sql = f"INSERT INTO positions ({columns}) VALUES ({placeholders}) ON CONFLICT(strategy_id) DO UPDATE SET {updates}"
        self.conn.execute(sql, list(payload.values()))
        self.conn.commit()

    def get_position(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM positions WHERE strategy_id = ?", (strategy_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def record_trade(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        price: Optional[float],
        size: Optional[float],
        pnl: Optional[float],
        pnl_pct: Optional[float],
        txid: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO trades (strategy_id, symbol, side, price, size, pnl, pnl_pct, txid, note, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                symbol,
                side,
                price,
                size,
                pnl,
                pnl_pct,
                txid,
                note,
                _utc_now(),
            ),
        )
        self.conn.commit()

    def record_equity(self, strategy_id: str, equity: float) -> None:
        self.conn.execute(
            "INSERT INTO equity (strategy_id, equity, timestamp) VALUES (?, ?, ?)",
            (strategy_id, equity, _utc_now()),
        )
        self.conn.commit()

    def list_trades(self, strategy_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM trades WHERE strategy_id = ? ORDER BY id DESC"
        params: List[Any] = [strategy_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def list_equity(self, strategy_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM equity WHERE strategy_id = ? ORDER BY id ASC"
        params: List[Any] = [strategy_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

class JsonStateStore:
    def __init__(self, path: str) -> None:
        _ensure_parent_dir(path)
        self.path = path
        if not os.path.exists(path):
            self._write(self._default_state())

    def _default_state(self) -> Dict[str, Any]:
        return {"settings": {}, "positions": {}, "trades": [], "equity": []}

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return self._default_state()
        with open(self.path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, data: Dict[str, Any]) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(data, file)
        os.replace(tmp_path, self.path)

    def set_setting(self, key: str, value: Any) -> None:
        data = self._read()
        data["settings"][key] = str(value)
        self._write(data)

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        data = self._read()
        return data["settings"].get(key, default)

    def upsert_position(self, payload: Dict[str, Any]) -> None:
        if "strategy_id" not in payload:
            raise ValueError("strategy_id is required")
        data = self._read()
        data["positions"][payload["strategy_id"]] = payload
        self._write(data)

    def get_position(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        data = self._read()
        return data["positions"].get(strategy_id)

    def record_trade(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        price: Optional[float],
        size: Optional[float],
        pnl: Optional[float],
        pnl_pct: Optional[float],
        txid: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        data = self._read()
        data["trades"].append(
            {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": side,
                "price": price,
                "size": size,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "txid": txid,
                "note": note,
                "timestamp": _utc_now(),
            }
        )
        self._write(data)

    def record_equity(self, strategy_id: str, equity: float) -> None:
        data = self._read()
        data["equity"].append({"strategy_id": strategy_id, "equity": equity, "timestamp": _utc_now()})
        self._write(data)

    def list_trades(self, strategy_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        data = self._read()
        trades = [t for t in data["trades"] if t["strategy_id"] == strategy_id]
        trades = list(reversed(trades))
        if limit:
            trades = trades[:limit]
        return trades

    def list_equity(self, strategy_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        data = self._read()
        equity = [e for e in data["equity"] if e["strategy_id"] == strategy_id]
        if limit:
            equity = equity[-limit:]
        return equity


class StateStore:
    def __init__(self, path: str) -> None:
        if path.lower().endswith(".json"):
            self.backend = JsonStateStore(path)
        else:
            self.backend = SQLiteStateStore(path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

class RiskManager:
    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        kelly_win_rate: float = 0.55,
        kelly_win_loss: float = 1.5,
        kelly_max_fraction: float = 0.25,
        kelly_min_fraction: float = 0.0,
    ) -> None:
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.kelly_win_rate = kelly_win_rate
        self.kelly_win_loss = kelly_win_loss
        self.kelly_max_fraction = kelly_max_fraction
        self.kelly_min_fraction = kelly_min_fraction

    def atr(self, df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        if not {"high", "low", "close"}.issubset(df.columns):
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]
        if pd.isna(atr):
            return None
        return float(atr)

    def dynamic_stop_loss(self, price: float, atr: Optional[float], current_stop: Optional[float]) -> Optional[float]:
        if atr is None or price is None:
            return None
        candidate = price - self.atr_mult * atr
        if current_stop is None:
            return candidate
        return max(current_stop, candidate)

    def kelly_fraction(self) -> float:
        win_rate = self.kelly_win_rate
        win_loss = max(self.kelly_win_loss, 1e-9)
        edge = win_rate - (1.0 - win_rate) / win_loss
        return min(self.kelly_max_fraction, max(self.kelly_min_fraction, edge))

    def position_size(self, equity: Optional[float], price: Optional[float]) -> Optional[float]:
        if equity is None or price is None or price <= 0:
            return None
        fraction = self.kelly_fraction()
        if fraction <= 0:
            return None
        return (equity * fraction) / price


class Metrics:
    @staticmethod
    def sharpe_ratio(equity: pd.Series, periods_per_year: int = 365) -> Optional[float]:
        if equity is None or len(equity) < 3:
            return None
        returns = equity.pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return None
        sharpe = (returns.mean() / returns.std()) * (periods_per_year**0.5)
        return float(sharpe)

    @staticmethod
    def max_drawdown(equity: pd.Series) -> Optional[float]:
        if equity is None or equity.empty:
            return None
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())


class NullNotifier:
    def send(self, message: str) -> None:
        return


class TelegramNotifier:
    def __init__(self, token: Optional[str], chat_id: Optional[str]) -> None:
        self.token = token
        self.chat_id = chat_id
        self._bot = None
        if token and chat_id:
            try:
                from telegram import Bot

                self._bot = Bot(token=token)
            except Exception:
                self._bot = None

    def send(self, message: str) -> None:
        if not self.token or not self.chat_id:
            return
        if self._bot is not None:
            try:
                self._bot.send_message(chat_id=self.chat_id, text=message)
                return
            except Exception:
                pass
        try:
            import requests

            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data={"chat_id": self.chat_id, "text": message},
                timeout=10,
            )
        except Exception:
            return

class BacktestEngine:
    @staticmethod
    def fetch_ohlcv_ccxt(
        symbol: str = "SOL/USDC",
        timeframe: str = "1m",
        limit: int = 1000,
        exchange_id: str = "binance",
    ) -> pd.DataFrame:
        import ccxt

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["datetime", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in df.columns:
            if "timestamp" in df.columns:
                df = df.copy()
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                raise ValueError("DataFrame must include datetime or timestamp column")
        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df[required].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.dropna()
        return df

    def run(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        cash: float = 10000.0,
        commission: float = 0.001,
    ) -> Dict[str, Any]:
        import backtrader as bt

        df = self.normalize_ohlcv(data)

        class SoltradeBacktestStrategy(bt.Strategy):
            params = dict(
                ema_short=5,
                ema_long=20,
                rsi_period=14,
                bb_period=14,
                rsi_low=31,
                rsi_high=68,
            )

            def __init__(self) -> None:
                self.ema_short = bt.ind.EMA(self.data.close, period=self.p.ema_short)
                self.ema_long = bt.ind.EMA(self.data.close, period=self.p.ema_long)
                self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
                self.bb = bt.ind.BollingerBands(self.data.close, period=self.p.bb_period)

            def next(self) -> None:
                price = self.data.close[0]
                if not self.position:
                    if (self.ema_short[0] > self.ema_long[0] or price < self.bb.lines.bot[0]) and self.rsi[0] <= self.p.rsi_low:
                        self.buy()
                else:
                    if (self.ema_short[0] < self.ema_long[0] or price > self.bb.lines.top[0]) and self.rsi[0] >= self.p.rsi_high:
                        self.sell()

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)
        datafeed = bt.feeds.PandasData(
            dataname=df,
            datetime="datetime",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )
        cerebro.adddata(datafeed)
        cerebro.addstrategy(SoltradeBacktestStrategy, **(params or {}))
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        drawdown = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown")
        trade_analysis = strat.analyzers.trades.get_analysis()

        return {
            "final_value": final_value,
            "pnl": final_value - cash,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "trades": trade_analysis,
        }

class ParameterOptimizer:
    def __init__(self, backtester: BacktestEngine) -> None:
        self.backtester = backtester

    def optimize(
        self,
        data: pd.DataFrame,
        n_trials: int = 30,
        cash: float = 10000.0,
    ) -> "optuna.Study":
        import optuna

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "ema_short": trial.suggest_int("ema_short", 3, 12),
                "ema_long": trial.suggest_int("ema_long", 15, 40),
                "rsi_period": trial.suggest_int("rsi_period", 7, 21),
                "bb_period": trial.suggest_int("bb_period", 10, 30),
            }
            metrics = self.backtester.run(data, params=params, cash=cash)
            score = metrics.get("sharpe")
            if score is None:
                return -1e9
            return float(score)

        # Ici, ajoutez votre ML custom pour scans AI.
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study


class SoltradeWrapper:
    def __init__(
        self,
        state_path: str = "state/soltrade_plus.db",
        strategy_id: str = "soltrade_plus",
        symbol: str = "SOL/USDC",
        starting_equity: float = 10000.0,
        default_trade_size: Optional[float] = None,
        enable_dynamic_stop: bool = True,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        kelly_win_rate: float = 0.55,
        kelly_win_loss: float = 1.5,
        kelly_max_fraction: float = 0.25,
        enable_position_sizing: bool = False,
        enable_telegram: bool = False,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        poll_seconds: Optional[int] = None,
        dry_run: bool = False,
        position_file_path: str = "position.json",
        use_onchain_balance: bool = False,
        env_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state = StateStore(state_path)
        self.state.set_setting("starting_equity", starting_equity)
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.position_file_path = position_file_path
        self.default_trade_size = default_trade_size
        self.enable_dynamic_stop = enable_dynamic_stop
        self.enable_position_sizing = enable_position_sizing
        self.dry_run = dry_run
        self.use_onchain_balance = use_onchain_balance
        self.env_overrides = env_overrides or {}
        self.risk = RiskManager(
            atr_period=atr_period,
            atr_mult=atr_mult,
            kelly_win_rate=kelly_win_rate,
            kelly_win_loss=kelly_win_loss,
            kelly_max_fraction=kelly_max_fraction,
        )
        if enable_telegram:
            self.notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        else:
            self.notifier = NullNotifier()
        self._config = None
        self._last_price: Optional[float] = None
        self._patched = False
        if SOLTRADE_AVAILABLE:
            self._config = self._load_soltrade_config()
            self.poll_seconds = poll_seconds or self._config.price_update_seconds
        else:
            self.poll_seconds = poll_seconds or 60

    def _load_soltrade_config(self) -> Any:
        if not SOLTRADE_AVAILABLE or soltrade_config is None:
            raise RuntimeError(f"Soltrade not available: {_SOLTRADE_IMPORT_ERROR}")
        for key, value in self.env_overrides.items():
            if value is not None:
                os.environ[str(key)] = str(value)
        return soltrade_config()

    def _require_soltrade(self) -> None:
        if not SOLTRADE_AVAILABLE:
            raise RuntimeError(f"Soltrade not available: {_SOLTRADE_IMPORT_ERROR}")

    def _log(self, message: str) -> None:
        if log_general is not None:
            log_general.info(message)
        else:
            print(message)

    def _fetch_candles_df(self) -> Optional[pd.DataFrame]:
        if fetch_candlestick is None:
            return None
        raw = fetch_candlestick()
        data = raw.get("Data", {}).get("Data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def _get_latest_price(self) -> Optional[float]:
        df = self._fetch_candles_df()
        if df is None or df.empty or "close" not in df.columns:
            return None
        price = float(df["close"].iat[-1])
        self._last_price = price
        return price

    def _apply_dynamic_stop(self, df: pd.DataFrame, snapshot: PositionSnapshot) -> None:
        if not snapshot.is_open:
            return
        atr = self.risk.atr(df)
        price = float(df["close"].iat[-1]) if "close" in df.columns else None
        new_stop = self.risk.dynamic_stop_loss(price, atr, snapshot.stop_loss)
        if new_stop is None:
            return
        if snapshot.stop_loss is None or new_stop > snapshot.stop_loss:
            update_position_state(self.position_file_path, True, new_stop, snapshot.take_profit)

    def _estimate_equity(self) -> Optional[float]:
        raw = self.state.get_setting("starting_equity")
        if raw is None:
            return None
        try:
            starting = float(raw)
        except ValueError:
            return None
        pos = self.state.get_position(self.strategy_id) or {}
        realized = float(pos.get("realized_pnl") or 0.0)
        unrealized = float(pos.get("unrealized_pnl") or 0.0)
        return starting + realized + unrealized

    def _estimate_position_size(self, side: str, price: Optional[float]) -> Optional[float]:
        if self.default_trade_size is not None:
            return float(self.default_trade_size)
        if self.use_onchain_balance and SOLTRADE_AVAILABLE and find_balance is not None and self._config is not None:
            try:
                if side == "BUY":
                    balance = find_balance(self._config.primary_mint)
                    return balance / price if price else None
                if side == "SELL":
                    return find_balance(self._config.secondary_mint)
            except Exception:
                return None
        return None

    def _compute_pnl(self, entry_price: Optional[float], exit_price: Optional[float], size: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if entry_price is None or exit_price is None:
            return None, None
        pnl_pct = (exit_price / entry_price) - 1.0
        pnl = (exit_price - entry_price) * size if size is not None else None
        return pnl, pnl_pct

    def _patch_perform_swap(self) -> None:
        if self._patched:
            return
        if not SOLTRADE_AVAILABLE:
            return
        import soltrade.transactions as transactions

        self._orig_perform_swap = transactions.perform_swap

        async def wrapped(sent_amount: float, sent_token_mint: str):
            adjusted_amount = sent_amount
            if self.enable_position_sizing and self._config is not None and sent_token_mint == self._config.primary_mint:
                equity = self._estimate_equity()
                price = self._last_price or self._get_latest_price()
                target_size = self.risk.position_size(equity, price)
                if target_size is not None and price is not None:
                    target_amount = target_size * price
                    adjusted_amount = min(sent_amount, target_amount)
            if self.dry_run:
                self._log("Dry run: skipping swap execution.")
                return True
            return await self._orig_perform_swap(adjusted_amount, sent_token_mint)

        transactions.perform_swap = wrapped
        self._patched = True

    def _record_trade(self, side: str, price: Optional[float], size: Optional[float], pnl: Optional[float], pnl_pct: Optional[float]) -> None:
        self.state.record_trade(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            side=side,
            price=price,
            size=size,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        if isinstance(self.notifier, NullNotifier):
            return
        if price is None:
            message = f"[Soltrade-Plus] {side} {self.symbol}"
        else:
            message = f"[Soltrade-Plus] {side} {self.symbol} @ {price:.4f}"
        self.notifier.send(message)

    def _sync_state(self, pre: PositionSnapshot, post: PositionSnapshot, price: Optional[float]) -> None:
        now = _utc_now()
        existing = self.state.get_position(self.strategy_id) or {}
        entry_price = existing.get("entry_price")
        entry_time = existing.get("entry_time")
        size = existing.get("size")
        realized_pnl = float(existing.get("realized_pnl") or 0.0)

        if not pre.is_open and post.is_open:
            entry_price = price or entry_price
            entry_time = now
            size = self._estimate_position_size("BUY", price)
            self._record_trade("BUY", price, size, None, None)

        elif pre.is_open and not post.is_open:
            pnl, pnl_pct = self._compute_pnl(entry_price, price, size)
            if pnl is not None:
                realized_pnl += pnl
            self._record_trade("SELL", price, size, pnl, pnl_pct)
            entry_price = None
            entry_time = None
            size = None

        unrealized_pnl = None
        if post.is_open and entry_price is not None and price is not None and size is not None:
            unrealized_pnl = (price - entry_price) * size

        payload = {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "is_open": int(post.is_open),
            "entry_price": entry_price,
            "entry_time": entry_time,
            "size": size,
            "stop_loss": post.stop_loss,
            "take_profit": post.take_profit,
            "last_price": price,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "updated_at": now,
        }
        self.state.upsert_position(payload)

        equity = self._estimate_equity()
        if equity is not None:
            self.state.record_equity(self.strategy_id, equity)

    def run_once(self) -> None:
        self._require_soltrade()
        if self._config is None:
            self._config = self._load_soltrade_config()
        pre = read_position_file(self.position_file_path)
        df = self._fetch_candles_df()
        price = None
        if df is not None and "close" in df.columns:
            price = float(df["close"].iat[-1])
            self._last_price = price
        if self.enable_dynamic_stop and df is not None:
            self._apply_dynamic_stop(df, pre)
            pre = read_position_file(self.position_file_path)
        if self.dry_run or self.enable_position_sizing:
            self._patch_perform_swap()
        perform_analysis()
        post = read_position_file(self.position_file_path)
        if price is None:
            price = self._get_latest_price()
        self._sync_state(pre, post, price)

    def run_forever(self) -> None:
        self._require_soltrade()
        self._log("Soltrade-Plus started.")
        while True:
            try:
                self.run_once()
            except Exception as exc:
                self._log(f"Soltrade-Plus loop error: {exc}")
            time.sleep(self.poll_seconds)


def run_dashboard(state_path: str, strategy_id: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="Soltrade-Plus", layout="wide")
    st.title("Soltrade-Plus Dashboard")

    store = StateStore(state_path)
    position = store.get_position(strategy_id) or {}

    trades = pd.DataFrame(store.list_trades(strategy_id))
    equity = pd.DataFrame(store.list_equity(strategy_id))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Position", "OPEN" if position.get("is_open") else "CLOSED")
    col2.metric("Last Price", f"{position.get('last_price') or 0:.4f}")
    col3.metric("Unrealized PnL", f"{position.get('unrealized_pnl') or 0:.2f}")
    col4.metric("Realized PnL", f"{position.get('realized_pnl') or 0:.2f}")

    st.subheader("Equity Curve")
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"])
        equity = equity.sort_values("timestamp")
        st.line_chart(equity.set_index("timestamp")["equity"])
        sharpe = Metrics.sharpe_ratio(equity["equity"])
        drawdown = Metrics.max_drawdown(equity["equity"])
        st.write(f"Sharpe: {sharpe:.3f}" if sharpe is not None else "Sharpe: n/a")
        st.write(f"Max Drawdown: {drawdown:.2%}" if drawdown is not None else "Max Drawdown: n/a")
    else:
        st.info("No equity data yet. Run the wrapper to populate metrics.")

    st.subheader("Trades")
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        st.dataframe(trades, use_container_width=True)
    else:
        st.info("No trades recorded yet.")


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Soltrade-Plus wrapper")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run Soltrade-Plus live wrapper")
    run_p.add_argument("--state", default=os.getenv("STATE_DB_PATH", "state/soltrade_plus.db"))
    run_p.add_argument("--strategy-id", default="soltrade_plus")
    run_p.add_argument("--symbol", default="SOL/USDC")
    run_p.add_argument("--starting-equity", type=float, default=_env_float("STARTING_EQUITY", 10000.0))
    run_p.add_argument("--default-trade-size", type=float, default=_env_float("DEFAULT_TRADE_SIZE", 0.0))
    run_p.add_argument("--enable-dynamic-stop", action="store_true")
    run_p.add_argument("--disable-dynamic-stop", action="store_true")
    run_p.add_argument("--atr-period", type=int, default=_env_int("ATR_PERIOD", 14))
    run_p.add_argument("--atr-mult", type=float, default=_env_float("ATR_MULT", 2.0))
    run_p.add_argument("--kelly-win-rate", type=float, default=_env_float("KELLY_WIN_RATE", 0.55))
    run_p.add_argument("--kelly-win-loss", type=float, default=_env_float("KELLY_WIN_LOSS", 1.5))
    run_p.add_argument("--kelly-max-fraction", type=float, default=_env_float("KELLY_MAX_FRACTION", 0.25))
    run_p.add_argument("--enable-position-sizing", action="store_true")
    run_p.add_argument("--enable-telegram", action="store_true", default=_env_bool("ENABLE_TELEGRAM", False))
    run_p.add_argument("--telegram-token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    run_p.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID"))
    run_p.add_argument("--poll-seconds", type=int, default=None)
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--use-onchain-balance", action="store_true")

    dash_p = sub.add_parser("dashboard", help="Run Streamlit dashboard")
    dash_p.add_argument("--state", default=os.getenv("STATE_DB_PATH", "state/soltrade_plus.db"))
    dash_p.add_argument("--strategy-id", default="soltrade_plus")

    back_p = sub.add_parser("backtest", help="Run backtest")
    back_p.add_argument("--symbol", default="SOL/USDC")
    back_p.add_argument("--timeframe", default="1m")
    back_p.add_argument("--limit", type=int, default=1000)
    back_p.add_argument("--exchange", default="binance")
    back_p.add_argument("--csv", default=None)
    back_p.add_argument("--cash", type=float, default=10000.0)
    back_p.add_argument("--ema-short", type=int, default=5)
    back_p.add_argument("--ema-long", type=int, default=20)
    back_p.add_argument("--rsi-period", type=int, default=14)
    back_p.add_argument("--bb-period", type=int, default=14)

    opt_p = sub.add_parser("optimize", help="Run Optuna optimization")
    opt_p.add_argument("--symbol", default="SOL/USDC")
    opt_p.add_argument("--timeframe", default="5m")
    opt_p.add_argument("--limit", type=int, default=1000)
    opt_p.add_argument("--exchange", default="binance")
    opt_p.add_argument("--csv", default=None)
    opt_p.add_argument("--cash", type=float, default=10000.0)
    opt_p.add_argument("--trials", type=int, default=30)

    args = parser.parse_args()

    if args.command == "run":
        enable_dynamic_stop = True
        if args.disable_dynamic_stop:
            enable_dynamic_stop = False
        if args.enable_dynamic_stop:
            enable_dynamic_stop = True
        default_trade_size = args.default_trade_size if args.default_trade_size > 0 else None
        wrapper = SoltradeWrapper(
            state_path=args.state,
            strategy_id=args.strategy_id,
            symbol=args.symbol,
            starting_equity=args.starting_equity,
            default_trade_size=default_trade_size,
            enable_dynamic_stop=enable_dynamic_stop,
            atr_period=args.atr_period,
            atr_mult=args.atr_mult,
            kelly_win_rate=args.kelly_win_rate,
            kelly_win_loss=args.kelly_win_loss,
            kelly_max_fraction=args.kelly_max_fraction,
            enable_position_sizing=args.enable_position_sizing,
            enable_telegram=args.enable_telegram,
            telegram_token=args.telegram_token,
            telegram_chat_id=args.telegram_chat_id,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
            use_onchain_balance=args.use_onchain_balance,
        )
        wrapper.run_forever()
        return

    if args.command == "dashboard":
        run_dashboard(args.state, args.strategy_id)
        return

    if args.command == "backtest":
        engine = BacktestEngine()
        if args.csv:
            data = _load_csv(args.csv)
        else:
            data = engine.fetch_ohlcv_ccxt(args.symbol, args.timeframe, args.limit, args.exchange)
        params = {
            "ema_short": args.ema_short,
            "ema_long": args.ema_long,
            "rsi_period": args.rsi_period,
            "bb_period": args.bb_period,
        }
        metrics = engine.run(data, params=params, cash=args.cash)
        print(json.dumps(metrics, indent=2, default=str))
        return

    if args.command == "optimize":
        engine = BacktestEngine()
        if args.csv:
            data = _load_csv(args.csv)
        else:
            data = engine.fetch_ohlcv_ccxt(args.symbol, args.timeframe, args.limit, args.exchange)
        optimizer = ParameterOptimizer(engine)
        study = optimizer.optimize(data, n_trials=args.trials, cash=args.cash)
        print("Best params:")
        print(study.best_params)
        print("Best score:")
        print(study.best_value)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
