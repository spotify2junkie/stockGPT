import os
import warnings
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf
from openai import OpenAI
from ta.momentum import RSIIndicator

warnings.filterwarnings("ignore")


PROMPT_TMPL = """
    Help me to aggregate the stock symobos by their dates' occurance and by their metrics,please return aggregated result only.

    For example for context
    [['MSFT', '2024-02-07'],
    ['MSFT', '2024-02-09'],
    ['AAPL', '2024-02-09']]

    You should return
    `
    Date Symbols
    2024-02-07 MSFT
    2024-02-09 AAPL,MSFT
    `


    here is the  context for RSI metrics :{}.

    here is the  context for MACD metrics: {}

    here is the  context for MA5 metrics for whether the stock drop below ma5: {}

    here is the  context for MA10 metrics for whether the stock drop below ma10: {}

    here is the  context for BOLL metrics for whether the stock across boll middle line: {}

    here is the  context for Volume metrics for whether the volume increased 10%: {}

    help me to get the result for different metrics,show me the result in markdown table format, with row being date, column being metrics,specifically for MA5 please use drop_below_MA5  and MA10 please use drop_below_MA10.Please make sure dont show more than 10 stocks in each cell in markdown.


"""


INVEST_PROMPT = """
    You are  very good at investing, here is some metrics with crossing conditions
    For example
    RSI:
    Date        Symbols
    -----------------
    2024-02-15  JPM,XOM,NFLX,INTU,INTC,WFC
    2024-02-14  BABA
    ====================

    means in 2024-02-15 JPM,XOM is recommend for RSI metric, this is the full context :

    {}
    Now synthesize different metrics and give me the final conclusion, give me your investment ideas for me to invest on the last day. For me to strong buy or strong sell.
"""


class StockData(ABC):
    def __init__(self, symbol, load_df=None):
        self.symbol = symbol
        if load_df is not None:
            self.data = load_df
        else:
            self.data = self._verify_symbol(symbol)

    def _verify_symbol(self, s):
        data = yf.download(s, period="1y", timeout=3)
        if data.shape[0] <= 1:
            raise ValueError(f"Input symbol {s} is not found")
        else:
            return data


@dataclass
class StockDataManager:
    data_dict: Dict[str, pd.DataFrame] = field(default_factory=dict)
    scan_list: List[str] = field(default_factory=list)

    def _fetch_list(self):
        import subprocess

        subprocess.run(["wget", os.environ["CAP_URL"], "-O", "cap.csv"])
        df = pd.read_csv("cap.csv")
        self.scan_list = df.head(100)["Symbol"].tolist()

    def fetch_df(self):
        self._fetch_list()
        for symbol in self.scan_list:
            self.data_dict[symbol] = yf.download(symbol, period="1y", timeout=3)
        import pickle

        import redis

        r = redis.Redis(host="localhost", port=6379, db=0)

        serialized_data = pickle.dumps(self.data_dict)
        r.set("ta_stats", serialized_data)
        r.close()


class RSICrossMixin:
    """
    define class
    """

    def get_stats_features(self):
        rsi_df = RSIIndicator(close=self.data["Close"], window=6)
        self.data["RSI_6"] = rsi_df.rsi()

        rsi_df = RSIIndicator(close=self.data["Close"], window=12)
        self.data["RSI_12"] = rsi_df.rsi()

        golden_cross = (self.data["RSI_6"] > self.data["RSI_12"]) & (
            self.data["RSI_6"].shift(1) <= self.data["RSI_12"].shift(1)
        )
        max_cross_date = self.data[golden_cross].reset_index()["Date"].max()
        if max_cross_date >= (datetime.now() - timedelta(7)):
            return True, max_cross_date.date().strftime("%Y-%m-%d")
        else:
            return False, max_cross_date.date().strftime("%Y-%m-%d")


class MACDCrossMixin:
    """
    define class
    """

    def get_stats_features(self):
        from ta.trend import MACD

        macd_indicator = MACD(close=self.data["Close"])

        # Add MACD signals to the DataFrame
        self.data["macd"] = macd_indicator.macd()
        self.data["signal_line"] = macd_indicator.macd_signal()
        golden_cross = (self.data["macd"] > self.data["signal_line"]) & (
            self.data["macd"].shift() <= self.data["signal_line"].shift()
        )
        max_cross_date = self.data[golden_cross].reset_index()["Date"].max()
        if max_cross_date >= (datetime.now() - timedelta(7)):
            return True, max_cross_date.date().strftime("%Y-%m-%d")
        else:
            return False, max_cross_date.date().strftime("%Y-%m-%d")


class MAMixin:
    def get_stats_features(self, ma_windows):
        from ta.trend import SMAIndicator

        # Calculate the Simple Moving Average (SMA)
        sma_indicator = SMAIndicator(
            close=self.data["Close"], window=ma_windows
        )  # 5-day SMA
        self.data["SMA"] = sma_indicator.sma_indicator()

        # Get the latest closing price and MA5 value
        latest_close = self.data["Close"].iloc[-1]
        latest_ma5 = self.data["SMA"].iloc[-1]

        if latest_close < latest_ma5:
            return True, self.data.reset_index()["Date"].max().strftime("%Y-%m-%d")
        else:
            return False, self.data.reset_index()["Date"].max().strftime("%Y-%m-%d")


class VolumeMixin:
    def get_stats_features(self):
        # 计算最近一周和之前的交易量的平均值
        recent_week_volume_mean = self.data["Volume"].tail(5).mean()
        previous_20days_volume_mean = self.data["Volume"].iloc[-25:-5].mean()

        # 判断最近一周交易量相对之前是否有20%的上涨
        volume_increase = (
            recent_week_volume_mean - previous_20days_volume_mean
        ) / previous_20days_volume_mean

        if volume_increase >= 0.1:
            return True, self.data.reset_index()["Date"].max().strftime("%Y-%m-%d")
        else:
            return False, self.data.reset_index()["Date"].max().strftime("%Y-%m-%d")


class BOLLMidBreakMixin:
    def get_stats_features(self):
        # 计算 Bollinger 布林线
        import ta

        bollinger_bands = ta.volatility.BollingerBands(
            close=self.data["Close"], window=20, window_dev=2
        )

        # 获取中轨线
        middle_band = bollinger_bands.bollinger_mavg()

        # 判断股价是否在中轨线附近并突破
        crossed_dates = []
        for i in range(1, len(self.data)):
            if (
                self.data["Close"][i] > middle_band[i]
                and self.data["Close"][i - 1] <= middle_band[i - 1]
            ):
                crossed_dates.append(self.data.index[i])

        max_cross_date = max(crossed_dates)
        if max_cross_date >= (datetime.now() - timedelta(7)):
            return True, max_cross_date.date().strftime("%Y-%m-%d")
        else:
            return False, max_cross_date.date().strftime("%Y-%m-%d")


class BOLLMidBouncekMixin:
    def get_stats_features(self):
        # 计算 Bollinger 布林线
        import ta

        bollinger_bands = ta.volatility.BollingerBands(
            close=self.data["Close"], window=20, window_dev=2
        )

        # 获取中轨线
        middle_band = bollinger_bands.bollinger_mavg()

        # 判断股价是否在中轨线附近并跌落
        crossed_dates = []
        for i in range(1, len(self.data)):
            if (
                self.data["Close"][i] < middle_band[i]
                and self.data["Close"][i - 1] >= middle_band[i - 1]
            ):
                crossed_dates.append(self.data.index[i])

        max_cross_date = max(crossed_dates)
        if max_cross_date >= (datetime.now() - timedelta(7)):
            return True, max_cross_date.date().strftime("%Y-%m-%d")
        else:
            return False, max_cross_date.date().strftime("%Y-%m-%d")


class RSIStock(StockData, MACDCrossMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features()
        if exist_quote:
            return [self.symbol, ds]


class MACDStock(StockData, RSICrossMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features()
        if exist_quote:
            return [self.symbol, ds]


class VolumeStock(StockData, VolumeMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features()
        if exist_quote:
            return [self.symbol, ds]


class MA5Stock(StockData, MAMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features(ma_windows=5)
        if exist_quote:
            return [self.symbol, ds]


class MA10Stock(StockData, MAMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features(ma_windows=10)
        if exist_quote:
            return [self.symbol, ds]


class BOLLStock(StockData, BOLLMidBreakMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features()
        if exist_quote:
            return [self.symbol, ds]


class BOLLBounceStock(StockData, BOLLMidBouncekMixin):
    def __init__(self, symbol, load_df=None):
        super().__init__(symbol, load_df)

    def get_quote(self):
        exist_quote, ds = self.get_stats_features()
        if exist_quote:
            return [self.symbol, ds]


class Scanner:
    def __init__(self, load_redis_key=None, mixin_class=None) -> None:
        self.scan_list = []
        if mixin_class is None:
            raise NotImplementedError("should specify mixin class for 技术指标")
        self.mixin_class = mixin_class
        self.stock_dict = None
        self.load_symbols()

        import pickle

        import redis

        r = redis.Redis(host="localhost", port=6379, db=0)

        if load_redis_key:
            self.stock_dict = pickle.loads(r.get(load_redis_key))

    def load_symbols(self):
        df = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cap.csv")
        )
        self.scan_list = df.head(100)["Symbol"].tolist()

    def scan_all(self):
        res = []
        for symbol in self.scan_list:
            res.append(
                self.mixin_class(
                    symbol=symbol, load_df=self.stock_dict.get(symbol)
                ).get_quote()
            )
        return res


def chat_openai(msg):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("BASE_URL")
    )

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": msg}],
        model="gpt-3.5-turbo",
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content


def metric_agg():
    # TODO: 应该是让gpt来做in-context, 补齐这个代码
    metric_stock_calc_list = [
        RSIStock,
        MACDStock,
        MA5Stock,
        MA10Stock,
        BOLLStock,
        BOLLBounceStock,
        VolumeStock,
    ]

    res_text = [
        Scanner("ta_stats", mixin_class=metriccls).scan_all()
        for metriccls in metric_stock_calc_list
    ]
    prompt_metrics = PROMPT_TMPL.format(*res_text)
    return chat_openai(prompt_metrics)


def recommend_stock(msg):
    p = INVEST_PROMPT.format(msg)
    resp = chat_openai(p)
    return resp


def get_latest_stock_data():
    StockDataManager().fetch_df()
