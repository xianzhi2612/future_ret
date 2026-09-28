from dis import Instruction
from gc import collect
from pathlib import Path
import select
from turtle import end_fill
from typing import overload
from dynaconf import base
import polars as pl
import blazestore as bs
import datacenter as dc
from xcals import api
from datetime import date
import numpy as np  
from scipy import stats
import matplotlib.pyplot as plt
import math
import polars_ds as pds
from alphamaster.rack import Rack
from alphamaster.polens import FactorAnalyzer
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False





class DailyPriceVolume_factor:

    #设置BlazeStore路径
    def __init__(self, blazestore_path: str):
        self.blazestore_path = Path(blazestore_path)

    #查找文件路径
    def get_factor_path(self, factor_name: str, date: str) -> Path:
        return (
            self.blazestore_path
            / "factors"
            / factor_name
            / f"date={date}"
            / "data.parquet"
        )
    
        #简化回测函数
    def backtest(
        self,
        factor_df: pl.LazyFrame,
        start_date: str,
        end_date: str,
        factor_name: str,
        quantiles: int,
        period: int,
        industry_name: str = "false",
    ) -> None:
        '''回测函数，输入因子数据和回测参数，输出回测结果
        输入参数：
        factor_df: pl.LazyFrame，包含因子数据的LazyFrame，必须包含date、asset和factor列
        start_date: str，回测开始日期，格式为'YYYY-MM-DD'
        end_date: str，回测结束日期，格式为'YYYY-MM-DD'
        factor_name: str，因子列的名称
        quantiles: int，分位数的数量，例如5表示分成5组
        period: int，回测的持有期，例如1表示持有1天，5表示持有5天
        industry_name: str，行业列名，支持"Lv1"、"Lv2"、"Lv3"，默认"false"表示不画行业分组图
        '''
        factor_df = factor_df.collect()  # 收集因子数据到内存中
        rack = Rack()
        rack.load_prices(start_date, end_date)  # 加载行情
        rack.set_factor(factor_df=factor_df, value_col=factor_name)  # 设置因子，指定因子列
        data = rack.get_data()
        if industry_name != "false":
            analyzer = FactorAnalyzer(data, group_col=industry_name)
        else:
            analyzer = FactorAnalyzer(data)
        # analyzer = (
        #     FactorAnalyzer(data, group_col=industry_name)
        #     if industry_name != "false" and industry_name in data.columns
        #     else FactorAnalyzer(data)
        # )
        analyzer.preprocess(periods=[1, 5, 10], quantiles=quantiles, demean=True)
        analyzer.analyze()
        stats = analyzer.summary_stats()
        print(stats)
        analyzer.plot("summary",period=period)
        analyzer.plot("turnover", period=period)
        if industry_name != "false" :
            print("yes")
            analyzer.plot("group_ic", group_col=industry_name)
        

        #改变股本数据格式（增加date列） 只保留了流通股本列和股票列和日期列
    def load_effective_capital_on(self, trade_date: str) -> pl.LazyFrame:
        return (
            dc.jy.capital(trade_date)
            .select(["asset", "AFloats"])
            .with_columns(
                pl.lit(trade_date).str.to_date().alias("date")
            )
        )



    #合成一个时间段内因子成一个lazyframe
    def collect_lazyframe(self, start_date: str, end_date: str, factor_func, industry: str="false") -> pl.LazyFrame:
        """
        计算 start_date 到 end_date 之间所有交易日的因子值，因子值全部后移一天避免未来数据
        并拼接成一个大的 LazyFrame 返回。

        参数
        ----
        start_date : str
            开始日期，格式如 "2023-01-03"
        end_date : str
            截止日期，格式如 "2023-02-03"
        factor_func : 需要拼接的因子函数名称
            单日因子函数，输入一个日期字符串，返回该日因子数据
            返回结果应至少包含三列：
            - date
            - asset
            - factor
        industry : str,
            是否需要拼接行业，默认为"false"，如果需要拼接行业数据，请传入对应的行业类型，例如"Lv1,Lv2,Lv3"

        返回
        ----
        pl.LazyFrame
        """
        start_date = api.get_last_tradingday(start_date)
        end_date = api.get_last_tradingday(end_date)
        extra_start_date = api.shift_tradeday(start_date, -1)
        trade_dates = api.get_tradingdays(extra_start_date, end_date)
        industry_dates = api.get_tradingdays(start_date, end_date)

        dfs = []
        for d in trade_dates:
            df = factor_func(d).collect()
            factor_columns = [column for column in df.columns if column not in ("date", "asset")]
            if factor_columns:
                df = df.with_columns(
                    pl.col(factor_columns).cast(pl.Float64)
                )
            dfs.append(df)
            print(f"{d} done")

        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)

        lf = (
            pl.concat(dfs)
            .sort(["asset", "date"])
            .with_columns(
                pl.exclude(["date", "asset"]).shift(1).over("asset")
            )
            .filter(
                (pl.col("date") >= start_dt) &
                (pl.col("date") <= end_dt)
            )
            .lazy()
        )
        if industry == "false":
            return lf
        else:
            industry_dfs = []
            for d in industry_dates:
                industry_df = (
                    dc.jy.industry(date=d)
                    .select(["asset", industry])
                    .with_columns(
                        pl.lit(d).str.to_date().alias("date")
                    )
                )
                industry_dfs.append(industry_df)

            industry_lf = pl.concat(industry_dfs).lazy()
            return lf.join(industry_lf, on=["asset", "date"], how="left")



    def calc_ret_nd(self, date: str, n: int = 1) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_ret_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        end_date = date
        start_date = api.shift_tradeday(end_date, -n)
        day_df = dc.md.read_data_batch(
            beg_date = start_date,
            end_date = end_date,
            instrument = dc.Instrument.STOCK,
            datatype = dc.DataType.KLINE_DAY
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "close"])
            .with_columns(
                pl.col("close").cast(pl.Float64).shift(n).over("asset").alias("close_n")
            )
            .with_columns(
                pl.when(
                    pl.col("close_n").is_not_null()
                    & (pl.col("close_n") != 0)
                )
                .then(
                    pl.col("close").cast(pl.Float64)
                    / pl.col("close_n").cast(pl.Float64)
                    - 1.0
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()

    def calc_exret_nd(
        self,
        date: str,
        n: int = 1,
        index_asset: str = "000001",
    ) -> pl.LazyFrame:
        '''
        计算过去 n 个交易日的个股超额收益率。

        超额收益率 = 个股过去 n 日收益率 - 指数过去 n 日收益率。

        参数
        ----
        date : str
            因子计算日期，格式为 "YYYY-MM-DD"。
        n : int
            回看交易日数量。
        index_asset : str
            指数代码，默认使用上证综合指数的 "000001"。

        返回
        ----
        pl.LazyFrame
            包含 date、asset 和 exret_{index_asset}_{n}d 三列的 LazyFrame。
        '''
        factor_name = f"DailyPriceVolume_exret_{index_asset}_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            cached_lf = pl.scan_parquet(factor_path)
            has_duplicate_keys = (
                cached_lf
                .select(
                    pl.struct(["date", "asset"])
                    .is_duplicated()
                    .any()
                    .alias("has_duplicates")
                )
                .collect()
                .item()
            )
            if not has_duplicate_keys:
                return cached_lf

        end_date = date
        start_date = api.shift_tradeday(end_date, -n)

        stock_df = (
            dc.md.read_data_batch(
                beg_date=start_date,
                end_date=end_date,
                instrument=dc.Instrument.STOCK,
                datatype=dc.DataType.KLINE_DAY,
            )
            .sort(["asset", "date"])
            .select(["date", "asset", "close"])
            .with_columns(
                pl.col("close").cast(pl.Float64).alias("stock_close")
            )
            .with_columns(
                pl.col("stock_close")
                .shift(n)
                .over("asset")
                .alias("stock_close_n")
            )
            .with_columns(
                pl.when(
                    pl.col("stock_close_n").is_not_null()
                    & (pl.col("stock_close_n") != 0)
                )
                .then(
                    pl.col("stock_close")
                    / pl.col("stock_close_n")
                    - 1.0
                )
                .otherwise(0.0)
                .alias("stock_ret")
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
        )

        index_df = (
            dc.md.read_data_batch(
                beg_date=start_date,
                end_date=end_date,
                instrument=dc.Instrument.INDEX,
                datatype=dc.DataType.KLINE_DAY,
            )
            .filter(pl.col("asset") == index_asset)
            .select(["date", "close"])
            .group_by("date")
            .agg(
                pl.col("close")
                .last()
                .cast(pl.Float64)
                .alias("index_close")
            )
            .sort("date")
            .with_columns(
                pl.col("index_close")
                .shift(n)
                .alias("index_close_n")
            )
            .with_columns(
                pl.when(
                    pl.col("index_close_n").is_not_null()
                    & (pl.col("index_close_n") != 0)
                )
                .then(
                    pl.col("index_close")
                    / pl.col("index_close_n")
                    - 1.0
                )
                .otherwise(0.0)
                .alias("index_ret")
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .select(["date", "index_ret"])
        )

        lf = (
            stock_df
            .join(
                index_df,
                on="date",
                how="left",
                validate="m:1",
            )
            .with_columns(
                (
                    pl.col("stock_ret")
                    - pl.col("index_ret")
                ).alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()



    def calc_vol_nd(self, date: str, n: int=5) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_vol_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        end_date = date
        start_date = api.shift_tradeday(end_date, -n)
        day_df = dc.md.read_data_batch(
            beg_date = start_date,
            end_date = end_date,
            instrument = dc.Instrument.STOCK,
            datatype = dc.DataType.KLINE_DAY
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "close", "prev_close"])
            .with_columns(
                pl.when(
                    pl.col("prev_close").is_not_null()
                    & (pl.col("prev_close") != 0)
                )
                .then(
                    (
                        pl.col("close").cast(pl.Float64)
                        / pl.col("prev_close").cast(pl.Float64)
                        - 1
                    )
                )
                .otherwise(0.0)
                .alias("day_ret")
            )
            .with_columns(
                pl.col('day_ret').rolling_std(window_size=n, min_samples=n).over('asset').alias(factor_name)
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()

    def calc_volume_ratio_nd(self, date: str, n: int= 5) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_volume_ratio_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        end_date = date
        start_date = api.shift_tradeday(end_date, -n)
        day_df = dc.md.read_data_batch(
            beg_date = start_date,
            end_date = end_date,
            instrument = dc.Instrument.STOCK,
            datatype = dc.DataType.KLINE_DAY
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "volume"])
            .with_columns(
                pl.col("volume").rolling_mean(window_size=n,min_samples= n).over("asset").alias('MA_volume')
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .with_columns(
                pl.when(
                    pl.col("MA_volume").is_not_null()
                    & (pl.col("MA_volume") != 0)
                )
                .then(
                    pl.col("volume").cast(pl.Float64)
                    / pl.col("MA_volume").cast(pl.Float64)
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()


    
    def calc_ocret_nd(self, date : str, n : int =1) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_ocret_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        n = n-1
        date = api.shift_tradeday(date, -n)
        day_df = dc.md.read_data_batch(
            date,
            date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_DAY
        )
        lf = (
            day_df
            .sort(['asset','date'])
            .select(['asset','date','close','open'])
            .with_columns(
                pl.when(
                    pl.col('open').is_not_null()
                    & (pl.col('open') != 0)
                )
                .then(
                    pl.col('close').cast(pl.Float64)
                    / pl.col('open').cast(pl.Float64)
                    - 1.0
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(['asset','date',factor_name])
            .sort(['asset','date'])
        )
        factor_path.parent.mkdir(parents=True,exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()



    def calc_gap_ret_nd(self, date: str, n : int =1) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_gap_ret_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        day_df = dc.md.read_data_batch(
            beg_date = date,
            end_date = date,
            instrument = dc.Instrument.STOCK,
            datatype = dc.DataType.KLINE_DAY
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "open", "prev_close"])
            .with_columns(
                pl.when(
                    pl.col("prev_close").is_not_null()
                    & (pl.col("prev_close") != 0)
                )
                .then(
                    pl.col("open").cast(pl.Float64)
                    / pl.col("prev_close").cast(pl.Float64)
                    - 1.0
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()




    def calc_hl_range_nd(self, date: str, n: int =1) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_hl_range_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        n =n -1 
        date = api.shift_tradeday(date, -n)
        day_df = dc.md.read_data_batch(
            beg_date=date,
            end_date=date,
            instrument=dc.Instrument.STOCK,
            datatype=dc.DataType.KLINE_DAY,
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "high", "low"])
            .with_columns(
                pl.when(
                    pl.col("low").is_not_null()
                    & (pl.col("low") != 0)
                )
                .then(
                    (
                        pl.col("high").cast(pl.Float64)
                        - pl.col("low").cast(pl.Float64)
                    )
                    / pl.col("low").cast(pl.Float64)
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()



    def calc_close_position_nd(self, date: str, n : int =1) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_close_position_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        n = n-1
        day_df = dc.md.read_data_batch(
            beg_date = date,
            end_date = date,
            instrument = dc.Instrument.STOCK,
            datatype = dc.DataType.KLINE_DAY
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "close", "high", "low"])
            .with_columns(
                pl.when(
                    pl.col("close").is_not_null()
                    & pl.col("high").is_not_null()
                    & pl.col("low").is_not_null()
                    & (
                        pl.col("high").cast(pl.Float64)
                        > pl.col("low").cast(pl.Float64)
                    )
                )
                .then(
                    (
                        pl.col("close").cast(pl.Float64)
                        - pl.col("low").cast(pl.Float64)
                    )
                    /
                    (
                        pl.col("high").cast(pl.Float64)
                        - pl.col("low").cast(pl.Float64)
                    )
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()



    def calc_bias_nd(self, date: str, n: int = 5) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_bias_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        end_date = date
        start_date = api.shift_tradeday(end_date, -n)
        day_df = dc.md.read_data_batch(
            beg_date=start_date,
            end_date=end_date,
            instrument=dc.Instrument.STOCK,
            datatype=dc.DataType.KLINE_DAY,
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "close"])
            .with_columns(
                pl.col("close").rolling_mean(window_size=n, min_periods=n).over("asset").alias("MA_close")
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .with_columns(
                pl.when(
                    pl.col("MA_close").is_not_null()
                    & (pl.col("MA_close") != 0)
                )
                .then(
                    pl.col("close").cast(pl.Float64)
                    / pl.col("MA_close").cast(pl.Float64)
                    - 1.0
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()




    def calc_ma_trend_n_m(self, date: str, n: int = 5, m: int = 20) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_ma_trend_{n}_{m}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        end_date = date
        start_date = api.shift_tradeday(end_date, -n)
        day_df = dc.md.read_data_batch(
            beg_date=start_date,
            end_date=end_date,
            instrument=dc.Instrument.STOCK,
            datatype=dc.DataType.KLINE_DAY,
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .select(["date", "asset", "close"])
            .with_columns(
                pl.col("close").rolling_mean(window_size=n, min_periods=n).over("asset").alias("MA_close")
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .with_columns(
                pl.when(
                    pl.col("MA_close").is_not_null()
                    & (pl.col("MA_close") != 0)
                )
                .then(
                    pl.col("close").cast(pl.Float64)
                    / pl.col("MA_close").cast(pl.Float64)
                    - 1.0
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()



    def calc_price_position_nd(self, date: str, n: int = 5) -> pl.LazyFrame:
        factor_name = f"DailyPriceVolume_price_position_{n}d"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        end_date = date
        start_date = api.shift_tradeday(end_date, -n)
        day_df = dc.md.read_data_batch(
            beg_date=start_date,
            end_date=end_date,
            instrument=dc.Instrument.STOCK,
            datatype=dc.DataType.KLINE_DAY,
        )
        lf = (
            day_df
            .sort(["asset", "date"])
            .with_columns(
                pl.col("close")
                .cast(pl.Float64)
                .rolling_min(window_size=n, min_periods=n)
                .over("asset")
                .alias("min_close")
            )
            .with_columns(
                pl.col("close")
                .cast(pl.Float64)
                .rolling_max(window_size=n, min_periods=n)
                .over("asset")
                .alias("max_close")
            )
            .filter(
                pl.col("date") == pl.lit(end_date).str.to_date()
            )
            .with_columns(
                pl.when(
                    pl.col("max_close").is_not_null()
                    & pl.col("min_close").is_not_null()
                    & (
                        pl.col("max_close").cast(pl.Float64)
                        > pl.col("min_close").cast(pl.Float64)
                    )
                )
                .then(
                    (
                        pl.col("close").cast(pl.Float64)
                        - pl.col("min_close").cast(pl.Float64)
                    )
                    /
                    (
                        pl.col("max_close").cast(pl.Float64)
                        - pl.col("min_close").cast(pl.Float64)
                    )
                )
                .otherwise(0.0)
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf.lazy()

    
