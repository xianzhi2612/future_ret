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


class Amihud_factor:

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

    


    def calc_mv(self, date: str) -> pl.LazyFrame:
        '''
        计算市值

        输入参数：
        date: str，计算市值的日期，格式为'YYYY-MM-DD'

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的市值
        市值列名为mv
        '''
        factor_name = "Amihud_mv"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        capital_df = self.load_effective_capital_on(date)
        daily_df = dc.md.read_data_batch(
            date,
            date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_DAY
        ).select(["asset", "date", "close"])

        lf = (
            daily_df
            .join(capital_df, on=["asset","date"], how="inner")
            .with_columns(
                (pl.col("close").cast(pl.Float64) * pl.col("AFloats").cast(pl.Float64))
                
                .alias(factor_name)
            )

            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf


    def calc_daily_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为Amihud_daily_factor
        '''
        factor_name = "Amihud_daily_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        
        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        daily_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_DAY
        )

        lf = (
            daily_df
            .sort(["asset", "date"])
            .with_columns(
                (
                    (
                        pl.col("close").cast(pl.Float64) /
                        pl.col("close").cast(pl.Float64).shift(1).over("asset")
                    ) - 1
                ).over("asset").alias("ret")
            )
            .filter(
                pl.col("ret").is_not_null() &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .with_columns(
                (
                    pl.col("ret").abs() /
                    pl.col("amount").cast(pl.Float64)
                ).alias("daily_ILLIQ")
            )
            .group_by("asset")
            .agg(
                pl.col("daily_ILLIQ").mean().log().alias(factor_name)
            )
            .with_columns(
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    

    def calc_60min_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算60min_Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为60min_Amihud_factor
        '''

        factor_name = "Amihud_60min_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        
        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )
        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
            pl.col("datetime").dt.hour().alias("_hour"),
            pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                pl.when(
                    ((pl.col("_hour") == 1) & (pl.col("_minute") >= 31)) |
                    ((pl.col("_hour") == 2) & (pl.col("_minute") <= 30)) 
                )
                .then(1)
                .when(
                ((pl.col("_hour") == 2) & (pl.col("_minute") >= 31)) |
                ((pl.col("_hour") == 3) & (pl.col("_minute") <= 30))
                )
                .then(2)
                .when(
                    ((pl.col("_hour") == 5) & (pl.col("_minute") >= 1)) |
                    ((pl.col("_hour") == 6) & (pl.col("_minute") == 0))
                )
                .then(3)
                .when(
                    ((pl.col("_hour") == 6) & (pl.col("_minute") >= 1)) |
                    ((pl.col("_hour") == 7) & (pl.col("_minute") == 0))
                )
                .then(4)
                .otherwise(None)
                .alias("bucket")
            )
            .filter(
                pl.col("bucket").is_not_null() &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .group_by(["asset", "date", "bucket"])
            .agg(
                pl.col("close").cast(pl.Float64).sort_by("datetime").last().alias("close_60m"),
                pl.col("open").cast(pl.Float64).sort_by("datetime").first().alias("open_60m"),
                pl.col("amount").cast(pl.Float64).sum().alias("amount_60m"),
                pl.len().alias("n_min")
            )
            .filter(
                pl.col("open_60m").is_not_null()
                & pl.col("close_60m").is_not_null()
                & pl.col("amount_60m").is_not_null()
                & (pl.col("amount_60m") > 0)
            )
            .with_columns(
                (
                    (
                        pl.col("close_60m") /
                        pl.col("open_60m")
                    ) - 1
                ).alias("ret_60m")
            )
            .with_columns(
                (
                    pl.col("ret_60m").abs() /
                    pl.col("amount_60m")
                ).alias("illiq_60m")
            )
            .group_by("asset")
            .agg(
                pl.col("illiq_60m").mean().alias("Amihud_raw"),
                pl.col("illiq_60m").count().alias("n_obs")
            )
            .filter(
                pl.col("Amihud_raw").is_not_null()
                & (pl.col("Amihud_raw") > 0)
            )
            .with_columns(
                pl.col("Amihud_raw").log().alias(factor_name),
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    


    def calc_30min_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算30min_Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为Amihud_30min_factor
        '''

        factor_name = "Amihud_30min_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        
        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )
        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
            pl.col("datetime").dt.hour().alias("_hour"),
            pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                pl.when(
                    ((pl.col("_hour") == 1) & (pl.col("_minute") >= 31)) |
                    ((pl.col("_hour") == 2) & (pl.col("_minute") == 0))
                ).then(1)

                .when(
                    (pl.col("_hour") == 2) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 30)
                ).then(2)

                .when(
                    ((pl.col("_hour") == 2) & (pl.col("_minute") >= 31)) |
                    ((pl.col("_hour") == 3) & (pl.col("_minute") == 0))
                ).then(3)

                .when(
                    (pl.col("_hour") == 3) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 30)
                ).then(4)

                .when(
                    (pl.col("_hour") == 5) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 30)
                ).then(5)

                .when(
                    ((pl.col("_hour") == 5) & (pl.col("_minute") >= 31)) |
                    ((pl.col("_hour") == 6) & (pl.col("_minute") == 0))
                ).then(6)

                .when(
                    (pl.col("_hour") == 6) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 30)
                ).then(7)

                .when(
                    ((pl.col("_hour") == 6) & (pl.col("_minute") >= 31)) |
                    ((pl.col("_hour") == 7) & (pl.col("_minute") == 0))
                ).then(8)
                .otherwise(None)
                .alias("bucket")
            )
            .filter(
                pl.col("bucket").is_not_null() &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .group_by(["asset", "date", "bucket"])
            .agg(
                pl.col("close").cast(pl.Float64).sort_by("datetime").last().alias("close_30m"),
                pl.col("open").cast(pl.Float64).sort_by("datetime").first().alias("open_30m"),
                pl.col("amount").cast(pl.Float64).sum().alias("amount_30m"),
                pl.len().alias("n_min")
            )
            .filter(
                pl.col("open_30m").is_not_null()
                & pl.col("close_30m").is_not_null()
                & pl.col("amount_30m").is_not_null()
                & (pl.col("amount_30m") > 0)
            )
            .with_columns(
                (
                    (
                        pl.col("close_30m") /
                        pl.col("open_30m")
                    ) - 1
                ).alias("ret_30m")
            )
            .with_columns(
                (
                    pl.col("ret_30m").abs() /
                    pl.col("amount_30m")
                ).alias("illiq_30m")
            )
            .group_by("asset")
            .agg(
                pl.col("illiq_30m").mean().alias("Amihud_raw"),
                pl.col("illiq_30m").count().alias("n_obs")
            )
            .filter(
                pl.col("Amihud_raw").is_not_null()
                & (pl.col("Amihud_raw") > 0)
            )
            .with_columns(
                pl.col("Amihud_raw").log().alias(factor_name),
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    



    def calc_15min_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算15min_Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为Amihud_15min_factor
        '''

        factor_name = "Amihud_15min_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )

        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
                pl.col("datetime").dt.hour().alias("_hour"),
                pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                # 09:31-09:45 / UTC 01:31-01:45
                pl.when(
                    (pl.col("_hour") == 1) &
                    (pl.col("_minute") >= 31) &
                    (pl.col("_minute") <= 45)
                ).then(1)

                # 09:46-10:00 / UTC 01:46-02:00
                .when(
                    ((pl.col("_hour") == 1) & (pl.col("_minute") >= 46)) |
                    ((pl.col("_hour") == 2) & (pl.col("_minute") == 0))
                ).then(2)

                # 10:01-10:15 / UTC 02:01-02:15
                .when(
                    (pl.col("_hour") == 2) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 15)
                ).then(3)

                # 10:16-10:30 / UTC 02:16-02:30
                .when(
                    (pl.col("_hour") == 2) &
                    (pl.col("_minute") >= 16) &
                    (pl.col("_minute") <= 30)
                ).then(4)

                # 10:31-10:45 / UTC 02:31-02:45
                .when(
                    (pl.col("_hour") == 2) &
                    (pl.col("_minute") >= 31) &
                    (pl.col("_minute") <= 45)
                ).then(5)

                # 10:46-11:00 / UTC 02:46-03:00
                .when(
                    ((pl.col("_hour") == 2) & (pl.col("_minute") >= 46)) |
                    ((pl.col("_hour") == 3) & (pl.col("_minute") == 0))
                ).then(6)

                # 11:01-11:15 / UTC 03:01-03:15
                .when(
                    (pl.col("_hour") == 3) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 15)
                ).then(7)

                # 11:16-11:30 / UTC 03:16-03:30
                .when(
                    (pl.col("_hour") == 3) &
                    (pl.col("_minute") >= 16) &
                    (pl.col("_minute") <= 30)
                ).then(8)

                # 13:01-13:15 / UTC 05:01-05:15
                .when(
                    (pl.col("_hour") == 5) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 15)
                ).then(9)

                # 13:16-13:30 / UTC 05:16-05:30
                .when(
                    (pl.col("_hour") == 5) &
                    (pl.col("_minute") >= 16) &
                    (pl.col("_minute") <= 30)
                ).then(10)

                # 13:31-13:45 / UTC 05:31-05:45
                .when(
                    (pl.col("_hour") == 5) &
                    (pl.col("_minute") >= 31) &
                    (pl.col("_minute") <= 45)
                ).then(11)

                # 13:46-14:00 / UTC 05:46-06:00
                .when(
                    ((pl.col("_hour") == 5) & (pl.col("_minute") >= 46)) |
                    ((pl.col("_hour") == 6) & (pl.col("_minute") == 0))
                ).then(12)

                # 14:01-14:15 / UTC 06:01-06:15
                .when(
                    (pl.col("_hour") == 6) &
                    (pl.col("_minute") >= 1) &
                    (pl.col("_minute") <= 15)
                ).then(13)

                # 14:16-14:30 / UTC 06:16-06:30
                .when(
                    (pl.col("_hour") == 6) &
                    (pl.col("_minute") >= 16) &
                    (pl.col("_minute") <= 30)
                ).then(14)

                # 14:31-14:45 / UTC 06:31-06:45
                .when(
                    (pl.col("_hour") == 6) &
                    (pl.col("_minute") >= 31) &
                    (pl.col("_minute") <= 45)
                ).then(15)

                # 14:46-15:00 / UTC 06:46-07:00
                .when(
                    ((pl.col("_hour") == 6) & (pl.col("_minute") >= 46)) |
                    ((pl.col("_hour") == 7) & (pl.col("_minute") == 0))
                ).then(16)

                .otherwise(None)
                .alias("bucket")
            )
            .filter(
                pl.col("bucket").is_not_null() &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .group_by(["asset", "date", "bucket"])
            .agg(
                pl.col("close").cast(pl.Float64).sort_by("datetime").last().alias("close_15m"),
                pl.col("open").cast(pl.Float64).sort_by("datetime").first().alias("open_15m"),
                pl.col("amount").cast(pl.Float64).sum().alias("amount_15m"),
                pl.len().alias("n_min")
            )
            .filter(
                pl.col("open_15m").is_not_null()
                & pl.col("close_15m").is_not_null()
                & pl.col("amount_15m").is_not_null()
                & (pl.col("amount_15m") > 0)
            )
            .with_columns(
                (
                    (
                        pl.col("close_15m") /
                        pl.col("open_15m")
                    ) - 1
                ).alias("ret_15m")
            )
            .with_columns(
                (
                    pl.col("ret_15m").abs() /
                    pl.col("amount_15m")
                ).alias("illiq_15m")
            )
            .group_by("asset")
            .agg(
                pl.col("illiq_15m").mean().alias("Amihud_raw"),
                pl.col("illiq_15m").count().alias("n_obs")
            )
            .filter(
                pl.col("Amihud_raw").is_not_null()
                & (pl.col("Amihud_raw") > 0)
            )
            .with_columns(
                pl.col("Amihud_raw").log().alias(factor_name),
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)

        return lf
    


    def calc_10min_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算10min_Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为10min_Amihud_factor
        '''

        factor_name = "Amihud_10min_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        
        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )

        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
                pl.col("datetime").dt.hour().alias("_hour"),
                pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                (
                    pl.col("_hour") * 60 + pl.col("_minute")
                ).alias("_minute_of_day")
            )
            .with_columns(
                pl.when(
                    # 上午盘 UTC 01:31 - 03:30
                    (pl.col("_minute_of_day") >= 1 * 60 + 31) &
                    (pl.col("_minute_of_day") <= 3 * 60 + 30)
                )
                .then(
                    ((pl.col("_minute_of_day") - (1 * 60 + 31)) // 10 + 1)
                )
                .when(
                    # 下午盘 UTC 05:01 - 07:00
                    (pl.col("_minute_of_day") >= 5 * 60 + 1) &
                    (pl.col("_minute_of_day") <= 7 * 60)
                )
                .then(
                    ((pl.col("_minute_of_day") - (5 * 60 + 1)) // 10 + 13)
                )
                .otherwise(None)
                .alias("bucket")
            )
            .filter(
                pl.col("bucket").is_not_null() &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .group_by(["asset", "date", "bucket"])
            .agg(
                pl.col("close").cast(pl.Float64).sort_by("datetime").last().alias("close_10m"),
                pl.col("open").cast(pl.Float64).sort_by("datetime").first().alias("open_10m"),
                pl.col("amount").cast(pl.Float64).sum().alias("amount_10m"),
                pl.len().alias("n_min")
            )
            .filter(
                pl.col("open_10m").is_not_null()
                & pl.col("close_10m").is_not_null()
                & pl.col("amount_10m").is_not_null()
                & (pl.col("amount_10m") > 0)
            )
            .with_columns(
                (
                    (
                        pl.col("close_10m") /
                        pl.col("open_10m")
                    ) - 1
                ).alias("ret_10m")
            )
            .with_columns(
                (
                    pl.col("ret_10m").abs() /
                    pl.col("amount_10m")
                ).alias("illiq_10m")
            )
            .group_by("asset")
            .agg(
                pl.col("illiq_10m").mean().alias("Amihud_raw"),
                pl.col("illiq_10m").count().alias("n_obs")
            )
            .filter(
                pl.col("Amihud_raw").is_not_null()
                & (pl.col("Amihud_raw") > 0)
            )
            .with_columns(
                pl.col("Amihud_raw").log().alias(factor_name),
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)

        return lf
    


    def calc_5min_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算5min_Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为Amihud_5min_factor
        '''

        factor_name = "Amihud_5min_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )

        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
                pl.col("datetime").dt.hour().alias("_hour"),
                pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                (
                    pl.col("_hour") * 60 + pl.col("_minute")
                ).alias("_minute_of_day")
            )
            .with_columns(
                pl.when(
                    # 上午盘 UTC 01:31 - 03:30
                    (pl.col("_minute_of_day") >= 1 * 60 + 31) &
                    (pl.col("_minute_of_day") <= 3 * 60 + 30)
                )
                .then(
                    ((pl.col("_minute_of_day") - (1 * 60 + 31)) // 5 + 1)
                )
                .when(
                    # 下午盘 UTC 05:01 - 07:00
                    (pl.col("_minute_of_day") >= 5 * 60 + 1) &
                    (pl.col("_minute_of_day") <= 7 * 60)
                )
                .then(
                    ((pl.col("_minute_of_day") - (5 * 60 + 1)) // 5 + 25)
                )
                .otherwise(None)
                .alias("bucket")
            )
            .filter(
                pl.col("bucket").is_not_null() &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .group_by(["asset", "date", "bucket"])
            .agg(
                pl.col("close").cast(pl.Float64).sort_by("datetime").last().alias("close_5m"),
                pl.col("open").cast(pl.Float64).sort_by("datetime").first().alias("open_5m"),
                pl.col("amount").cast(pl.Float64).sum().alias("amount_5m"),
                pl.len().alias("n_min")
            )
            .filter(
                pl.col("open_5m").is_not_null()
                & pl.col("close_5m").is_not_null()
                & pl.col("amount_5m").is_not_null()
                & (pl.col("amount_5m") > 0)
            )
            .with_columns(
                (
                    (
                        pl.col("close_5m") /
                        pl.col("open_5m")
                    ) - 1
                ).alias("ret_5m")
            )
            .with_columns(
                (
                    pl.col("ret_5m").abs() /
                    pl.col("amount_5m")
                ).alias("illiq_5m")
            )
            .group_by("asset")
            .agg(
                pl.col("illiq_5m").mean().alias("Amihud_raw"),
                pl.col("illiq_5m").count().alias("n_obs")
            )
            .filter(
                pl.col("Amihud_raw").is_not_null()
                & (pl.col("Amihud_raw") > 0)
            )
            .with_columns(
                pl.col("Amihud_raw").log().alias(factor_name),
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)

        return lf
    

    def calc_1min_amihud_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算1min_Amihud因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Amihud的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Amihud因子值
        Amihud因子列名为1min_Amihud_factor
        '''

        factor_name = "Amihud_1min_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        
        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )

        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
                pl.col("datetime").dt.hour().alias("_hour"),
                pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                (
                    pl.col("_hour") * 60 + pl.col("_minute")
                ).alias("_minute_of_day")
            )
            .filter(
                (
                    # 上午盘 UTC 01:31 - 03:30
                    (
                        (pl.col("_minute_of_day") >= 1 * 60 + 31) &
                        (pl.col("_minute_of_day") <= 3 * 60 + 30)
                    )
                    |
                    # 下午盘 UTC 05:01 - 07:00
                    (
                        (pl.col("_minute_of_day") >= 5 * 60 + 1) &
                        (pl.col("_minute_of_day") <= 7 * 60)
                    )
                )
                &
                pl.col("open").is_not_null()
                &
                pl.col("close").is_not_null()
                &
                pl.col("amount").is_not_null()
                &
                (pl.col("amount") > 0)
            )
            .with_columns(
                (
                    (
                        pl.col("close").cast(pl.Float64) /
                        pl.col("open").cast(pl.Float64)
                    ) - 1
                ).alias("ret_1m")
            )
            .with_columns(
                (
                    pl.col("ret_1m").abs() /
                    pl.col("amount").cast(pl.Float64)
                ).alias("illiq_1m")
            )
            .group_by("asset")
            .agg(
                pl.col("illiq_1m").mean().alias("Amihud_raw"),
                pl.col("illiq_1m").count().alias("n_obs")
            )
            .filter(
                pl.col("Amihud_raw").is_not_null()
                & (pl.col("Amihud_raw") > 0)
            )
            .with_columns(
                pl.col("Amihud_raw").log().alias(factor_name),
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)

        return lf
    


    def calc_rv05m_factor(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算Rv05m因子：5分钟频率的已实现波动率

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算Rv05m的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的Rv05m因子值
        Rv05m因子列名为Rv05m_factor
        '''

        factor_name = "Amihud_Rv05m_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        end_date = date
        start_date = api.shift_tradeday(end_date, -t)

        minute_df = dc.md.read_data_batch(
            start_date,
            end_date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )

        lf = (
            minute_df
            .sort(["asset", "date", "datetime"])
            .with_columns(
                pl.col("datetime").dt.hour().alias("_hour"),
                pl.col("datetime").dt.minute().alias("_minute")
            )
            .with_columns(
                (
                    pl.col("_hour") * 60 + pl.col("_minute")
                ).alias("_minute_of_day")
            )
            .with_columns(
                pl.when(
                    # 上午盘 UTC 01:31 - 03:30
                    (pl.col("_minute_of_day") >= 1 * 60 + 31) &
                    (pl.col("_minute_of_day") <= 3 * 60 + 30)
                )
                .then(
                    ((pl.col("_minute_of_day") - (1 * 60 + 31)) // 5 + 1)
                )
                .when(
                    # 下午盘 UTC 05:01 - 07:00
                    (pl.col("_minute_of_day") >= 5 * 60 + 1) &
                    (pl.col("_minute_of_day") <= 7 * 60)
                )
                .then(
                    ((pl.col("_minute_of_day") - (5 * 60 + 1)) // 5 + 25)
                )
                .otherwise(None)
                .alias("bucket")
            )
            .filter(
                pl.col("bucket").is_not_null()
                & pl.col("open").is_not_null()
                & pl.col("close").is_not_null()
            )

            # 先把1分钟数据聚合成5分钟K线
            .group_by(["asset", "date", "bucket"])
            .agg(
                pl.col("open").cast(pl.Float64).sort_by("datetime").first().alias("open_5m"),
                pl.col("close").cast(pl.Float64).sort_by("datetime").last().alias("close_5m"),
                pl.len().alias("n_min")
            )
            .filter(
                pl.col("open_5m").is_not_null()
                & pl.col("close_5m").is_not_null()
                & (pl.col("open_5m") > 0)
            )

            # 计算5分钟收益率
            .with_columns(
                (
                    pl.col("close_5m") / pl.col("open_5m") - 1
                ).alias("ret_5m")
            )

            # 每只股票每天：sqrt(sum(5分钟收益率平方))
            .group_by(["asset", "date"])
            .agg(
                pl.col("ret_5m").pow(2).sum().sqrt().alias("Rv05m_day"),
                pl.col("ret_5m").count().alias("n_5m")
            )
            .filter(
                pl.col("Rv05m_day").is_not_null()
                & (pl.col("Rv05m_day") >= 0)
            )

            # 过去t天每日Rv05m取平均
            .group_by("asset")
            .agg(
                pl.col("Rv05m_day").mean().alias(factor_name),
                pl.col("Rv05m_day").count().alias("n_obs")
            )
            .filter(
                pl.col(factor_name).is_not_null()
            )
            .with_columns(
                pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d").alias("date")
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)

        return lf


