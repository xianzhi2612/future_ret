from gc import collect
from pathlib import Path
import select
from turtle import end_fill
from typing import overload
from dynaconf import base
import polars as pl
import blazestore as bs
import datacenter as dc
import xcals
from xcals import api
from datetime import date
import numpy as np  
from scipy import stats
import matplotlib.pyplot as plt
import math
import polars_ds as pds
from alphamaster.rack import Rack
from alphamaster.polens import FactorAnalyzer

class Liquid_factor:

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
        analyzer = (
            FactorAnalyzer(data, group_col=industry_name)
            if industry_name != "false" and industry_name in data.columns
            else FactorAnalyzer(data)
        )
        analyzer.preprocess(periods=[1, 5, 10], quantiles=quantiles, demean=True)
        analyzer.analyze()
        stats = analyzer.summary_stats()
        print(stats)
        a_1=analyzer.plot("summary",period=period)
        analyzer.plot("turnover", period=period)
        if industry_name != "false" and industry_name in data.columns:
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
    

    def calc_turnover(self, date: str) -> pl.LazyFrame:
        '''
        计算换手率

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'

        返回值：
        pl.LazyFrame，包含三列：
        1. date：日期
        2. asset：股票代码
        3. turnover：对应股票当日的换手率
        '''
        factor_name = "Liquid_turnover"
        factor_path = self.get_factor_path(factor_name, date)

        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        

        daily_data = dc.md.read_data_batch(
            date,
            date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_DAY
        )
        capital_df=self.load_effective_capital_on(date)
        '''计算换手率'''
        lf = (
            daily_data
            .join(
                capital_df,
                on=["asset", "date"],
                how="inner"
            )
            .with_columns(
                (pl.col("volume") / pl.col("AFloats")).alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["asset", "date"])
        )

        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)

        return lf



    def calc_turnover_factor(self, date: str) -> pl.LazyFrame:
        '''
        计算换手率因子

        传入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的换手率因子值
        返回列名为：
        date, asset, turnover_factor
        '''
        factor_name = "Liquid_turnover_factor"
        factor_path = self.get_factor_path(factor_name, date)

        if factor_path.exists():
            return pl.scan_parquet(factor_path)
        
        end_date = date
        start_date = api.shift_tradeday(end_date, -19)
        trade_dates = api.get_tradingdays(start_date, end_date)
        dfs = []
        for d in trade_dates:
            df = self.calc_turnover(d).collect()
            dfs.append(df)
        df=pl.concat(dfs).sort(["asset", "date"]).lazy()
        lf = (
            df
            .with_columns(
                pl.col("Liquid_turnover").cast(pl.Float64)
            )
            .sort(["asset", "date"])
            .with_columns(
                pl.col("Liquid_turnover")
                .rolling_sum(window_size=20,min_samples=20)
                .over("asset")
                .alias("turnover_sum")
            )
            .filter(pl.col("turnover_sum")>0)
            .with_columns(
                pl.col("turnover_sum")
                .log()
                .alias(factor_name)
            )
            .select(
                ["date", "asset", factor_name]
            )
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf


    def calc_ILLIQ(self, date: str, t: int = 20) -> pl.LazyFrame:
        '''
        计算ILLIQ因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        t: int，计算ILLIQ的时间窗口，默认为20天

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的ILLIQ因子值
        ILLIQ因子列名为ILLIQ_factor
        '''
        factor_name = "Liquid_ILLIQ_factor"
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
                        pl.col("close").cast(pl.Float64).shift(1)
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
                pl.col("daily_ILLIQ").mean().alias(factor_name)
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



    def calc_LIQ_factor(self, date: str) -> pl.LazyFrame:
        '''
        获取指定日期的LIQ因子数据

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应的LIQ因子值
        LIQ因子列名为LIQ_factor
        '''
        factor_name = "Liquid_LIQ_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        capital_df=self.load_effective_capital_on(date)
        daily_df = dc.md.read_data_batch(
            date,
            date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_DAY
        )
        turn_over_factor=self.calc_turnover_factor(date)
        lf = (
            daily_df
            .select(["date", "asset", "close", "amount"])
            .sort(["asset", "date"])
            .join(
                capital_df, 
                on=["asset", "date"], 
                how="inner"
            )
            .join(
                turn_over_factor,
                on=["asset", "date"],
                how="inner"
            )
            .with_columns(
                (
                pl.col("close").cast(pl.Float64) *
                pl.col("AFloats").cast(pl.Float64)
                )
                .log()
                .alias("LnFMV")
            )
            .select(["date", "asset", "LnFMV", "Liquid_turnover_factor"])
            
            .filter(
                pl.col("LnFMV").is_not_null() &
                pl.col("Liquid_turnover_factor").is_not_null()
            )
            .with_columns(
                pds.simple_lin_reg(
                    pl.col("LnFMV").cast(pl.Float64),
                    target=pl.col("Liquid_turnover_factor").cast(pl.Float64),
                    add_bias=True,
                    return_pred=True
                ).alias("lr_pred")
            )
            .unnest("lr_pred")
            .rename({
                "pred": "turnover_fit",
                "resid": factor_name
            })
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf



    #构建时段换手率
    def calc_intraday_turnover_table(self, date: str) -> pl.LazyFrame:
        '''
        计算指定交易日四个时间段的换手率

        输入参数：
        date: str，计算时间段换手率的日期，格式为'YYYY-MM-DD'

        返回值：
        pl.LazyFrame，包含日期、股票代码和四个时间段的换手率
        换手率列名为turnover_1、turnover_2、turnover_3、turnover_4
        '''
        factor_name = "Liquid_intraday_turnover"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        minute_data = dc.md.read_data_batch(
            date,
            date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_MINUTE
        )

        base_df = self.load_effective_capital_on(date).select(["date", "asset", "AFloats"])

        session_volume_df = (
            minute_data
            .select(["asset", "date", "datetime", "volume"])
            .with_columns([
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.minute().alias("minute"),
            ])
            .with_columns(
                pl.when(
                    ((pl.col("hour") == 1) & (pl.col("minute") >= 30)) |
                    ((pl.col("hour") == 2) & (pl.col("minute") < 30))
                ).then(pl.lit("volume_1"))
                .when(
                    ((pl.col("hour") == 2) & (pl.col("minute") >= 30)) |
                    ((pl.col("hour") == 3) & (pl.col("minute") < 30))
                ).then(pl.lit("volume_2"))
                .when(
                    pl.col("hour") == 5
                ).then(pl.lit("volume_3"))
                .when(
                    (pl.col("hour") == 6) |
                    ((pl.col("hour") == 7) & (pl.col("minute") == 0))
                ).then(pl.lit("volume_4"))
                .otherwise(None)
                .alias("session")
            )
            .filter(pl.col("session").is_not_null())
            .group_by(["asset", "session"])
            .agg(pl.col("volume").sum().alias("session_volume"))
            .collect()
            .pivot(
                values="session_volume",
                index="asset",
                on="session"
            )
            .lazy()
        )

        lf = (
            base_df
            .join(session_volume_df, on="asset", how="left")
            .with_columns([
                (pl.col("volume_1").fill_null(0) / pl.col("AFloats")).alias("turnover_1"),
                (pl.col("volume_2").fill_null(0) / pl.col("AFloats")).alias("turnover_2"),
                (pl.col("volume_3").fill_null(0) / pl.col("AFloats")).alias("turnover_3"),
                (pl.col("volume_4").fill_null(0) / pl.col("AFloats")).alias("turnover_4"),
            ])
            .select(["date", "asset", "turnover_1", "turnover_2", "turnover_3", "turnover_4"])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    

    def calc_intraday_turnover_table_factor(self, date: str ,bucket: int = 1) -> pl.LazyFrame:
        '''
        计算指定交易日四个时间段中某一时间段的换手率因子

        输入参数：
        date: str，计算时间段换手率因子的日期，格式为'YYYY-MM-DD'
        bucket: int，要计算的时间段，取值为1-4，默认为1

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应时间段的换手率因子值
        换手率因子列名为turnover_{bucket}_factor
        '''
        factor_name = f"Liquid_turnover_{bucket}_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        turnover_col = f"turnover_{bucket}"
        end_date = date
        start_date = api.shift_tradeday(end_date, -19)
        trade_dates = api.get_tradingdays(start_date, end_date)
        dfs = []
        for d in trade_dates:
            df = self.calc_intraday_turnover_table(d).collect()
            dfs.append(df)
        df = pl.concat(dfs).sort(["asset", "date"]).lazy()

        lf = (
            df
            .with_columns(
                pl.col(turnover_col).cast(pl.Float64)
            )
            .select(["date", "asset", turnover_col])
            .sort(["asset", "date"])
            .with_columns(
                pl.col(turnover_col)
                .rolling_sum(window_size=20, min_samples=20)
                .over("asset")
                .alias(f"turnover_sum_{bucket}")
            )
            .filter(pl.col(f"turnover_sum_{bucket}") > 0)
            .with_columns(
                pl.col(f"turnover_sum_{bucket}")
                .log()
                .alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    

    def calc_LIQ_table_factor(self,date: str, bucket : int = 1) -> pl.LazyFrame:
        '''
        计算指定日期对应时间段的LIQ_table因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        bucket: int，要计算的时段，取值为1-4，默认为1

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应时间段的LIQ_table因子值
        因子列名为LIQ_{bucket}_factor
        '''
        factor_name = f"Liquid_LIQ_{bucket}_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        capital_df=self.load_effective_capital_on(date)
        daily_df = dc.md.read_data_batch(
            date,
            date,
            dc.Instrument.STOCK,
            dc.DataType.KLINE_DAY
        )
        turn_over_factor=self.calc_intraday_turnover_table_factor(date, bucket)
        lf = (
            daily_df
            .select(["date", "asset", "close", "amount"])
            .sort(["asset", "date"])
            .join(
                capital_df, 
                on=["asset", "date"], 
                how="inner"
            )
            .join(
                turn_over_factor,
                on=["asset", "date"],
                how="inner"
            )   
            .with_columns(
                (pl.col("close").cast(pl.Float64) *
                pl.col("AFloats").cast(pl.Float64))
                .log()
                .alias("LnFMV")
            )
            .select(["date", "asset", "LnFMV", f"Liquid_turnover_{bucket}_factor"])
            .filter(
                pl.col("LnFMV").is_not_null() &
                pl.col(f"Liquid_turnover_{bucket}_factor").is_not_null()
            )
            .with_columns(
                pds.simple_lin_reg(
                    pl.col("LnFMV").cast(pl.Float64),
                    target=pl.col(f"Liquid_turnover_{bucket}_factor").cast(pl.Float64),
                    add_bias=True,
                    return_pred=True
                ).alias("lr_pred")
            )
            .unnest("lr_pred")
            .rename({
                "pred": f"turnover_fit_{bucket}",
                "resid": factor_name
            })
            .select(["date","asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    


    def calc_pure_LIQ_factor(self,date: str, bucket: int = 1) -> pl.LazyFrame:
        '''
        计算指定日期对应时间段的pure_LIQ因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        bucket: int，要计算的时段，取值为1-4，默认为1

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应时间段的pure_LIQ因子值
        因子列名为pureLIQ_{bucket}_factor
        '''
        factor_name = f"Liquid_pureLIQ_{bucket}_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        LIQ_factor = self.calc_LIQ_factor(date)
        LIQ_table = self.calc_LIQ_table_factor(date , bucket)
        lf = (
            LIQ_factor
            .join(
                LIQ_table,
                on=["date", "asset"],
                how="inner"
            )
            .sort(["asset", "date"])
            .with_columns(
                pds.simple_lin_reg
                (pl.col("Liquid_LIQ_factor").cast(pl.Float64),
                target=pl.col(f"Liquid_LIQ_{bucket}_factor").cast(pl.Float64),
                add_bias=True,
                return_pred=True
                ).alias("lr_pred")
                )
            .unnest("lr_pred")
            .rename({
                "pred": f"LIQ_table_fit_{bucket}",
                "resid": factor_name
            })
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    

    def calc_mixLIQ_factor(self, date: str) -> pl.LazyFrame:
        '''
        计算指定日期对应时间段的mixLIQ因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应时间段的mixLIQ因子值
        因子列名为mixLIQ_factor
        '''
        factor_name = "Liquid_mixLIQ_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)


        LIQ_factor = self.calc_LIQ_factor(date)
        pureLIQ_2_factor = self.calc_pure_LIQ_factor(date, bucket=2)
        pureLIQ_4_factor = self.calc_pure_LIQ_factor(date, bucket=4)
        lf = (
            LIQ_factor
            .join(
                pureLIQ_2_factor,
                on=["date", "asset"],
                how="inner"
            )
            .join(
                pureLIQ_4_factor,
                on=["date", "asset"],
                how="inner"
            )
            #因子横截面标准化处理
            .with_columns([
                (
                    (pl.col("Liquid_pureLIQ_2_factor") - pl.col("Liquid_pureLIQ_2_factor").mean()) 
                    / pl.col("Liquid_pureLIQ_2_factor").std()
                ).alias("z_pureLIQ_2"),
                (
                    (pl.col("Liquid_pureLIQ_4_factor") - pl.col("Liquid_pureLIQ_4_factor").mean()) 
                    / pl.col("Liquid_pureLIQ_4_factor").std()
                ).alias("z_pureLIQ_4"),
                (
                    (pl.col("Liquid_LIQ_factor") - pl.col("Liquid_LIQ_factor").mean()) 
                    / pl.col("Liquid_LIQ_factor").std()
                ).alias("z_LIQ")
            ])
            .with_columns(
                (
                    - pl.col("z_pureLIQ_2") /3
                    + pl.col("z_pureLIQ_4") /3
                    + pl.col("z_LIQ") /3
                ).alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
    

    def calc_new_mixLIQ_factor(self, date: str, w_LIQ: float=0.0019808279645508833, w_pure_LIQ_2: float=-0.0007422102136545118, w_pure_LIQ_4: float=0.0027605382919507065) -> pl.LazyFrame:

        '''
        计算指定日期对应时间段的new_mixLIQ因子

        输入参数：
        date: str，计算因子的日期，格式为'YYYY-MM-DD'
        w_LIQ: float，LIQ因子的权重，默认为0.0019808279645508833
        w_pure_LIQ_2: float，pure_LIQ_2因子的权重，默认为-0.0007422102136545118
        w_pure_LIQ_4: float，pure_LIQ_4因子的权重，默认为0.0027605382919507065

        返回值：
        pl.LazyFrame，包含日期、股票代码和对应时间段的new_mixLIQ因子值
        因子列名为new_mixLIQ_factor
        '''
        factor_name = "Liquid_new_mixLIQ_factor"
        factor_path = self.get_factor_path(factor_name, date)
        if factor_path.exists():
            return pl.scan_parquet(factor_path)

        
        LIQ_factor = self.calc_LIQ_factor(date)
        pureLIQ_2_factor = self.calc_pure_LIQ_factor(date, bucket=2)
        pureLIQ_4_factor = self.calc_pure_LIQ_factor(date, bucket=4)
        lf = (
            LIQ_factor
            .join(
                pureLIQ_2_factor,
                on=["date", "asset"],
                how="inner"
            )
            .join(
                pureLIQ_4_factor,
                on=["date", "asset"],
                how="inner"
            )
            .sort(["asset", "date"])
            .with_columns(
                (
                    w_LIQ * pl.col("Liquid_LIQ_factor") +
                    w_pure_LIQ_2 * pl.col("Liquid_pureLIQ_2_factor") +
                    w_pure_LIQ_4 * pl.col("Liquid_pureLIQ_4_factor")
                ).alias(factor_name)
            )
            .select(["date", "asset", factor_name])
            .sort(["date", "asset"])
        )
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        lf.collect().write_parquet(factor_path)
        return lf
