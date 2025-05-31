# utils/forecasting.py

import pandas as pd
from datetime import timedelta

def prepare_sales_weekly_all(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as before: aggregate raw orders into weekly sums, for all weeks (past & future).
    Result columns: ItemCode, ds (week ending Sunday), y (sum).
    """
    df = sales_df.copy()
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df["QuantityOrdered"] = pd.to_numeric(df["QuantityOrdered"], errors="coerce", downcast="integer")
    df = df.dropna(subset=["OrderDate", "ItemCode", "QuantityOrdered"])

    weekly = (
        df
        .groupby(
            ["ItemCode", pd.Grouper(key="OrderDate", freq="W-SUN")],
            as_index=False
        )
        .agg({"QuantityOrdered": "sum"})
        .rename(columns={"OrderDate": "ds", "QuantityOrdered": "y"})
    )
    return weekly

def split_hist_future(weekly_all: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Split weekly_all into:
      - weekly_hist: weeks up to and including today
      - weekly_future_actual: weeks strictly after today
    """
    today = pd.Timestamp.today().normalize()
    weekly_hist = weekly_all[weekly_all["ds"] <= today].copy()
    weekly_future_actual = weekly_all[weekly_all["ds"] > today].copy()
    return weekly_hist, weekly_future_actual

def seasonal_naive_forecast(weekly_hist: pd.DataFrame, forecast_weeks: int) -> pd.DataFrame:
    """
    For each SKU, forecast the next `forecast_weeks` Sundays by taking, for that ISO week number,
    the average of all historic values in the same ISO week of prior years.

    weekly_hist: DataFrame with columns [ItemCode, ds (week-ending Sunday), y]
    forecast_weeks: number of future weeks to forecast (each week-ending Sunday)

    Returns a DataFrame with columns [ItemCode, ds (future Sunday), yhat]
    """
    if weekly_hist.empty:
        return pd.DataFrame(columns=["ItemCode", "ds", "yhat"])

    # 1) Annotate historic data with ISO year and ISO week number
    w = weekly_hist.copy()
    w["ISO_Year"] = w["ds"].dt.isocalendar().year
    w["ISO_Week"] = w["ds"].dt.isocalendar().week

    # 2) Compute (SKU, ISO_Week) → average historic y across all years
    seasonal_avg = (
        w.groupby(["ItemCode", "ISO_Week"], as_index=False)
         .agg({"y": "mean"})
         .rename(columns={"y": "y_seasonal_avg"})
    )
    # 3) Determine the next N week-ending Sundays starting from next Sunday
    last_hist_sunday = weekly_hist["ds"].max()
    # If last_hist_sunday is not a Sunday (unlikely, since we aggregated by W-SUN), still proceed
    # But generally ds are Sundays.

    future_sundays = []
    cur = last_hist_sunday + pd.Timedelta(days=7)
    for _ in range(forecast_weeks):
        future_sundays.append(cur)
        cur = cur + pd.Timedelta(days=7)
    future_df = pd.DataFrame({"ds": future_sundays})
    # Compute ISO week for each future Sunday (use dt.isocalendar)
    future_df["ISO_Week"] = future_df["ds"].dt.isocalendar().week

    # 4) Cross join each SKU with future_df to assign each future Sunday an SKU
    unique_skus = weekly_hist["ItemCode"].unique()
    sku_df = pd.DataFrame({"ItemCode": unique_skus})
    sku_future = sku_df.merge(future_df, how="cross")  # each SKU × each future ds

    # 5) Left‐join on (ItemCode, ISO_Week) to get y_seasonal_avg
    sku_future = sku_future.merge(
        seasonal_avg,
        on=["ItemCode", "ISO_Week"],
        how="left"
    )

    # 6) Fill missing seasonal avg with zero (if no historic data in that week)
    sku_future["yhat"] = sku_future["y_seasonal_avg"].fillna(0.0)

    # 7) Return only columns needed
    return sku_future[["ItemCode", "ds", "yhat"]]

def batch_seasonal_naive_forecast(weekly_hist: pd.DataFrame, forecast_weeks: int) -> pd.DataFrame:
    """
    A wrapper that simply calls seasonal_naive_forecast. 
    Provided for naming consistency with previous batch_forecast_weekly.
    """
    return seasonal_naive_forecast(weekly_hist, forecast_weeks)
