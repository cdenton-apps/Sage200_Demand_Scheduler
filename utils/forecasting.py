import pandas as pd
from prophet import Prophet

def prepare_sales_weekly(sales_df: pd.DataFrame) -> pd.DataFrame:
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
    weekly = weekly[weekly["ds"] <= pd.Timestamp.today()]
    return weekly

def forecast_for_sku_weekly(weekly_df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    if weekly_df.shape[0] < 4:
        return pd.DataFrame(columns=["ItemCode", "ds", "yhat", "yhat_lower", "yhat_upper"])

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(weekly_df[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods, freq="W-SUN")
    forecast = model.predict(future)

    last_hist_week = weekly_df["ds"].max()
    fcst_future = forecast[forecast["ds"] > last_hist_week].copy()
    fcst_future["ItemCode"] = weekly_df["ItemCode"].iloc[0]
    return fcst_future[["ItemCode", "ds", "yhat", "yhat_lower", "yhat_upper"]]

def batch_forecast_weekly(sales_weekly: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    all_forecasts = []
    for sku, group in sales_weekly.groupby("ItemCode"):
        sku_df = group.sort_values("ds")
        sku_df["ItemCode"] = sku
        fcst = forecast_for_sku_weekly(sku_df, periods=periods)
        all_forecasts.append(fcst)
    if all_forecasts:
        return pd.concat(all_forecasts, ignore_index=True)
    else:
        return pd.DataFrame(columns=["ItemCode", "ds", "yhat", "yhat_lower", "yhat_upper"])
