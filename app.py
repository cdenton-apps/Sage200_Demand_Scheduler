# app.py

import streamlit as st
import pandas as pd
import io
from datetime import datetime
from utils.forecasting import (
    prepare_sales_weekly_all,
    split_hist_future,
    batch_seasonal_naive_forecast,
)

st.set_page_config(page_title="Weekly Demand Forecast (Seasonal)", layout="wide")
st.title("🗓️ Sage 200 Weekly Demand Forecast (CSV Only, Seasonal-Naive)")

st.markdown(
    """
    Upload **Stock** and **Sales Orders** CSV exports from Sage 200 (including future-dated orders).  
    The app will:
    1. Aggregate all sales data by week (Sunday-ending).  
    2. Split into “Historic” (≤ today) and “Actual Future” (> today).  
    3. Forecast the next N weeks for each SKU by taking the average of the same ISO week in prior years.  
    4. Compare forecast vs. actual future (from future-dated orders).  
    5. Show a combined Demand Report (CurrentStock vs. Forecast vs. Actual).  
    6. Provide interactive charts for “Historic vs. Forecast vs. Actual” per SKU.  
    7. Allow CSV export of the full Demand Report.
    """
)

# ===========================
# STEP 1: Upload CSVs
# ===========================
st.header("1. Upload Sage 200 CSV Exports")

col1, col2 = st.columns(2)

with col1:
    stock_file = st.file_uploader(
        label="Upload Stock CSV",
        type=["csv"],
        help="Columns required: ItemCode, ItemDescription (optional), QuantityOnHand"
    )
with col2:
    sales_file = st.file_uploader(
        label="Upload Sales Orders CSV",
        type=["csv"],
        help="Columns required: OrderDate (YYYY-MM-DD), ItemCode, QuantityOrdered (including future dates)"
    )

if not stock_file or not sales_file:
    st.info("Please upload both Stock and Sales Orders CSVs to proceed.")
    st.stop()

# ===========================
# STEP 2: Read & Validate
# ===========================
try:
    stock_df = pd.read_csv(stock_file)
    sales_df = pd.read_csv(sales_file)
except Exception as e:
    st.error(f"❌ Could not read CSV files: {e}")
    st.stop()

required_stock_cols = {"ItemCode", "QuantityOnHand"}
required_sales_cols = {"OrderDate", "ItemCode", "QuantityOrdered"}

missing_stock = required_stock_cols - set(stock_df.columns)
missing_sales = required_sales_cols - set(sales_df.columns)

if missing_stock:
    st.error(f"❌ Stock CSV is missing required column(s): {missing_stock}")
    st.stop()
if missing_sales:
    st.error(f"❌ Sales Orders CSV is missing required column(s): {missing_sales}")
    st.stop()

st.success("✅ Both CSVs loaded successfully.")
st.write("**Stock Columns Detected:**", list(stock_df.columns))
st.write("**Sales Orders Columns Detected:**", list(sales_df.columns))

# ===========================
# STEP 3: Preview Raw Data
# ===========================
st.header("2. Preview Raw Data")

with st.expander("Preview Stock Data"):
    st.dataframe(stock_df.head(10))

with st.expander("Preview Sales Orders Data"):
    st.dataframe(sales_df.head(10))

# ===========================
# STEP 4: Prepare Weekly Data
# ===========================
st.header("3. Weekly Demand Preparation")

# 4A) Full weekly aggregation (including future-dated orders)
weekly_all = prepare_sales_weekly_all(sales_df)

if weekly_all.empty:
    st.error("❌ No weekly sales data could be computed. Check your Sales Orders CSV.")
    st.stop()

# 4B) Split into Historic vs. Future
weekly_hist, weekly_future_actual = split_hist_future(weekly_all)

st.write("**Historic Weekly Sales (last 10 rows):**")
st.dataframe(weekly_hist.tail(10))

st.write("**Actual Future Weekly Sales (if any future-dated orders):**")
st.dataframe(weekly_future_actual.tail(10))

# ===========================
# STEP 5: Seasonal-Naive Forecast
# ===========================
st.header("4. Seasonal-Naive Forecast (Historic Only)")

if weekly_hist.empty:
    st.error("❌ Not enough historic data (all sales are future-dated or CSV empty).")
    st.stop()

forecast_weeks = st.slider(
    "Forecast Horizon (weeks)", min_value=4, max_value=24, value=12
)

forecast_df = batch_seasonal_naive_forecast(weekly_hist, forecast_weeks)

if forecast_df.empty:
    st.warning("⚠️ Not enough history to forecast any SKU (need ≥ 1 year of data in each ISO week).")
else:
    # Rename columns for display: ds → "Week Ending", yhat → "Forecasted Sales"
    display_fcst = forecast_df.head(10).copy()
    display_fcst = display_fcst.rename(columns={
        "ds": "Week Ending",
        "yhat": "Forecasted Sales"
    })
    st.write(f"**Seasonal-Naive Forecast for next {forecast_weeks} weeks (first 10 rows):**")
    st.dataframe(display_fcst)

# ===========================
# STEP 6: Build Demand Report
# ===========================
st.header("5. Demand Report: Current Stock vs. Forecast vs. Actual")

if not forecast_df.empty:
    # 6A) Pivot forecast so each row is ItemCode, each column is forecast-week string
    future_weeks = sorted(forecast_df["ds"].unique())
    pivot_fcst = (
        forecast_df[["ItemCode", "ds", "yhat"]]
        .pivot(index="ItemCode", columns="ds", values="yhat")
        .fillna(0)
    )
    pivot_fcst.columns = [col.date().isoformat() for col in pivot_fcst.columns]
    pivot_fcst.reset_index(inplace=True)

    # 6B) Pivot actual future demand so each row is ItemCode, each column is week
    if not weekly_future_actual.empty:
        pivot_actual = (
            weekly_future_actual[["ItemCode", "ds", "y"]]
            .pivot(index="ItemCode", columns="ds", values="y")
            .fillna(0)
        )
        pivot_actual.columns = [f"Actual_{col.date().isoformat()}" for col in pivot_actual.columns]
        pivot_actual.reset_index(inplace=True)
    else:
        pivot_actual = pd.DataFrame({"ItemCode": pivot_fcst["ItemCode"].tolist()})

    # 6C) Merge stock + forecast + actual
    report_df = pd.merge(
        stock_df.rename(columns={"QuantityOnHand": "CurrentStock"}),
        pivot_fcst,
        on="ItemCode",
        how="right"
    ).fillna(0)

    report_df = pd.merge(
        report_df,
        pivot_actual,
        on="ItemCode",
        how="left"
    ).fillna(0)

    # 6D) Compute “TotalForecastNextXW” = sum of forecast columns
    forecast_cols = [
        c for c in report_df.columns
        if (c not in {"ItemCode", "ItemDescription", "CurrentStock"}) and not c.startswith("Actual_")
    ]
    report_df[f"TotalForecastNext{forecast_weeks}W"] = report_df[forecast_cols].sum(axis=1)

    # 6E) Compute “TotalActualNextXW” = sum of actual future columns
    actual_cols = [c for c in report_df.columns if c.startswith("Actual_")]
    report_df[f"TotalActualNext{forecast_weeks}W"] = report_df[actual_cols].sum(axis=1)

    # 6F) Reorder recommendation (based on forecast only)
    report_df["RecommendReorderQty"] = (
        report_df[f"TotalForecastNext{forecast_weeks}W"] - report_df["CurrentStock"]
    ).apply(lambda x: int(x) if x > 0 else 0)

    # 6G) Final column ordering
    cols_order = ["ItemCode"]
    if "ItemDescription" in report_df.columns:
        cols_order.append("ItemDescription")
    cols_order += [
        "CurrentStock",
        f"TotalForecastNext{forecast_weeks}W",
        f"TotalActualNext{forecast_weeks}W",
        "RecommendReorderQty"
    ]
    cols_order += forecast_cols + actual_cols
    report_df = report_df[cols_order]

    # 6H) Display
    st.write("**Demand Report (Stock vs. Seasonal Forecast vs. Actual Future):**")
    st.dataframe(report_df, use_container_width=True)

    # 6I) CSV Download
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Demand Report as CSV",
        data=csv_buffer.getvalue(),
        file_name="seasonal_demand_report_actual_vs_forecast.csv",
        mime="text/csv"
    )

    # ===========================
    # STEP 7: Interactive Charts per SKU
    # ===========================
    st.header("6. Visualize Weekly Demand for a Selected SKU")

    sku_list = report_df["ItemCode"].unique().tolist()
    chosen_sku = st.selectbox("Select an SKU to visualize:", sku_list)

    if chosen_sku:
        # 7A) Historical weekly (ds ≤ today) for chosen SKU
        hist_df = weekly_hist[weekly_hist["ItemCode"] == chosen_sku][["ds", "y"]].copy().sort_values("ds")
        hist_df = hist_df.rename(columns={"y": "HistoricalSales"})

        # 7B) Forecast for chosen SKU
        fcst_sku = forecast_df[forecast_df["ItemCode"] == chosen_sku][["ds", "yhat"]].copy().sort_values("ds")
        fcst_sku = fcst_sku.rename(columns={"yhat": "Forecast"})

        # 7C) Actual future for chosen SKU
        if not weekly_future_actual.empty:
            actual_sku = weekly_future_actual[weekly_future_actual["ItemCode"] == chosen_sku][["ds", "y"]].copy().sort_values("ds")
            actual_sku = actual_sku.rename(columns={"y": "ActualFuture"})
        else:
            actual_sku = pd.DataFrame(columns=["ds", "ActualFuture"])

        # 7D) Combine into one DataFrame for plotting
        plot_idx = pd.DataFrame({
            "ds": pd.concat([hist_df["ds"], fcst_sku["ds"], actual_sku["ds"]], ignore_index=True)
        }).drop_duplicates().sort_values("ds")
        plot_df = plot_idx.merge(hist_df, on="ds", how="left")
        plot_df = plot_df.merge(fcst_sku, on="ds", how="left")
        plot_df = plot_df.merge(actual_sku, on="ds", how="left")
        plot_df = plot_df.fillna(0).set_index("ds")

        # 7E) Line chart (Historical vs. Seasonal Forecast vs. Actual Future)
        st.subheader(f"Weekly Demand Comparison for {chosen_sku}")
        st.line_chart(plot_df[["HistoricalSales", "Forecast", "ActualFuture"]], height=450)

        # 7F) Bar chart: “Forecast vs. Actual Future” summed over next X weeks
        total_fcst = float(fcst_sku["Forecast"].sum()) if not fcst_sku.empty else 0.0
        total_actual = float(actual_sku["ActualFuture"].sum()) if not actual_sku.empty else 0.0
        current_stock = float(report_df.loc[report_df["ItemCode"] == chosen_sku, "CurrentStock"].iloc[0])

        bar_df = pd.DataFrame({
            "Category": ["Current Stock", f"Forecast Next {forecast_weeks}W", f"Actual Next {forecast_weeks}W"],
            "Quantity": [current_stock, total_fcst, total_actual]
        }).set_index("Category")

        st.subheader(f"Stock vs. Seasonal Forecast vs. Actual Next {forecast_weeks} Weeks")
        st.bar_chart(bar_df, height=300)

else:
    st.warning("⚠️ No seasonal forecasts generated. Ensure you have sufficient historic data per SKU.")
