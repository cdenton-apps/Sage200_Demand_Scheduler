import streamlit as st
import pandas as pd
import io
from utils.forecasting import prepare_sales_weekly, batch_forecast_weekly

st.set_page_config(page_title="Weekly Demand Forecast (Sage 200 CSV)", layout="wide")
st.title("🗓️ Sage 200 Weekly Demand Forecast (CSV Only)")

st.markdown(
    """
    Upload **Stock** and **Sales Orders** CSV exports from Sage 200.  
    The app will:
    1. Aggregate sales data by week (Sunday‐ending).
    2. Forecast the next 12 weeks of demand (Prophet).
    3. Compare forecast vs. current stock to recommend reorder quantities.
    4. Show interactive charts for historical vs. forecasted weekly demand.
    5. Allow CSV export of the full “Demand Report.”
    """
)

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
        help="Columns required: OrderDate (YYYY-MM-DD), ItemCode, QuantityOrdered"
    )

if not stock_file or not sales_file:
    st.info("Please upload both Stock and Sales Orders CSVs to proceed.")
    st.stop()

try:
    stock_df = pd.read_csv(stock_file)
    sales_df = pd.read_csv(sales_file)
except Exception as e:
    st.error(f"Could not read CSV files: {e}")
    st.stop()

required_stock_cols = {"ItemCode", "QuantityOnHand"}
required_sales_cols = {"OrderDate", "ItemCode", "QuantityOrdered"}

missing_stock = required_stock_cols - set(stock_df.columns)
missing_sales = required_sales_cols - set(sales_df.columns)

if missing_stock:
    st.error(f"Stock CSV is missing required column(s): {missing_stock}")
    st.stop()
if missing_sales:
    st.error(f"Sales Orders CSV is missing required column(s): {missing_sales}")
    st.stop()

st.success("✅ Both CSVs loaded successfully.")
st.write("**Stock Columns Detected:**", list(stock_df.columns))
st.write("**Sales Orders Columns Detected:**", list(sales_df.columns))

st.header("2. Preview Raw Data")

with st.expander("Preview Stock Data"):
    st.dataframe(stock_df.head(10))

with st.expander("Preview Sales Orders Data"):
    st.dataframe(sales_df.head(10))

st.header("3. Weekly Demand Preparation & Forecast")

weekly_sales = prepare_sales_weekly(sales_df)
if weekly_sales.empty:
    st.error("No valid weekly sales data found. Check your Sales Orders CSV.")
    st.stop()

st.write("**Weekly Sales (last 10 rows):**")
st.dataframe(weekly_sales.tail(10))

forecast_weeks = st.slider(
    "Forecast Horizon (weeks)", min_value=4, max_value=24, value=12
)

forecast_df = batch_forecast_weekly(weekly_sales, periods=forecast_weeks)
if forecast_df.empty:
    st.warning("Not enough data to generate any forecasts (need ≥ 4 weeks of history per SKU).")
else:
    st.write(f"**Forecast for next {forecast_weeks} weeks (first 10 rows):**")
    st.dataframe(forecast_df.head(10))

st.header("4. Demand Report: Stock vs. Weekly Forecast")

if not forecast_df.empty:
    future_weeks = sorted(forecast_df["ds"].unique())
    pivot_fcst = (
        forecast_df[["ItemCode", "ds", "yhat"]]
        .pivot(index="ItemCode", columns="ds", values="yhat")
        .fillna(0)
    )
    pivot_fcst.columns = [col.date().isoformat() for col in pivot_fcst.columns]
    pivot_fcst.reset_index(inplace=True)

    report_df = pd.merge(
        stock_df.rename(columns={"QuantityOnHand": "CurrentStock"}),
        pivot_fcst,
        on="ItemCode",
        how="right"
    ).fillna(0)

    forecast_columns = [c for c in report_df.columns if c not in {"ItemCode", "ItemDescription", "CurrentStock"}]
    report_df["TotalForecastNext{}W".format(forecast_weeks)] = report_df[forecast_columns].sum(axis=1)

    report_df["RecommendReorderQty"] = (
        report_df["TotalForecastNext{}W".format(forecast_weeks)] - report_df["CurrentStock"]
    ).apply(lambda x: int(x) if x > 0 else 0)

    cols_order = ["ItemCode"]
    if "ItemDescription" in report_df.columns:
        cols_order.append("ItemDescription")
    cols_order += ["CurrentStock", "TotalForecastNext{}W".format(forecast_weeks), "RecommendReorderQty"]
    cols_order += forecast_columns
    report_df = report_df[cols_order]

    st.write("**Demand Report (Stock vs. Weekly Forecast):**")
    st.dataframe(report_df)

    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Demand Report as CSV",
        data=csv_buffer.getvalue(),
        file_name="weekly_demand_report.csv",
        mime="text/csv"
    )

    st.header("5. Visualize Weekly Demand for a Selected SKU")

    sku_list = report_df["ItemCode"].tolist()
    chosen_sku = st.selectbox("Select an SKU to visualize:", sku_list)

    if chosen_sku:
        hist_df = weekly_sales[weekly_sales["ItemCode"] == chosen_sku][["ds", "y"]].copy()
        hist_df = hist_df.sort_values("ds")

        fcst_sku = forecast_df[forecast_df["ItemCode"] == chosen_sku][["ds", "yhat"]].copy()

        plot_df = pd.DataFrame({
            "ds": pd.concat([hist_df["ds"], fcst_sku["ds"]], ignore_index=True)
        }).drop_duplicates().sort_values("ds")
        plot_df = plot_df.merge(hist_df.rename(columns={"y": "HistoricalSales"}), on="ds", how="left")
        plot_df = plot_df.merge(fcst_sku.rename(columns={"yhat": "Forecast"}), on="ds", how="left")
        plot_df = plot_df.fillna(0)

        plot_df = plot_df.set_index("ds")
        st.line_chart(plot_df[["HistoricalSales", "Forecast"]], height=400)

        total_fcst = float(fcst_sku["yhat"].sum())
        current_stock = float(report_df.set_index("ItemCode").loc[chosen_sku, "CurrentStock"])

        bar_df = pd.DataFrame({
            "Category": ["Current Stock", f"Forecast Next {forecast_weeks}W"],
            "Quantity": [current_stock, total_fcst]
        }).set_index("Category")

        st.bar_chart(bar_df, height=300)
else:
    st.warning("No forecast data available. Ensure you have at least 4 weeks of sales history per SKU.")
