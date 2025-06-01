# ======================================
# File: app.py
# ======================================

import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
from utils.forecasting import (
    batch_seasonal_naive_forecast,
)

st.set_page_config(
    page_title="Solidus Demand Forecast",
    layout="wide"
)

st.title("Solidus Demand Forecast")

st.markdown(
    """
    Upload **four** CSV exports from Sage 200:
    1. **Current Stock** (ItemCode, ItemDescription*, QuantityOnHand)  
    2. **Past Order Despatches** (DispatchDate, ItemCode, QuantityDispatched)  
    3. **Sales Orders** (OrderDate, ItemCode, QuantityOrdered)  
    4. **Open Works Orders** (EndDate, ItemCode, QuantityPlanned)  

    This app will:
    1. Aggregate **despatches** by week (≤ today) for historic data → used to forecast.  
    2. Aggregate **sales orders** by week and split into:
       - Historic (≤ today) — used to calculate overdue/backlog  
       - **Actual Future** (> today) — used to compare against forecast  
    3. Aggregate **works orders** by week, take only future weeks (> today) as “Planned Manufacturing.”  
    4. Use a **seasonal‐naive** approach (average of the same ISO week in prior years on despatches) to forecast N weeks.  
    5. Compute **Overdue Orders** = max(HistoricSales − HistoricDespatched, 0).  
    6. Build a **Demand Report** with columns:  
       - CurrentStock  
       - OverdueOrders  
       - TotalForecastNextNW  
       - TotalActualNextNW  
       - TotalPlannedNextNW  
       - NetDemand = (Forecast + Actual + Overdue − Planned)  
       - RecommendReorderQty = max(NetDemand − CurrentStock, 0)  
    7. Provide interactive charts showing weekly series (Historic Despatched, Forecast, Actual, Planned) and a bar chart of CurrentStock vs. Forecast vs. Actual vs. Planned vs. Overdue vs. Net.  
    8. Allow CSV export of the full Demand Report.
    """
)

# ===========================
# STEP 1: UPLOAD CSVs
# ===========================
st.header("1. Upload Sage 200 CSVs")

col1, col2 = st.columns(2)
with col1:
    stock_file = st.file_uploader(
        label="Upload Current Stock CSV",
        type=["csv"],
        help="Columns: ItemCode, ItemDescription (optional), QuantityOnHand"
    )
    despatch_file = st.file_uploader(
        label="Upload Past Despatches CSV",
        type=["csv"],
        help="Columns: DispatchDate (YYYY-MM-DD), ItemCode, QuantityDispatched"
    )
with col2:
    sales_file = st.file_uploader(
        label="Upload Sales Orders CSV",
        type=["csv"],
        help="Columns: OrderDate (YYYY-MM-DD), ItemCode, QuantityOrdered"
    )
    works_file = st.file_uploader(
        label="Upload Open Works Orders CSV",
        type=["csv"],
        help="Columns: EndDate (YYYY-MM-DD), ItemCode, QuantityPlanned"
    )

if not (stock_file and despatch_file and sales_file and works_file):
    st.info("Please upload all four CSVs to proceed.")
    st.stop()

# ===========================
# STEP 2: READ & VALIDATE
# ===========================
try:
    stock_df = pd.read_csv(stock_file)
    despatch_df = pd.read_csv(despatch_file)
    sales_df = pd.read_csv(sales_file)
    works_df = pd.read_csv(works_file)
except Exception as e:
    st.error(f"❌ Could not read one or more CSV files: {e}")
    st.stop()

# Validate required columns for each
req_stock = {"ItemCode", "QuantityOnHand"}
req_despatch = {"DispatchDate", "ItemCode", "QuantityDispatched"}
req_sales = {"OrderDate", "ItemCode", "QuantityOrdered"}
req_works = {"EndDate", "ItemCode", "QuantityPlanned"}

missing_stock = req_stock - set(stock_df.columns)
missing_despatch = req_despatch - set(despatch_df.columns)
missing_sales = req_sales - set(sales_df.columns)
missing_works = req_works - set(works_df.columns)

if missing_stock:
    st.error(f"Stock CSV missing required column(s): {', '.join(missing_stock)}")
    st.stop()
if missing_despatch:
    st.error(f"Despatches CSV missing required column(s): {', '.join(missing_despatch)}")
    st.stop()
if missing_sales:
    st.error(f"Sales Orders CSV missing required column(s): {', '.join(missing_sales)}")
    st.stop()
if missing_works:
    st.error(f"Works Orders CSV missing required column(s): {', '.join(missing_works)}")
    st.stop()

st.success("✅ All CSVs loaded and validated.")


# ===========================
# STEP 3: PREVIEW RAW DATA
# ===========================
st.header("2. Preview Raw Data")

with st.expander("Preview Stock Data (first 10 rows)"):
    st.dataframe(stock_df.head(10))

with st.expander("Preview Past Despatches Data (first 10 rows)"):
    st.dataframe(despatch_df.head(10))

with st.expander("Preview Sales Orders Data (first 10 rows)"):
    st.dataframe(sales_df.head(10))

with st.expander("Preview Open Works Orders Data (first 10 rows)"):
    st.dataframe(works_df.head(10))


# ===========================
# STEP 4: PREPARE WEEKLY DATA
# ===========================
st.header("3. Weekly Demand Preparation")

# 4A) Despatches → historic weekly (group by week-ending Sunday)
despatch_df["DispatchDate"] = pd.to_datetime(despatch_df["DispatchDate"], errors="coerce")
despatch_df["QuantityDispatched"] = pd.to_numeric(despatch_df["QuantityDispatched"], errors="coerce", downcast="integer")
despatch_df = despatch_df.dropna(subset=["DispatchDate", "ItemCode", "QuantityDispatched"])

weekly_despatch = (
    despatch_df
    .groupby(
        ["ItemCode", pd.Grouper(key="DispatchDate", freq="W-SUN")],
        as_index=False
    )
    .agg({"QuantityDispatched": "sum"})
    .rename(columns={"DispatchDate": "ds", "QuantityDispatched": "y"})
)

# 4B) Sales Orders → all‐weeks
sales_df["OrderDate"] = pd.to_datetime(sales_df["OrderDate"], errors="coerce")
sales_df["QuantityOrdered"] = pd.to_numeric(sales_df["QuantityOrdered"], errors="coerce", downcast="integer")
sales_df = sales_df.dropna(subset=["OrderDate", "ItemCode", "QuantityOrdered"])

weekly_sales_all = (
    sales_df
    .groupby(
        ["ItemCode", pd.Grouper(key="OrderDate", freq="W-SUN")],
        as_index=False
    )
    .agg({"QuantityOrdered": "sum"})
    .rename(columns={"OrderDate": "ds", "QuantityOrdered": "y"})
)

# 4C) Works Orders → all‐weeks
works_df["EndDate"] = pd.to_datetime(works_df["EndDate"], errors="coerce")
works_df["QuantityPlanned"] = pd.to_numeric(works_df["QuantityPlanned"], errors="coerce", downcast="integer")
works_df = works_df.dropna(subset=["EndDate", "ItemCode", "QuantityPlanned"])

weekly_works_all = (
    works_df
    .groupby(
        ["ItemCode", pd.Grouper(key="EndDate", freq="W-SUN")],
        as_index=False
    )
    .agg({"QuantityPlanned": "sum"})
    .rename(columns={"EndDate": "ds", "QuantityPlanned": "y"})
)

# 4D) Split by today
today = pd.Timestamp(datetime.today().date())

# Historic to forecast: from despatches only
weekly_hist = weekly_despatch[weekly_despatch["ds"] <= today].copy()

# Sales: split
weekly_sales_hist = weekly_sales_all[weekly_sales_all["ds"] <= today].copy()
weekly_sales_future = weekly_sales_all[weekly_sales_all["ds"] > today].copy()

# Works: planned future manufacturing (> today)
weekly_works_future = weekly_works_all[weekly_works_all["ds"] > today].copy()

# 4E) Display tables with friendly headers
def format_weekly_df(df, value_col, rename_col):
    """
    df: columns ["ItemCode","ds","y"]
    value_col: original column name ("y")
    rename_col: friendly name for value (e.g. "Historical Despatched")
    Returns: DataFrame with columns:
      ["ItemCode","Week Number","Week Beginning","Week Ending", rename_col]
    """
    temp = df.copy()
    temp = temp.rename(columns={value_col: rename_col})
    temp["Week Ending"] = temp["ds"]
    temp["Week Beginning"] = temp["ds"] - pd.Timedelta(days=6)
    temp["Week Number"] = temp["ds"].dt.isocalendar().week
    temp["Week Beginning"] = temp["Week Beginning"].dt.strftime("%d-%m-%Y")
    temp["Week Ending"] = temp["Week Ending"].dt.strftime("%d-%m-%Y")
    return temp[["ItemCode", "Week Number", "Week Beginning", "Week Ending", rename_col]]

st.subheader("Historic Weekly Despatches (for Forecast)")
if not weekly_hist.empty:
    hist_despatch_display = format_weekly_df(weekly_hist, "y", "Historical Despatched")
    st.dataframe(hist_despatch_display.tail(10))
else:
    st.write("⚠️ No historic despatch data available for forecasting.")

st.subheader("Actual Future Weekly Sales (from Sales Orders)")
if not weekly_sales_future.empty:
    sales_future_display = format_weekly_df(weekly_sales_future, "y", "Actual Future Sales")
    st.dataframe(sales_future_display.tail(10))
else:
    st.write("⚠️ No future-dated sales orders detected.")

st.subheader("Planned Future Weekly Manufacturing (from Works Orders)")
if not weekly_works_future.empty:
    works_future_display = format_weekly_df(weekly_works_future, "y", "Planned Manufacturing")
    st.dataframe(works_future_display.tail(10))
else:
    st.write("⚠️ No future-dated works orders detected (so no planned manufacturing).")


# ===========================
# STEP 5: SEASONAL-NAIVE FORECAST
# ===========================
st.header("4. Seasonal-Naive Forecast (Historic Despatches Only)")

if weekly_hist.empty:
    st.error("❌ Not enough historic despatch data to forecast. At least one year of past weeks is recommended.")
    st.stop()

forecast_weeks = st.slider(
    "Forecast Horizon (weeks)", min_value=4, max_value=24, value=12
)

forecast_df = batch_seasonal_naive_forecast(weekly_hist, forecast_weeks)

disp_fcst = forecast_df.copy()
disp_fcst["Week Beginning"] = disp_fcst["ds"] - pd.Timedelta(days=6)
disp_fcst["Week Number"] = disp_fcst["ds"].dt.isocalendar().week
disp_fcst["Week Beginning"] = disp_fcst["Week Beginning"].dt.strftime("%d-%m-%Y")
disp_fcst = disp_fcst.rename(columns={"yhat": "Forecasted Demand"})
disp_fcst = disp_fcst[["ItemCode", "Week Number", "Week Beginning", "Forecasted Demand"]]

st.write(f"**Seasonal-Naive Forecast for next {forecast_weeks} weeks (first 10 rows):**")
st.dataframe(disp_fcst.head(10))


# ===========================
# STEP 6: BUILD DEMAND REPORT
# ===========================
st.header("5. Demand Report: Stock vs Overdue vs Forecast vs Actual vs Planned vs Net")

# 6A) Pivot Forecast (future weeks only)
future_weeks = sorted(forecast_df["ds"].unique())
pivot_fcst = (
    forecast_df[["ItemCode", "ds", "yhat"]]
    .pivot(index="ItemCode", columns="ds", values="yhat")
    .fillna(0)
)
pivot_fcst.columns = [col.strftime("%d-%m-%Y") for col in pivot_fcst.columns]
pivot_fcst.reset_index(inplace=True)

# 6B) Pivot Sales Future
if not weekly_sales_future.empty:
    pivot_sales = (
        weekly_sales_future[["ItemCode", "ds", "y"]]
        .pivot(index="ItemCode", columns="ds", values="y")
        .fillna(0)
    )
    pivot_sales.columns = [f"Actual_{col.strftime('%d-%m-%Y')}" for col in pivot_sales.columns]
    pivot_sales.reset_index(inplace=True)
else:
    pivot_sales = pd.DataFrame({"ItemCode": pivot_fcst["ItemCode"].tolist()})

# 6C) Pivot Works Future
if not weekly_works_future.empty:
    pivot_works = (
        weekly_works_future[["ItemCode", "ds", "y"]]
        .pivot(index="ItemCode", columns="ds", values="y")
        .fillna(0)
    )
    pivot_works.columns = [f"Planned_{col.strftime('%d-%m-%Y')}" for col in pivot_works.columns]
    pivot_works.reset_index(inplace=True)
else:
    pivot_works = pd.DataFrame({"ItemCode": pivot_fcst["ItemCode"].tolist()})

# 6D) Merge Stock + Forecast + Sales + Works
report_df = pd.merge(
    stock_df.rename(columns={"QuantityOnHand": "CurrentStock"}),
    pivot_fcst,
    on="ItemCode",
    how="right"
).fillna(0)

report_df = pd.merge(report_df, pivot_sales, on="ItemCode", how="left").fillna(0)
report_df = pd.merge(report_df, pivot_works, on="ItemCode", how="left").fillna(0)

# 6E) Compute Totals
all_cols = set(report_df.columns)
junk = {"ItemCode", "ItemDescription", "CurrentStock"}
forecast_cols = [c for c in all_cols if c not in junk and not c.startswith("Actual_") and not c.startswith("Planned_")]
actual_cols = [c for c in all_cols if c.startswith("Actual_")]
planned_cols = [c for c in all_cols if c.startswith("Planned_")]

report_df[f"TotalForecastNext{forecast_weeks}W"] = report_df[forecast_cols].sum(axis=1)
report_df[f"TotalActualNext{forecast_weeks}W"]   = report_df[actual_cols].sum(axis=1) if actual_cols else 0
report_df[f"TotalPlannedNext{forecast_weeks}W"]  = report_df[planned_cols].sum(axis=1) if planned_cols else 0

# 6F) Compute Overdue Orders = max(TotalSoldHistoric – TotalDespatched, 0)
total_despatched = (
    weekly_hist.groupby("ItemCode", as_index=False)
    .agg({"y": "sum"})
    .rename(columns={"y": "TotalDespatched"})
)
total_sold_historic = (
    weekly_sales_hist.groupby("ItemCode", as_index=False)
    .agg({"y": "sum"})
    .rename(columns={"y": "TotalSoldHistoric"})
)
backlog_df = pd.merge(
    total_sold_historic,
    total_despatched,
    on="ItemCode",
    how="outer"
).fillna(0)
backlog_df["OverdueOrders"] = (backlog_df["TotalSoldHistoric"] - backlog_df["TotalDespatched"]).clip(lower=0)
backlog_df = backlog_df[["ItemCode", "OverdueOrders"]]

report_df = pd.merge(report_df, backlog_df, on="ItemCode", how="left").fillna({"OverdueOrders": 0})

# 6G) NetDemand = Forecast + Actual + Overdue − Planned
report_df[f"NetDemandNext{forecast_weeks}W"] = (
    report_df[f"TotalForecastNext{forecast_weeks}W"]
    + report_df[f"TotalActualNext{forecast_weeks}W"]
    + report_df["OverdueOrders"]
    - report_df[f"TotalPlannedNext{forecast_weeks}W"]
)

# 6H) RecommendReorder = max(NetDemand − CurrentStock, 0)
report_df["RecommendReorderQty"] = (
    report_df[f"NetDemandNext{forecast_weeks}W"] - report_df["CurrentStock"]
).apply(lambda x: int(x) if x > 0 else 0)

# 6I) Final column order (ItemDescription only if present)
if "ItemDescription" in report_df.columns:
    cols_order = ["ItemCode", "ItemDescription"]
else:
    cols_order = ["ItemCode"]

cols_order += [
    "CurrentStock",
    "OverdueOrders",
    f"TotalForecastNext{forecast_weeks}W",
    f"TotalActualNext{forecast_weeks}W",
    f"TotalPlannedNext{forecast_weeks}W",
    f"NetDemandNext{forecast_weeks}W",
    "RecommendReorderQty"
] + forecast_cols + actual_cols + planned_cols

report_df = report_df[cols_order]

st.write("**Demand Report (Stock | Overdue | Forecast | Actual | Planned | Net)**")
st.dataframe(report_df, use_container_width=True)

# 6J) CSV Download
csv_buf = io.StringIO()
report_df.to_csv(csv_buf, index=False)
st.download_button(
    label="📥 Download Demand Report CSV",
    data=csv_buf.getvalue(),
    file_name="demand_report_all.csv",
    mime="text/csv"
)

# ===========================
# STEP 7: INTERACTIVE CHARTS (PER SKU)
# ===========================
st.header("6. Visualize Weekly Demand for a Selected SKU")

sku_list = report_df["ItemCode"].unique().tolist()
chosen_sku = st.selectbox("Select an SKU to visualize:", sku_list)

# 7A) Historic Despatch series
hist_s = weekly_hist[weekly_hist["ItemCode"] == chosen_sku][["ds", "y"]].copy().sort_values("ds")
hist_s = hist_s.rename(columns={"y": "HistoricalDespatched"})

# 7B) Forecast series
fcst_s = forecast_df[forecast_df["ItemCode"] == chosen_sku][["ds", "yhat"]].copy().sort_values("ds")
fcst_s = fcst_s.rename(columns={"yhat": "Forecast"})

# 7C) Actual Future series
act_s = weekly_sales_future[weekly_sales_future["ItemCode"] == chosen_sku][["ds", "y"]].copy().sort_values("ds")
act_s = act_s.rename(columns={"y": "ActualFuture"})

# 7D) Planned series
plan_s = weekly_works_future[weekly_works_future["ItemCode"] == chosen_sku][["ds", "y"]].copy().sort_values("ds")
plan_s = plan_s.rename(columns={"y": "Planned"})

# 7E) Combine all four into one DataFrame for plotting
all_dates = pd.DataFrame({
    "ds": pd.concat([hist_s["ds"], fcst_s["ds"], act_s["ds"], plan_s["ds"]], ignore_index=True)
}).drop_duplicates().sort_values("ds")

plot_df = all_dates.merge(hist_s, on="ds", how="left")
plot_df = plot_df.merge(fcst_s, on="ds", how="left")
plot_df = plot_df.merge(act_s, on="ds", how="left")
plot_df = plot_df.merge(plan_s, on="ds", how="left")
plot_df = plot_df.fillna(0).set_index("ds")

# 7F) Line chart
st.subheader(f"Weekly Series for {chosen_sku}")
st.line_chart(plot_df[["HistoricalDespatched", "Forecast", "ActualFuture", "Planned"]], height=400)

# 7G) Compute totals for bar chart
total_fcst   = float(fcst_s["Forecast"].sum()) if not fcst_s.empty else 0.0
total_act    = float(act_s["ActualFuture"].sum()) if not act_s.empty else 0.0
total_plan   = float(plan_s["Planned"].sum()) if not plan_s.empty else 0.0
current_stk  = float(report_df.loc[report_df["ItemCode"] == chosen_sku, "CurrentStock"].iloc[0])
overdue_val  = float(report_df.loc[report_df["ItemCode"] == chosen_sku, "OverdueOrders"].iloc[0])
net_demand   = float(report_df.loc[report_df["ItemCode"] == chosen_sku, f"NetDemandNext{forecast_weeks}W"].iloc[0])

bar_df = pd.DataFrame({
    "Category": ["Current Stock", "Forecast", "ActualFuture", "Planned", "OverdueOrders", "NetDemand"],
    "Quantity": [current_stk, total_fcst, total_act, total_plan, overdue_val, net_demand]
}).set_index("Category")

st.subheader(f"Stock | Forecast | Actual | Planned | Overdue | Net for next {forecast_weeks} weeks")
st.bar_chart(bar_df, height=300)
