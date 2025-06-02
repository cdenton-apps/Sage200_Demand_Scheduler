# ======================================
# File: app.py
# ======================================

import streamlit as st
import pandas as pd
import io
import math
from datetime import datetime, timedelta
from PIL import Image
from utils.forecasting import batch_seasonal_naive_forecast

# ─────────────────────────────────────────
# (1) STREAMLIT PAGE CONFIG (must come first)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Solidus Demand Forecast",
    layout="wide"
)

# ─────────────────────────────────────────
# (2) HIDE STREAMLIT MENU & FOOTER (optional)
# ─────────────────────────────────────────
hide_streamlit_style = """
    <style>
      /* Hide top-right menu */
      #MainMenu { visibility: hidden; }
      /* Hide “Made with Streamlit” footer */
      footer { visibility: hidden; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ─────────────────────────────────────────
# (3) SHOW LOGO + TITLE SIDE-BY-SIDE
# ─────────────────────────────────────────
# Create two columns with ratio 1:3
col_logo, col_text = st.columns([1, 3], gap="medium")

with col_logo:
    logo_path = "assets/solidus_logo.png"
    try:
        logo_img = Image.open(logo_path)
        # Force a fixed width so it doesn’t expand too tall.
        st.image(logo_img, width=150)
    except Exception:
        st.warning(
            f"⚠️ Could not load logo at '{logo_path}'.\n"
            "Please confirm the file exists and is a valid PNG."
        )

with col_text:
    # Title in brand color (#0D4B6A is an example; Streamlit theme primaryColor also applies)
    st.markdown(
        "<h1 style='color:#0D4B6A; margin-bottom:0.25em;'>"
        "Solidus Demand Forecast</h1>",
        unsafe_allow_html=True
    )
    # Instructions / description next to the logo
    st.markdown(
        """
        **Upload four CSV exports from Sage 200:**
        1. **Current Stock** (ItemCode, ItemDescription*, QuantityOnHand)  
        2. **Past Order Despatches** (DispatchDate, ItemCode, QuantityDispatched)  
        3. **Sales Orders** (OrderDate, ItemCode, QuantityOrdered)  
        4. **Open Works Orders** (EndDate, ItemCode, QuantityPlanned)  

        This app will:
        1. Aggregate **despatches** by week (≤ today) for historic data → used to forecast.  
        2. Aggregate **sales orders** by week and split into:
           - Historic (≤ today) — used to calculate overdue/backlog  
           - **Actual Future** (> today) — used to compare against forecast  
        3. Aggregate **works orders** by week, taking only future weeks (> today) as “Planned Manufacturing.”  
        4. Use a **seasonal-naive** approach (average of the same ISO week in prior years on despatches)  
           to forecast N weeks, using **week-commencing** (Monday) dates.  
        5. Compute **Overdue Orders** = max(HistoricSales – HistoricDespatched, 0).  
        6. Build a **Demand Report** with columns:  
           - ItemCode (SKU)  
           - ItemDescription (SKU description)  
           - CurrentStock  
           - OverdueOrders  
           - TotalForecastNextNW  
           - TotalActualNextNW  
           - TotalPlannedNextNW  
           - **NetDemand** = (CurrentStock + TotalPlannedNextNW) − TotalActualNextNW  
           - **RecommendReorderQty** = round up(NetDemand) to next multiple of 10 (if NetDemand > 0)  
           - Followed by interleaved weekly columns: Forecast / Actual / Planned  
             for each week commencing.  
        7. Provide interactive charts of weekly series (Historic Despatched, Forecast, Actual, Planned) 
           and a bar chart showing CurrentStock vs Forecast vs Actual vs Planned vs Overdue vs Net.  
        8. Allow CSV export of the full Demand Report.
        """,
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────
# (4) STEP 1: UPLOAD CSVs
# ─────────────────────────────────────────
st.header("1. Upload Sage 200 CSVs")

col1, col2 = st.columns(2, gap="large")
with col1:
    stock_file = st.file_uploader(
        label="Upload Current Stock CSV",
        type=["csv"],
        help="Columns: ItemCode, ItemDescription (optional), QuantityOnHand"
    )
    despatch_file = st.file_uploader(
        label="Upload Past Order Despatches CSV",
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

# ─────────────────────────────────────────
# (5) STEP 2: READ & VALIDATE
# ─────────────────────────────────────────
try:
    stock_df = pd.read_csv(stock_file)
    despatch_df = pd.read_csv(despatch_file)
    sales_df = pd.read_csv(sales_file)
    works_df = pd.read_csv(works_file)
except Exception as e:
    st.error(f"❌ Could not read one or more CSV files: {e}")
    st.stop()

# Required‐columns check
req_stock    = {"ItemCode", "QuantityOnHand"}
req_despatch = {"DispatchDate", "ItemCode", "QuantityDispatched"}
req_sales    = {"OrderDate", "ItemCode", "QuantityOrdered"}
req_works    = {"EndDate", "ItemCode", "QuantityPlanned"}

missing_stock    = req_stock    - set(stock_df.columns)
missing_despatch = req_despatch - set(despatch_df.columns)
missing_sales    = req_sales    - set(sales_df.columns)
missing_works    = req_works    - set(works_df.columns)

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

# ─────────────────────────────────────────
# (6) STEP 3: PREVIEW RAW DATA
# ─────────────────────────────────────────
st.header("2. Preview Raw Data")

with st.expander("Preview Stock Data (first 10 rows)"):
    st.dataframe(stock_df.head(10))

with st.expander("Preview Past Order Despatches Data (first 10 rows)"):
    st.dataframe(despatch_df.head(10))

with st.expander("Preview Sales Orders Data (first 10 rows)"):
    st.dataframe(sales_df.head(10))

with st.expander("Preview Open Works Orders Data (first 10 rows)"):
    st.dataframe(works_df.head(10))


# ─────────────────────────────────────────
# (7) STEP 4: PREPARE WEEKLY DATA
# ─────────────────────────────────────────
st.header("3. Weekly Demand Preparation")

# 4A) Past despatches → weekly (week-ending Sunday)
despatch_df["DispatchDate"] = pd.to_datetime(
    despatch_df["DispatchDate"], errors="coerce"
)
despatch_df["QuantityDispatched"] = pd.to_numeric(
    despatch_df["QuantityDispatched"], errors="coerce", downcast="integer"
)
despatch_df = despatch_df.dropna(
    subset=["DispatchDate", "ItemCode", "QuantityDispatched"]
)

weekly_despatch = (
    despatch_df
    .groupby(
        ["ItemCode", pd.Grouper(key="DispatchDate", freq="W-SUN")],
        as_index=False
    )
    .agg({"QuantityDispatched": "sum"})
    .rename(columns={"DispatchDate": "ds", "QuantityDispatched": "y"})
)
# Add “week_begin” (Monday) for each Sunday‐ending week
weekly_despatch["week_begin"] = weekly_despatch["ds"] - pd.Timedelta(days=6)

# 4B) Sales Orders → weekly (week-ending Sunday)
sales_df["OrderDate"] = pd.to_datetime(sales_df["OrderDate"], errors="coerce")
sales_df["QuantityOrdered"] = pd.to_numeric(
    sales_df["QuantityOrdered"], errors="coerce", downcast="integer"
)
sales_df = sales_df.dropna(
    subset=["OrderDate", "ItemCode", "QuantityOrdered"]
)

weekly_sales_all = (
    sales_df
    .groupby(
        ["ItemCode", pd.Grouper(key="OrderDate", freq="W-SUN")],
        as_index=False
    )
    .agg({"QuantityOrdered": "sum"})
    .rename(columns={"OrderDate": "ds", "QuantityOrdered": "y"})
)
weekly_sales_all["week_begin"] = weekly_sales_all["ds"] - pd.Timedelta(days=6)

# 4C) Works Orders → weekly (week-ending Sunday)
works_df["EndDate"] = pd.to_datetime(works_df["EndDate"], errors="coerce")
works_df["QuantityPlanned"] = pd.to_numeric(
    works_df["QuantityPlanned"], errors="coerce", downcast="integer"
)
works_df = works_df.dropna(
    subset=["EndDate", "ItemCode", "QuantityPlanned"]
)

weekly_works_all = (
    works_df
    .groupby(
        ["ItemCode", pd.Grouper(key="EndDate", freq="W-SUN")],
        as_index=False
    )
    .agg({"QuantityPlanned": "sum"})
    .rename(columns={"EndDate": "ds", "QuantityPlanned": "y"})
)
weekly_works_all["week_begin"] = weekly_works_all["ds"] - pd.Timedelta(days=6)

# 4D) Split by “today”
today = pd.Timestamp(datetime.today().date())

weekly_hist = weekly_despatch[
    weekly_despatch["week_begin"] <= today
].copy()
weekly_sales_hist = weekly_sales_all[
    weekly_sales_all["week_begin"] <= today
].copy()
weekly_sales_future = weekly_sales_all[
    weekly_sales_all["week_begin"] > today
].copy()
weekly_works_future = weekly_works_all[
    weekly_works_all["week_begin"] > today
].copy()

# 4E) Helper to show “Week Number / Week Commencing / Week Ending / Value”
def format_weekly_df(df, value_col, rename_col):
    temp = df.copy()
    temp = temp.rename(columns={value_col: rename_col})
    temp["Week Ending"] = temp["week_begin"] + pd.Timedelta(days=6)
    temp["Week Number"] = temp["week_begin"].dt.isocalendar().week
    temp["Week Commencing"] = temp["week_begin"].dt.strftime("%d-%m-%Y")
    temp["Week Ending"] = temp["Week Ending"].dt.strftime("%d-%m-%Y")
    return temp[
        ["ItemCode", "Week Number", "Week Commencing", "Week Ending", rename_col]
    ]

st.subheader("Historic Weekly Despatches (for Forecast)")
if not weekly_hist.empty:
    hist_disp = format_weekly_df(weekly_hist, "y", "Historical Despatched")
    st.dataframe(hist_disp.tail(10))
else:
    st.write("⚠️ No historic despatch data available for forecasting.")

st.subheader("Actual Future Weekly Sales (from Sales Orders)")
if not weekly_sales_future.empty:
    sales_disp = format_weekly_df(weekly_sales_future, "y", "Actual Future Sales")
    st.dataframe(sales_disp.tail(10))
else:
    st.write("⚠️ No future-dated sales orders detected.")

st.subheader("Planned Future Weekly Manufacturing (from Works Orders)")
if not weekly_works_future.empty:
    works_disp = format_weekly_df(weekly_works_future, "y", "Planned Manufacturing")
    st.dataframe(works_disp.tail(10))
else:
    st.write("⚠️ No future-dated works orders detected (so no planned manufacturing).")

# ─────────────────────────────────────────
# (8) STEP 5: SEASONAL-NAIVE FORECAST
# ─────────────────────────────────────────
st.header("4. Seasonal-Naive Forecast (Historic Despatches Only)")

if weekly_hist.empty:
    st.error("❌ Not enough historic despatch data to forecast. At least one year of past weeks is recommended.")
    st.stop()

forecast_weeks = st.slider(
    "Forecast Horizon (weeks)", min_value=4, max_value=24, value=12
)

forecast_df = batch_seasonal_naive_forecast(weekly_hist, forecast_weeks)
forecast_df["week_begin"] = forecast_df["ds"] - pd.Timedelta(days=6)

disp_fcst = forecast_df.copy()
disp_fcst["Week Commencing"] = disp_fcst["week_begin"].dt.strftime("%d-%m-%Y")
disp_fcst["Week Number"] = disp_fcst["week_begin"].dt.isocalendar().week
disp_fcst = disp_fcst.rename(columns={"yhat": "Forecasted Demand"})
disp_fcst = disp_fcst[["ItemCode", "Week Number", "Week Commencing", "Forecasted Demand"]]

st.write(f"**Seasonal-Naive Forecast for next {forecast_weeks} weeks (first 10 rows):**")
st.dataframe(disp_fcst.head(10))

# ─────────────────────────────────────────
# (9) STEP 6: BUILD DEMAND REPORT
# ─────────────────────────────────────────
st.header("5. Demand Report: Stock vs Overdue vs Forecast vs Actual vs Planned vs Net")

# 6A) Pivot forecast
pivot_fcst_raw = (
    forecast_df.pivot(index="ItemCode", columns="week_begin", values="yhat")
    .fillna(0)
)

# 6B) Pivot sales future
pivot_sales_raw = pd.DataFrame()
if not weekly_sales_future.empty:
    pivot_sales_raw = (
        weekly_sales_future.pivot(index="ItemCode", columns="week_begin", values="y")
        .fillna(0)
    )

# 6C) Pivot works future
pivot_works_raw = pd.DataFrame()
if not weekly_works_future.empty:
    pivot_works_raw = (
        weekly_works_future.pivot(index="ItemCode", columns="week_begin", values="y")
        .fillna(0)
    )

# 6D) Combine with stock
if "ItemDescription" in stock_df.columns:
    report_df = stock_df[["ItemCode", "ItemDescription", "QuantityOnHand"]].copy()
    report_df = report_df.rename(columns={"QuantityOnHand": "CurrentStock"})
else:
    report_df = stock_df[["ItemCode", "QuantityOnHand"]].copy()
    report_df = report_df.rename(columns={"QuantityOnHand": "CurrentStock"})
    report_df["ItemDescription"] = ""

# Ensure every SKU in forecast appears in stock
for sku in pivot_fcst_raw.index:
    if sku not in report_df["ItemCode"].values:
        report_df = report_df.append(
            {"ItemCode": sku, "CurrentStock": 0, "ItemDescription": ""},
            ignore_index=True
        )

report_df = report_df.set_index("ItemCode")

all_week_begins = set(pivot_fcst_raw.columns)
all_week_begins |= set(pivot_sales_raw.columns) if not pivot_sales_raw.empty else set()
all_week_begins |= set(pivot_works_raw.columns) if not pivot_works_raw.empty else set()
all_week_begins = sorted(all_week_begins)

forecast_cols = []
actual_cols = []
planned_cols = []

for wb in all_week_begins:
    wb_str = wb.strftime("%d-%m-%Y")

    # Forecast column
    fc_col = f"Forecast {wb_str}"
    report_df[fc_col] = pivot_fcst_raw[wb] if wb in pivot_fcst_raw.columns else 0
    forecast_cols.append(fc_col)

    # Actual column
    ac_col = f"Actual {wb_str}"
    if not pivot_sales_raw.empty and wb in pivot_sales_raw.columns:
        report_df[ac_col] = pivot_sales_raw[wb]
    else:
        report_df[ac_col] = 0
    actual_cols.append(ac_col)

    # Planned column
    pl_col = f"Planned {wb_str}"
    if not pivot_works_raw.empty and wb in pivot_works_raw.columns:
        report_df[pl_col] = pivot_works_raw[wb]
    else:
        report_df[pl_col] = 0
    planned_cols.append(pl_col)

report_df = report_df.reset_index()

# 6E) Totals
report_df[f"TotalForecastNext{forecast_weeks}W"] = report_df[forecast_cols].sum(axis=1)
report_df[f"TotalActualNext{forecast_weeks}W"]   = report_df[actual_cols].sum(axis=1)
report_df[f"TotalPlannedNext{forecast_weeks}W"]  = report_df[planned_cols].sum(axis=1)

# 6F) Overdue = max(HistoricSales - HistoricDespatched, 0)
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
backlog_df["OverdueOrders"] = (
    backlog_df["TotalSoldHistoric"] - backlog_df["TotalDespatched"]
).clip(lower=0)
backlog_df = backlog_df[["ItemCode", "OverdueOrders"]]

report_df = pd.merge(report_df, backlog_df, on="ItemCode", how="left").fillna({"OverdueOrders": 0})

# 6G) NetDemand = TotalActualNextNW - CurrentStock - TotalPlannedNextNW
report_df[f"NetDemandNext{forecast_weeks}W"] = (
    report_df["CurrentStock"] + report_df[f"TotalPlannedNext{forecast_weeks}W"]
    - report_df[f"TotalActualNext{forecast_weeks}W"]
)

# 6H) RecommendReorderQty = round up NetDemand to next multiple of 10
def round_up_to_10(x):
    return math.ceil(x / 10) * 10 if x > 0 else 0

report_df["RecommendReorderQty"] = report_df[f"NetDemandNext{forecast_weeks}W"].apply(round_up_to_10)

# 6I) Final column order
cols_order = [
    "ItemCode", "ItemDescription",
    "CurrentStock", "OverdueOrders",
    f"TotalForecastNext{forecast_weeks}W",
    f"TotalActualNext{forecast_weeks}W",
    f"TotalPlannedNext{forecast_weeks}W",
    f"NetDemandNext{forecast_weeks}W",
    "RecommendReorderQty"
] + [col for trio in zip(forecast_cols, actual_cols, planned_cols) for col in trio]

report_df = report_df[cols_order]

st.write("**Demand Report (SKU | Description | Stock | Overdue | Forecast | Actual | Planned | Net)**")
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

# ─────────────────────────────────────────
# (10) STEP 7: INTERACTIVE CHARTS (PER SKU)
# ─────────────────────────────────────────
st.header("6. Visualize Weekly Demand for a Selected SKU")

sku_list = report_df["ItemCode"].unique().tolist()
chosen_sku = st.selectbox("Select an SKU to visualize:", sku_list)

# 7A) Historic Despatch series (week_begin)
hist_s = (
    weekly_hist[weekly_hist["ItemCode"] == chosen_sku][["week_begin", "y"]]
    .copy()
    .sort_values("week_begin")
)
hist_s = hist_s.rename(columns={"y": "HistoricalDespatched"})

# 7B) Forecast series (week_begin)
fcst_s = (
    forecast_df[forecast_df["ItemCode"] == chosen_sku][["week_begin", "yhat"]]
    .copy()
    .sort_values("week_begin")
)
fcst_s = fcst_s.rename(columns={"yhat": "Forecast"})

# 7C) Actual Future series (week_begin)
act_s = (
    weekly_sales_future[weekly_sales_future["ItemCode"] == chosen_sku][["week_begin", "y"]]
    .copy()
    .sort_values("week_begin")
)
act_s = act_s.rename(columns={"y": "ActualFuture"})

# 7D) Planned series (week_begin)
plan_s = (
    weekly_works_future[weekly_works_future["ItemCode"] == chosen_sku][["week_begin", "y"]]
    .copy()
    .sort_values("week_begin")
)
plan_s = plan_s.rename(columns={"y": "Planned"})

# 7E) Combine all four for plotting
all_dates_plot = pd.DataFrame({
    "week_begin": pd.concat(
        [hist_s["week_begin"], fcst_s["week_begin"], act_s["week_begin"], plan_s["week_begin"]],
        ignore_index=True
    )
}).drop_duplicates().sort_values("week_begin")

plot_df = all_dates_plot.merge(hist_s, on="week_begin", how="left")
plot_df = plot_df.merge(fcst_s, on="week_begin", how="left")
plot_df = plot_df.merge(act_s, on="week_begin", how="left")
plot_df = plot_df.merge(plan_s, on="week_begin", how="left")
plot_df = plot_df.fillna(0).set_index("week_begin")

# 7F) Line chart
st.subheader(f"Weekly Series for {chosen_sku}")
st.line_chart(
    plot_df[["HistoricalDespatched", "Forecast", "ActualFuture", "Planned"]],
    height=400
)

# 7G) Bar chart totals
total_fcst   = float(fcst_s["Forecast"].sum()) if not fcst_s.empty else 0.0
total_act    = float(act_s["ActualFuture"].sum()) if not act_s.empty else 0.0
total_plan   = float(plan_s["Planned"].sum()) if not plan_s.empty else 0.0
current_stk  = float(report_df.loc[report_df["ItemCode"] == chosen_sku, "CurrentStock"].iloc[0])
overdue_val  = float(report_df.loc[report_df["ItemCode"] == chosen_sku, "OverdueOrders"].iloc[0])
net_demand   = float(report_df.loc[report_df["ItemCode"] == chosen_sku, f"NetDemandNext{forecast_weeks}W"].iloc[0])

bar_df = pd.DataFrame({
    "Category": ["Current Stock", "Forecast", "Actual", "Planned", "Overdue", "NetDemand"],
    "Quantity": [current_stk, total_fcst, total_act, total_plan, overdue_val, net_demand]
}).set_index("Category")

st.subheader(f"Stock | Forecast | Actual | Planned | Overdue | Net for next {forecast_weeks} weeks")
st.bar_chart(bar_df, height=300)
