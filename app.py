# ===========================
# STEP 6: BUILD DEMAND REPORT
# ===========================
st.header("5. Demand Report: Stock vs Overdue vs Forecast vs Actual vs Planned vs Net")

# 6A) Pivot Forecast on week_begin (Monday)
pivot_fcst_raw = (
    forecast_df.pivot(index="ItemCode", columns="week_begin", values="yhat")
    .fillna(0)
)

# 6B) Pivot Sales Future on week_begin
pivot_sales_raw = pd.DataFrame()
if not weekly_sales_future.empty:
    pivot_sales_raw = (
        weekly_sales_future.pivot(index="ItemCode", columns="week_begin", values="y")
        .fillna(0)
    )

# 6C) Pivot Works Future on week_begin
pivot_works_raw = pd.DataFrame()
if not weekly_works_future.empty:
    pivot_works_raw = (
        weekly_works_future.pivot(index="ItemCode", columns="week_begin", values="y")
        .fillna(0)
    )

# 6D) Build combined DataFrame step by step
# Start with stock (including ItemDescription if present)
if "ItemDescription" in stock_df.columns:
    report_df = stock_df[["ItemCode", "ItemDescription", "QuantityOnHand"]].copy()
    report_df = report_df.rename(columns={"QuantityOnHand": "CurrentStock"})
else:
    report_df = stock_df[["ItemCode", "QuantityOnHand"]].copy()
    report_df = report_df.rename(columns={"QuantityOnHand": "CurrentStock"})
    report_df["ItemDescription"] = ""  # blank if missing

# Ensure every SKU in forecast is present in stock; if not, add with zero CurrentStock & blank description
for sku in pivot_fcst_raw.index:
    if sku not in report_df["ItemCode"].values:
        report_df = report_df.append(
            {"ItemCode": sku, "CurrentStock": 0, "ItemDescription": ""}, ignore_index=True
        )

report_df = report_df.set_index("ItemCode")

# Get sorted union of all week_begin dates (chronological Mondays)
all_week_begins = set(pivot_fcst_raw.columns)
all_week_begins |= set(pivot_sales_raw.columns) if not pivot_sales_raw.empty else set()
all_week_begins |= set(pivot_works_raw.columns) if not pivot_works_raw.empty else set()
all_week_begins = sorted(all_week_begins)  # chronological order by datetime

forecast_cols = []
actual_cols   = []
planned_cols  = []

# For each week_begin date, insert Forecast, Actual, Planned columns in order
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

# Reset index so ItemCode becomes a column again
report_df = report_df.reset_index()

# 6E) Compute Totals
report_df[f"TotalForecastNext{forecast_weeks}W"] = report_df[forecast_cols].sum(axis=1)
report_df[f"TotalActualNext{forecast_weeks}W"]   = report_df[actual_cols].sum(axis=1)
report_df[f"TotalPlannedNext{forecast_weeks}W"]  = report_df[planned_cols].sum(axis=1)

# 6F) Compute Overdue Orders = max(TotalSoldHistoric − TotalDespatched, 0)
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

# 6G) NetDemand = – ((CurrentStock + TotalPlannedNextNW) − TotalActualNextNW)   # <<< changed
report_df[f"NetDemandNext{forecast_weeks}W"] = -(
    report_df["CurrentStock"]
    + report_df[f"TotalPlannedNext{forecast_weeks}W"]
    - report_df[f"TotalActualNext{forecast_weeks}W"]
)

# 6H) RecommendReorderQty = round up NetDemand to nearest 10                    # <<< changed
def round_up_to_10(x):
    return int(((x + 9) // 10) * 10) if x > 0 else 0

report_df["RecommendReorderQty"] = report_df[f"NetDemandNext{forecast_weeks}W"].apply(round_up_to_10)

# 6I) Final column order—explicitly including both ItemCode and ItemDescription:
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
