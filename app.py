# ======================================
# File: app.py
# ======================================

import streamlit as st
import pandas as pd
import io
import math
from datetime import datetime, timedelta
from utils.forecasting import batch_seasonal_naive_forecast

# ─────────────────────────────────────────
# (1) SET PAGE CONFIG FIRST
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Solidus Demand Forecast",
    layout="wide"
)

# ─────────────────────────────────────────
# (2) OPTIONAL: hide Streamlit menu/footer
# ─────────────────────────────────────────
hide_streamlit_style = """
    <style>
    /* Hide default Streamlit menu */
    #MainMenu {visibility: hidden;}
    /* Hide “Made with Streamlit” footer */
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ─────────────────────────────────────────
# (3) DISPLAY YOUR LOGO NEXT
# ─────────────────────────────────────────
# Make sure 'assets/solidus_logo.png' actually exists
from PIL import Image
logo_path = "assets/solidus_logo.png"
try:
    Image.open(logo_path).verify()
    st.image(logo_path, width=250)
except Exception:
    st.error(f"❌ Could not load logo at '{logo_path}'. Check that the file exists and is a valid PNG.")
    st.stop()

# ─────────────────────────────────────────
# (4) NOW YOUR TITLE / REST OF APP
# ─────────────────────────────────────────
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
    4. Use a **seasonal‐naive** approach (average of the same ISO week in prior years on  
       despatches) to forecast N weeks, using **week‐commencing** (Monday) dates.  
    5. Compute **Overdue Orders** = max(HistoricSales − HistoricDespatched, 0).  
    6. Build a **Demand Report** with columns:  
       - ItemCode (SKU)  
       - ItemDescription (SKU description)  
       - CurrentStock  
       - OverdueOrders  
       - TotalForecastNextNW  
       - TotalActualNextNW  
       - TotalPlannedNextNW  
       - NetDemand = (CurrentStock + TotalPlannedNextNW) − TotalActualNextNW  
       - RecommendReorderQty = round up(NetDemand) to next multiple of 10 (if NetDemand > 0)  
       - Followed by interleaved weekly columns: Forecast/Actual/Planned for each week commencing.  
    7. Provide interactive charts showing weekly series (Historic Despatched, Forecast, Actual, Planned)  
       and a bar chart of CurrentStock vs. Forecast vs. Actual vs. Planned vs. Overdue vs. Net.  
    8. Allow CSV export of the full Demand Report.
    """
)

# …continue with your existing code (file uploads, data preparation, forecasting, etc.)…
