import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Marketplace Price Intelligence", layout="wide")

st.title("📊 Marketplace Price Intelligence Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("marketplace_cleaned copy.csv")
    df["title"] = df["title"].astype(str)
    df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")
    df = df.dropna(subset=["price_clean"])
    return df

df = load_data()

# ---------------- DEBUG (IMPORTANT)
st.write("✅ Rows Loaded:", df.shape[0])

if df.empty:
    st.error("❌ No data found. Check CSV file.")
    st.stop()

# -------------------------------------------------
# SIDEBAR FILTER (SAFE)
# -------------------------------------------------
st.sidebar.header("🔍 Filter")

titles = ["All"] + sorted(df["title"].unique())
selected_title = st.sidebar.selectbox("Select Item", titles)

filtered_df = df if selected_title == "All" else df[df["title"] == selected_title]

st.write("📌 Filtered Rows:", filtered_df.shape[0])

if filtered_df.empty:
    st.warning("No records for selected filter.")
    st.stop()

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
st.subheader("📄 Data Preview")

preview = filtered_df.copy()
preview["url"] = preview["url"].apply(lambda x: f"[Open Link]({x})")

st.dataframe(preview.head(10), use_container_width=True)

# -------------------------------------------------
# PRICE INTELLIGENCE
# -------------------------------------------------
st.header("💰 Price Intelligence")

avg_price = filtered_df["price_clean"].mean()
min_price = filtered_df["price_clean"].min()
max_price = filtered_df["price_clean"].max()

c1, c2, c3 = st.columns(3)
c1.metric("Average Price", f"₹ {avg_price:.2f}")
c2.metric("Minimum Price", f"₹ {min_price:.2f}")
c3.metric("Maximum Price", f"₹ {max_price:.2f}")

# -------------------------------------------------
# INTERACTIVE HISTOGRAM
# -------------------------------------------------
st.subheader("📊 Price Distribution")

fig_hist = px.histogram(
    filtered_df,
    x="price_clean",
    nbins=30,
    title="Price Distribution"
)
st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------
# PRICE DEVIATION
# -------------------------------------------------
st.subheader("🚨 Price Deviations")

filtered_df["z_score"] = zscore(filtered_df["price_clean"])
deviations = filtered_df[abs(filtered_df["z_score"]) > 2]

st.dataframe(
    deviations[["title", "price_clean", "z_score", "url"]].head(10),
    use_container_width=True
)

# -------------------------------------------------
# DEAL SCORING ⭐
# -------------------------------------------------
st.header("⭐ Deal Scoring")

avg_item_price = (
    filtered_df.groupby("title")["price_clean"]
    .mean()
    .reset_index(name="avg_item_price")
)

scored = filtered_df.merge(avg_item_price, on="title")
scored["deal_score"] = scored["avg_item_price"] - scored["price_clean"]

# Best Deals
st.subheader("🔥 Best Deals")
st.dataframe(
    scored.sort_values("deal_score", ascending=False)
    [["title", "price_clean", "deal_score", "url"]]
    .head(10),
    use_container_width=True
)

# -------------------------------------------------
# ML PRICE PREDICTION 🤖
# -------------------------------------------------
st.header("🤖 ML Price Prediction")

X = scored[["avg_item_price"]]
y = scored["price_clean"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

scored["predicted_price"] = model.predict(X)
scored["price_difference"] = scored["predicted_price"] - scored["price_clean"]

def recommend(diff):
    if diff > 50:
        return "✅ Buy"
    elif diff > -50:
        return "⏳ Wait"
    else:
        return "❌ Avoid"

scored["recommendation"] = scored["price_difference"].apply(recommend)

# -------------------------------------------------
# INTERACTIVE ML SCATTER
# -------------------------------------------------
st.subheader("📈 Actual vs Predicted Price")

fig_pred = px.scatter(
    scored,
    x="price_clean",
    y="predicted_price",
    hover_data=["title"],
    title="Actual vs Predicted Price"
)
st.plotly_chart(fig_pred, use_container_width=True)

# -------------------------------------------------
# FINAL TABLE
# -------------------------------------------------
st.subheader("📌 ML Recommendations")

st.dataframe(
    scored[[
        "title",
        "price_clean",
        "predicted_price",
        "price_difference",
        "recommendation",
        "url"
    ]].head(10),
    use_container_width=True
)

st.success("✅ Dashboard Loaded Successfully")
