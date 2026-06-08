import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_text(value: Any) -> str:
    """Convert text to a comparable lowercase form."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Find a column name using keyword matching."""
    normalized_map = {col: normalize_text(col) for col in df.columns}

    for original_col, normalized_col in normalized_map.items():
        for keyword in keywords:
            if normalize_text(keyword) in normalized_col:
                return original_col
    return None


def convert_to_numeric(series: pd.Series) -> pd.Series:
    """Convert strings like 'Rs. 5,000' or '$1,200' to numeric values."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Detect common business columns in a dataframe."""
    return {
        "date_col": find_column(df, ["date", "order_date", "invoice_date", "time", "month"]),
        "revenue_col": find_column(df, ["revenue", "sales", "amount", "value", "total", "net_sales"]),
        "product_col": find_column(df, ["product", "item", "sku", "name"]),
        "region_col": find_column(df, ["region", "city", "state", "country", "market"]),
        "quantity_col": find_column(df, ["quantity", "qty", "units", "count"]),
    }


def _empty_cleaned_dataframe() -> pd.DataFrame:
    """Create an empty dataframe with standard columns."""
    empty = pd.DataFrame()
    empty["Date"] = pd.Series(dtype="datetime64[ns]")
    empty["Revenue"] = pd.Series(dtype="float64")
    empty["Product"] = pd.Series(dtype="object")
    empty["Region"] = pd.Series(dtype="object")
    empty["Quantity"] = pd.Series(dtype="float64")
    empty["MonthNum"] = pd.Series(dtype="Int64")
    empty["Month"] = pd.Series(dtype="object")
    empty["Quarter"] = pd.Series(dtype="object")
    empty["Week"] = pd.Series(dtype="Int64")
    empty["Day"] = pd.Series(dtype="object")
    return empty


def clean_sales_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean uploaded sales data and create standard columns."""
    schema = detect_columns(df)

    if df is None or df.empty:
        schema["standard_columns"] = ["Date", "Revenue", "Product", "Region", "Quantity"]
        return _empty_cleaned_dataframe(), schema

    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]

    # Date column
    if schema["date_col"] and schema["date_col"] in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(cleaned[schema["date_col"]], errors="coerce")
    else:
        cleaned["Date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(cleaned), freq="D")

    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce").dt.normalize()

    # Revenue column
    if schema["revenue_col"] and schema["revenue_col"] in cleaned.columns:
        cleaned["Revenue"] = convert_to_numeric(cleaned[schema["revenue_col"]]).fillna(0.0)
    else:
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cleaned["Revenue"] = pd.to_numeric(cleaned[numeric_cols[0]], errors="coerce").fillna(0.0)
        else:
            cleaned["Revenue"] = 0.0

    # Optional columns
    if schema["product_col"] and schema["product_col"] in cleaned.columns:
        cleaned["Product"] = cleaned[schema["product_col"]].fillna("Unknown").astype(str)
    else:
        cleaned["Product"] = "Unknown"

    if schema["region_col"] and schema["region_col"] in cleaned.columns:
        cleaned["Region"] = cleaned[schema["region_col"]].fillna("Unknown").astype(str)
    else:
        cleaned["Region"] = "Unknown"

    if schema["quantity_col"] and schema["quantity_col"] in cleaned.columns:
        cleaned["Quantity"] = pd.to_numeric(cleaned[schema["quantity_col"]], errors="coerce").fillna(0)
    else:
        cleaned["Quantity"] = 0

    # Remove invalid dates
    cleaned = cleaned.dropna(subset=["Date"]).reset_index(drop=True)

    if cleaned.empty:
        schema["standard_columns"] = ["Date", "Revenue", "Product", "Region", "Quantity"]
        return _empty_cleaned_dataframe(), schema

    # Useful time columns
    cleaned["MonthNum"] = cleaned["Date"].dt.month
    cleaned["Month"] = cleaned["Date"].dt.strftime("%b")
    cleaned["Quarter"] = "Q" + cleaned["Date"].dt.quarter.astype(str)
    cleaned["Week"] = cleaned["Date"].dt.isocalendar().week.astype(int)
    cleaned["Day"] = cleaned["Date"].dt.day_name()

    schema["standard_columns"] = ["Date", "Revenue", "Product", "Region", "Quantity"]
    return cleaned, schema


def format_currency(value: float) -> str:
    """Format a number as currency-like text."""
    return f"${value:,.0f}"


def _daily_series(df: pd.DataFrame) -> pd.Series:
    """Create a daily revenue series."""
    if df is None or df.empty or "Date" not in df.columns or "Revenue" not in df.columns:
        return pd.Series(dtype=float)

    series = df.groupby("Date")["Revenue"].sum().sort_index()
    if len(series) > 0:
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
        series = series.reindex(full_index, fill_value=0.0)
    return series.astype(float)


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate important business KPIs."""
    if df is None or df.empty:
        return {
            "total_revenue": 0.0,
            "avg_revenue": 0.0,
            "total_records": 0,
            "profit": 0.0,
            "margin_pct": 0.0,
            "unique_products": 0,
            "unique_regions": 0,
            "growth_pct": 0.0,
            "volatility_pct": 0.0,
            "top_product": None,
            "top_product_revenue": 0.0,
            "top_product_share": 0.0,
            "weak_product": None,
            "weak_product_revenue": 0.0,
            "weak_product_share": 0.0,
            "top_region": None,
            "top_region_revenue": 0.0,
            "top_region_share": 0.0,
            "weak_region": None,
            "weak_region_revenue": 0.0,
            "weak_region_share": 0.0,
        }

    revenue = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    total_revenue = float(revenue.sum())
    avg_revenue = float(revenue.mean()) if len(revenue) else 0.0
    total_records = int(len(df))

    profit = float(df["Profit"].sum()) if "Profit" in df.columns else total_revenue * 0.35
    margin = (profit / total_revenue * 100) if total_revenue > 0 else 0.0

    unique_products = int(df["Product"].nunique()) if "Product" in df.columns else 0
    unique_regions = int(df["Region"].nunique()) if "Region" in df.columns else 0

    daily = _daily_series(df)
    if len(daily) >= 60:
        recent = daily.tail(30).mean()
        previous = daily.iloc[-60:-30].mean()
    elif len(daily) >= 10:
        split = len(daily) // 2
        recent = daily.iloc[split:].mean()
        previous = daily.iloc[:split].mean()
    else:
        recent = daily.mean() if len(daily) else 0.0
        previous = daily.mean() if len(daily) else 0.0

    if previous > 0:
        growth_pct = float(((recent - previous) / previous) * 100)
    elif recent > 0:
        growth_pct = 100.0
    else:
        growth_pct = 0.0

    volatility_pct = (revenue.std() / revenue.mean() * 100) if revenue.mean() > 0 else 0.0

    top_product = None
    top_product_revenue = 0.0
    weak_product = None
    weak_product_revenue = 0.0
    top_product_share = 0.0
    weak_product_share = 0.0

    if "Product" in df.columns:
        product_group = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
        if len(product_group) > 0:
            top_product = str(product_group.index[0])
            top_product_revenue = float(product_group.iloc[0])
            weak_product = str(product_group.index[-1])
            weak_product_revenue = float(product_group.iloc[-1])
            if total_revenue > 0:
                top_product_share = top_product_revenue / total_revenue * 100
                weak_product_share = weak_product_revenue / total_revenue * 100

    top_region = None
    top_region_revenue = 0.0
    weak_region = None
    weak_region_revenue = 0.0
    top_region_share = 0.0
    weak_region_share = 0.0

    if "Region" in df.columns:
        region_group = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
        if len(region_group) > 0:
            top_region = str(region_group.index[0])
            top_region_revenue = float(region_group.iloc[0])
            weak_region = str(region_group.index[-1])
            weak_region_revenue = float(region_group.iloc[-1])
            if total_revenue > 0:
                top_region_share = top_region_revenue / total_revenue * 100
                weak_region_share = weak_region_revenue / total_revenue * 100

    return {
        "total_revenue": total_revenue,
        "avg_revenue": avg_revenue,
        "total_records": total_records,
        "profit": profit,
        "margin_pct": margin,
        "unique_products": unique_products,
        "unique_regions": unique_regions,
        "growth_pct": growth_pct,
        "volatility_pct": volatility_pct,
        "top_product": top_product,
        "top_product_revenue": top_product_revenue,
        "top_product_share": top_product_share,
        "weak_product": weak_product,
        "weak_product_revenue": weak_product_revenue,
        "weak_product_share": weak_product_share,
        "top_region": top_region,
        "top_region_revenue": top_region_revenue,
        "top_region_share": top_region_share,
        "weak_region": weak_region,
        "weak_region_revenue": weak_region_revenue,
        "weak_region_share": weak_region_share,
    }


def detect_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect unusual revenue spikes or drops."""
    daily = _daily_series(df)
    if len(daily) < 10:
        return []

    q1 = daily.quantile(0.25)
    q3 = daily.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    anomalies = []
    flagged = daily[(daily < lower) | (daily > upper)]

    for idx, value in flagged.items():
        anomaly_type = "high" if value > upper else "low"
        anomalies.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "value": float(value),
                "type": anomaly_type,
            }
        )

    return anomalies[:5]


def generate_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate business-friendly insights."""
    if df is None or df.empty:
        return [
            {
                "type": "info",
                "title": "No usable data found",
                "detail": "Please upload a valid sales file to generate insights.",
                "action": "Use a CSV or Excel file with sales records.",
            }
        ]

    kpis = calculate_kpis(df)
    insights: List[Dict[str, Any]] = []

    # Revenue trend
    if kpis["growth_pct"] >= 0:
        insights.append(
            {
                "type": "success",
                "title": "Revenue trend is positive",
                "detail": f"Revenue growth is {kpis['growth_pct']:.1f}% versus the previous period.",
                "action": "Keep the current strategy and scale the best-performing channels.",
            }
        )
    else:
        insights.append(
            {
                "type": "warning",
                "title": "Revenue trend needs attention",
                "detail": f"Revenue dropped by {abs(kpis['growth_pct']):.1f}% versus the previous period.",
                "action": "Review pricing, promotions, and weak regions or products.",
            }
        )

    # Best and weak product
    if kpis["top_product"]:
        insights.append(
            {
                "type": "success",
                "title": f"Top product: {kpis['top_product']}",
                "detail": (
                    f"Total revenue: {format_currency(kpis['top_product_revenue'])} "
                    f"({kpis['top_product_share']:.1f}% of total)."
                ),
                "action": f"Increase inventory and marketing for {kpis['top_product']}.",
            }
        )

    if kpis["weak_product"]:
        insights.append(
            {
                "type": "warning",
                "title": f"Underperforming product: {kpis['weak_product']}",
                "detail": (
                    f"Total revenue: {format_currency(kpis['weak_product_revenue'])} "
                    f"({kpis['weak_product_share']:.1f}% of total)."
                ),
                "action": f"Review pricing or reposition {kpis['weak_product']}.",
            }
        )

    # Best and weak region
    if kpis["top_region"]:
        insights.append(
            {
                "type": "success",
                "title": f"Best region: {kpis['top_region']}",
                "detail": (
                    f"Revenue contribution: {format_currency(kpis['top_region_revenue'])} "
                    f"({kpis['top_region_share']:.1f}% of total)."
                ),
                "action": f"Push more campaigns in {kpis['top_region']}.",
            }
        )

    if kpis["weak_region"]:
        insights.append(
            {
                "type": "warning",
                "title": f"Region needing attention: {kpis['weak_region']}",
                "detail": (
                    f"Revenue contribution: {format_currency(kpis['weak_region_revenue'])} "
                    f"({kpis['weak_region_share']:.1f}% of total)."
                ),
                "action": f"Investigate demand and channel issues in {kpis['weak_region']}.",
            }
        )

    # Consistency
    if kpis["volatility_pct"] > 40:
        insights.append(
            {
                "type": "warning",
                "title": "Sales are volatile",
                "detail": f"Revenue volatility is {kpis['volatility_pct']:.0f}%.",
                "action": "Focus on repeat customers and consistent campaigns.",
            }
        )
    else:
        insights.append(
            {
                "type": "success",
                "title": "Sales are stable",
                "detail": f"Revenue volatility is {kpis['volatility_pct']:.0f}%.",
                "action": "Keep the current strategy and monitor seasonal patterns.",
            }
        )

    # Anomalies
    anomalies = detect_anomalies(df)
    if anomalies:
        insights.append(
            {
                "type": "info",
                "title": "Anomalies detected",
                "detail": f"{len(anomalies)} unusual revenue days were found.",
                "action": "Check spikes or drops for campaign, stock, or pricing issues.",
            }
        )

    return insights


def generate_executive_summary(
    df: pd.DataFrame,
    forecast_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a short executive summary for the demo."""
    if df is None or df.empty:
        return {
            "headline": "No usable data was found.",
            "summary_text": "Upload a sales file to generate a business summary.",
            "bullets": ["No records available."],
            "recommendations": ["Upload a valid CSV or Excel file."],
        }

    kpis = calculate_kpis(df)

    if kpis["growth_pct"] >= 0:
        headline = "Revenue is trending upward."
    else:
        headline = "Revenue is under pressure."

    bullets = [
        f"Total revenue: {format_currency(kpis['total_revenue'])}.",
        f"Average order value: {format_currency(kpis['avg_revenue'])}.",
        f"Growth versus previous period: {kpis['growth_pct']:.1f}%.",
    ]

    if kpis["top_product"]:
        bullets.append(
            f"Top product: {kpis['top_product']} with {format_currency(kpis['top_product_revenue'])} revenue."
        )

    if kpis["top_region"]:
        bullets.append(
            f"Top region: {kpis['top_region']} with {format_currency(kpis['top_region_revenue'])} revenue."
        )

    if forecast_payload and forecast_payload.get("best_model"):
        bullets.append(
            f"Best forecast model: {forecast_payload['best_model']}. "
            f"Expected sales over the selected horizon: {format_currency(forecast_payload['forecast_sum'])}."
        )

    recommendations = []

    if kpis["growth_pct"] < 0:
        recommendations.append("Launch a sales push for weak products and regions.")
    if kpis["volatility_pct"] > 40:
        recommendations.append("Stabilize revenue with repeat-customer and retention campaigns.")
    if kpis["top_product"]:
        recommendations.append(f"Promote {kpis['top_product']} and keep inventory ready.")
    if kpis["top_region"]:
        recommendations.append(f"Invest more in {kpis['top_region']} since it is the strongest region.")

    anomalies = detect_anomalies(df)
    if anomalies:
        recommendations.append("Investigate unusual spikes and drops before they become bigger issues.")

    if forecast_payload and forecast_payload.get("forecast_sum", 0) > 0:
        recommendations.append("Use the forecast to plan inventory and campaign timing.")

    if not recommendations:
        recommendations.append("Keep monitoring data and continue the current strategy.")

    return {
        "headline": headline,
        "summary_text": " ".join(bullets),
        "bullets": bullets,
        "recommendations": recommendations[:5],
    }