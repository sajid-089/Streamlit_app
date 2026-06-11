"""Business copilot for Auralytix.

This assistant answers sales and analytics questions using the
current dataframe, KPI snapshot, summary, and forecast data.
"""

from typing import Dict, List, Optional
import pandas as pd


def _safe_get(kpis: dict, *keys, default=None):
    """Return the first available value from a KPI dictionary."""
    for key in keys:
        if key in kpis:
            return kpis[key]
    return default


def _currency(value: float) -> str:
    """Format currency in a business-friendly style."""
    try:
        return f"$ {float(value):,.0f}"
    except Exception:
        return "Rs. 0"


def build_chat_context(
    df: pd.DataFrame,
    kpis: dict,
    summary: dict,
    insights: List[dict],
    forecast_result: Optional[dict] = None,
) -> dict:
    """Build a compact context object for the assistant."""
    top_product = _safe_get(kpis, "top_product", default="Unknown")
    top_region = _safe_get(kpis, "top_region", default="Unknown")
    weak_product = _safe_get(kpis, "weak_product", default="Unknown")
    weak_region = _safe_get(kpis, "weak_region", default="Unknown")
    growth = _safe_get(kpis, "growth", "growth_pct", default=0.0)
    volatility = _safe_get(kpis, "volatility", "volatility_pct", default=0.0)
    total_revenue = _safe_get(kpis, "total_revenue", default=0.0)
    margin = _safe_get(kpis, "margin", "margin_pct", default=0.0)
    forecast_model = forecast_result.get("best_model") if forecast_result else None
    forecast_sum = forecast_result.get("forecast_sum") if forecast_result else None

    return {
        "rows": len(df),
        "total_revenue": total_revenue,
        "margin": margin,
        "growth": growth,
        "volatility": volatility,
        "top_product": top_product,
        "top_region": top_region,
        "weak_product": weak_product,
        "weak_region": weak_region,
        "forecast_model": forecast_model,
        "forecast_sum": forecast_sum,
        "summary_headline": summary.get("headline", ""),
        "summary_text": summary.get("summary_text", ""),
        "recommendations": summary.get("recommendations", []),
        "insights": insights,
    }


def _match_question(question: str) -> str:
    """Classify the user's question into a business intent."""
    q = question.lower().strip()

    if any(word in q for word in ["top product", "best product", "product best", "highest product"]):
        return "top_product"
    if any(word in q for word in ["weak product", "bad product", "underperforming product"]):
        return "weak_product"
    if any(word in q for word in ["top region", "best region", "which region", "region best"]):
        return "top_region"
    if any(word in q for word in ["weak region", "low region", "region weak"]):
        return "weak_region"
    if any(word in q for word in ["growth", "trend", "revenue up", "revenue down"]):
        return "growth"
    if any(word in q for word in ["margin", "profit", "profit margin"]):
        return "margin"
    if any(word in q for word in ["forecast", "prediction", "next month", "future sales"]):
        return "forecast"
    if any(word in q for word in ["risk", "volatile", "stability", "stable"]):
        return "risk"
    if any(word in q for word in ["summary", "overview", "business summary"]):
        return "summary"
    if any(word in q for word in ["action", "recommend", "next step", "what should i do"]):
        return "actions"

    return "general"


def answer_question(question: str, context: dict) -> str:
    """Return a business-friendly answer based on the current dashboard context."""
    intent = _match_question(question)

    total_revenue = _currency(context["total_revenue"])
    margin = f"{context['margin']:.1f}%"
    growth = f"{context['growth']:.1f}%"
    volatility = f"{context['volatility']:.0f}%"
    forecast_sum = _currency(context["forecast_sum"]) if context["forecast_sum"] is not None else None

    if intent == "top_product":
        return (
            f"Top performing product is **{context['top_product']}**. "
            f"Total revenue contribution is strong, so this product should be prioritized in inventory and marketing."
        )

    if intent == "weak_product":
        return (
            f"Underperforming product is **{context['weak_product']}**. "
            f"It needs review on pricing, positioning, or demand-side issues."
        )

    if intent == "top_region":
        return (
            f"Best region is **{context['top_region']}**. "
            f"This region is currently contributing the most to revenue, so it is a good candidate for more campaigns."
        )

    if intent == "weak_region":
        return (
            f"Weak region is **{context['weak_region']}**. "
            f"That area should be checked for lower demand, weak distribution, or campaign gaps."
        )

    if intent == "growth":
        if context["growth"] >= 0:
            return (
                f"Revenue is trending upward with **{growth}** growth versus the previous period. "
                f"Current total revenue is {total_revenue}."
            )
        return (
            f"Revenue is under pressure with **{growth}** change versus the previous period. "
            f"You should review campaigns, pricing, and weak segments."
        )

    if intent == "margin":
        return (
            f"Profit margin is **{margin}**. "
            f"If you want better margin, focus on stronger products and optimize weak segments."
        )

    if intent == "forecast":
        if context["forecast_model"] and forecast_sum:
            return (
                f"Forecasting model selected is **{context['forecast_model']}**. "
                f"Expected revenue over the selected horizon is **{forecast_sum}**. "
                f"Use this for stock and campaign planning."
            )
        return "Forecast is not available yet. Please generate forecast first."

    if intent == "risk":
        if context["volatility"] > 40 or context["growth"] < 0:
            return (
                f"Business risk looks **elevated**. Volatility is {volatility} and growth is {growth}. "
                f"Sales are not fully stable right now."
            )
        return (
            f"Business risk looks **controlled**. Volatility is {volatility} and growth is {growth}. "
            f"Current performance seems reasonably stable."
        )

    if intent == "summary":
        return (
            f"{context['summary_headline']} "
            f"{context['summary_text']}"
        )

    if intent == "actions":
        recs = context["recommendations"][:3] if context["recommendations"] else []
        if recs:
            return "Recommended actions:\n- " + "\n- ".join(recs)
        return "No clear action found. Current performance seems stable."

    # General fallback
    return (
        f"Here is the current business snapshot:\n"
        f"- Total revenue: {total_revenue}\n"
        f"- Margin: {margin}\n"
        f"- Growth: {growth}\n"
        f"- Volatility: {volatility}\n"
        f"- Top product: {context['top_product']}\n"
        f"- Top region: {context['top_region']}\n\n"
        f"You can ask me things like:\n"
        f"- Which product is performing best?\n"
        f"- Which region needs attention?\n"
        f"- What should I do next?\n"
        f"- Give me a short summary."
    )


def suggest_followups() -> List[str]:
    """Return suggested follow-up questions for the chat UI."""
    return [
        "Which product is performing best?",
        "Which region needs attention?",
        "Give me a short business summary.",
        "What should I do next?",
        "Show me the forecast outlook.",
    ]