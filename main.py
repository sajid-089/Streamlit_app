"""FastAPI backend for upload, analysis, and forecasting."""

from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics import (
    calculate_kpis,
    clean_sales_dataframe,
    generate_executive_summary,
    generate_insights,
)
from auth import login, signup
from ml_pipeline import compare_forecasting_models, prepare_daily_series

app = FastAPI(title="InsightFlow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupPayload(BaseModel):
    """Signup payload."""

    email: str
    password: str
    role: str = "viewer"


class LoginPayload(BaseModel):
    """Login payload."""

    email: str
    password: str


def _read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Read a CSV or Excel file into a dataframe."""
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file.file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file.file)

    raise HTTPException(status_code=400, detail="Only CSV and Excel files are allowed.")


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {"status": "ok", "message": "Welcome to InsightFlow API."}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "message": "InsightFlow API is running."}


@app.post("/auth/signup")
def api_signup(payload: SignupPayload) -> Dict[str, Any]:
    """Create a new account."""
    result = signup(payload.email, payload.password, payload.role)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/auth/login")
def api_login(payload: LoginPayload) -> Dict[str, Any]:
    """Login endpoint."""
    result = login(payload.email, payload.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...), horizon: int = 30) -> Dict[str, Any]:
    """Analyze uploaded sales data and return insights plus forecasts."""
    try:
        raw_df = _read_uploaded_file(file)
        if raw_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        cleaned_df, schema = clean_sales_dataframe(raw_df)
        if cleaned_df.empty:
            raise HTTPException(status_code=400, detail="No usable rows found after cleaning.")

        kpis = calculate_kpis(cleaned_df)
        insights = generate_insights(cleaned_df)
        summary = generate_executive_summary(cleaned_df)

        if cleaned_df["Date"].notna().sum() < 5:
            forecast_payload = {
                "best_model": "Not enough data",
                "comparison": [],
                "forecast": [],
                "forecast_sum": 0.0,
                "forecast_average": 0.0,
                "artifact_path": None,
                "message": "Not enough dates for forecasting.",
            }
        else:
            daily_series = prepare_daily_series(cleaned_df, "Date", "Revenue")
            forecast_payload = compare_forecasting_models(daily_series, horizon=horizon)

            # Rebuild the summary with forecast context
            summary = generate_executive_summary(cleaned_df, forecast_payload)

        return {
            "schema": schema,
            "kpis": kpis,
            "summary": summary,
            "insights": insights,
            "forecast": forecast_payload,
            "sample_rows": cleaned_df.head(10).to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/forecast/upload")
async def forecast_only(file: UploadFile = File(...), horizon: int = 30) -> Dict[str, Any]:
    """Return forecast result only."""
    try:
        raw_df = _read_uploaded_file(file)
        cleaned_df, _ = clean_sales_dataframe(raw_df)

        if cleaned_df.empty:
            raise HTTPException(status_code=400, detail="No usable rows found after cleaning.")

        daily_series = prepare_daily_series(cleaned_df, "Date", "Revenue")
        if len(daily_series) < 5:
            raise HTTPException(status_code=400, detail="Not enough data for forecasting.")

        return compare_forecasting_models(daily_series, horizon=horizon)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))