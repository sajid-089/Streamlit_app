"""Forecasting pipeline with statistical and ML models."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def prepare_daily_series(
    df: pd.DataFrame,
    date_col: str = "Date",
    value_col: str = "Revenue",
) -> pd.Series:
    """Convert a dataframe into a complete daily time series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce").dt.normalize()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce").fillna(0.0)
    temp = temp.dropna(subset=[date_col])

    if temp.empty:
        return pd.Series(dtype=float)

    series = temp.groupby(date_col)[value_col].sum().sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index, fill_value=0.0)
    series.index.name = "Date"
    return series.astype(float)


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE with zero protection."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.clip(np.abs(actual), 1e-6, None)
    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100)


def _evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE, and MAPE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = _safe_mape(actual, predicted)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def _moving_average_forecast(train_series: pd.Series, steps: int, window: int = 7) -> Dict[str, Any]:
    """A safe fallback forecast."""
    if train_series is None or len(train_series) == 0:
        forecast = np.zeros(steps, dtype=float)
        return {
            "model": None,
            "forecast": forecast,
            "lower": forecast,
            "upper": forecast,
            "residual_std": 0.0,
        }

    window = min(window, max(1, len(train_series)))
    last_avg = float(train_series.tail(window).mean())
    forecast = np.array([last_avg] * steps, dtype=float)

    rolling_mean = train_series.rolling(window, min_periods=1).mean()
    resid_std = float(np.nanstd((train_series - rolling_mean).fillna(0.0).values))
    if np.isnan(resid_std):
        resid_std = 0.0

    lower = np.clip(forecast - 1.96 * resid_std, 0, None)
    upper = np.clip(forecast + 1.96 * resid_std, 0, None)

    return {
        "model": None,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "residual_std": resid_std,
    }


def _fit_best_sarimax(train_series: pd.Series) -> Dict[str, Any]:
    """Fit a small SARIMAX grid and return the best model."""
    candidate_orders = [(1, 1, 0), (1, 1, 1), (2, 1, 1), (0, 1, 1), (2, 0, 1)]
    if len(train_series) >= 28:
        seasonal_orders = [(1, 1, 1, 7), (1, 1, 0, 7), (0, 1, 1, 7), (0, 0, 0, 0)]
    else:
        seasonal_orders = [(0, 0, 0, 0)]

    best_fit = None
    best_order = None
    best_seasonal_order = None
    best_aic = float("inf")

    for order in candidate_orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_fit = fitted
                    best_order = order
                    best_seasonal_order = seasonal_order
            except Exception:
                continue

    if best_fit is None:
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        best_fit = model.fit(disp=False)
        best_order = (1, 1, 1)
        best_seasonal_order = (0, 0, 0, 0)

    return {
        "model": best_fit,
        "order": best_order,
        "seasonal_order": best_seasonal_order,
    }


def forecast_sarimax(train_series: pd.Series, steps: int) -> Dict[str, Any]:
    """Forecast with SARIMAX and confidence intervals."""
    fitted_payload = _fit_best_sarimax(train_series)
    fitted = fitted_payload["model"]
    forecast = fitted.forecast(steps=steps)
    conf = fitted.get_forecast(steps=steps).conf_int()

    return {
        "model": fitted,
        "order": fitted_payload["order"],
        "seasonal_order": fitted_payload["seasonal_order"],
        "forecast": np.asarray(forecast, dtype=float),
        "lower": np.asarray(conf.iloc[:, 0], dtype=float),
        "upper": np.asarray(conf.iloc[:, 1], dtype=float),
    }


def forecast_sarimax_with_config(
    series: pd.Series,
    steps: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """Forecast with SARIMAX using a fixed configuration."""
    fitted = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast = fitted.forecast(steps=steps)
    conf = fitted.get_forecast(steps=steps).conf_int()

    return {
        "model": fitted,
        "order": order,
        "seasonal_order": seasonal_order,
        "forecast": np.asarray(forecast, dtype=float),
        "lower": np.asarray(conf.iloc[:, 0], dtype=float),
        "upper": np.asarray(conf.iloc[:, 1], dtype=float),
    }


def forecast_ets(train_series: pd.Series, steps: int) -> Dict[str, Any]:
    """Forecast with Exponential Smoothing."""
    if len(train_series) >= 28:
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
    else:
        model = ExponentialSmoothing(
            train_series,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )

    fitted = model.fit(optimized=True)
    forecast = np.asarray(fitted.forecast(steps), dtype=float)

    resid = getattr(fitted, "resid", np.array([0.0]))
    resid_std = float(np.nanstd(resid)) if len(resid) else 0.0
    if np.isnan(resid_std):
        resid_std = 0.0

    lower = np.clip(forecast - 1.96 * resid_std, 0, None)
    upper = np.clip(forecast + 1.96 * resid_std, 0, None)

    return {
        "model": fitted,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "residual_std": resid_std,
    }


def _make_supervised(series: pd.Series, n_lags: int = 21) -> pd.DataFrame:
    """Create lag and rolling features for ML forecasting."""
    df = pd.DataFrame({"y": series.astype(float)})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["roll_mean_7"] = df["y"].shift(1).rolling(7).mean()
    df["roll_mean_14"] = df["y"].shift(1).rolling(14).mean()
    df["roll_std_7"] = df["y"].shift(1).rolling(7).std()
    df["roll_std_14"] = df["y"].shift(1).rolling(14).std()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day"] = df.index.day

    return df.dropna()


def _future_feature_row(
    history: List[float],
    future_date: pd.Timestamp,
    n_lags: int = 21,
) -> pd.DataFrame:
    """Create one feature row for the next forecast step."""
    minimum_history = n_lags + 14
    if len(history) < minimum_history:
        pad_value = history[0] if history else 0.0
        history = [pad_value] * (minimum_history - len(history)) + history

    row = {}
    for lag in range(1, n_lags + 1):
        row[f"lag_{lag}"] = float(history[-lag])

    last_7 = history[-7:]
    last_14 = history[-14:]

    row["roll_mean_7"] = float(np.mean(last_7))
    row["roll_mean_14"] = float(np.mean(last_14))
    row["roll_std_7"] = float(np.std(last_7))
    row["roll_std_14"] = float(np.std(last_14))
    row["dayofweek"] = int(future_date.dayofweek)
    row["month"] = int(future_date.month)
    row["day"] = int(future_date.day)

    return pd.DataFrame([row])


def _fit_recursive_tree_forecast(
    train_series: pd.Series,
    steps: int,
    estimator: Any,
    n_lags: int = 21,
) -> Dict[str, Any]:
    """Fit a tree model and forecast recursively."""
    supervised = _make_supervised(train_series, n_lags=n_lags)
    if supervised.empty:
        raise ValueError("Not enough data for tree-based forecasting.")

    X = supervised.drop(columns=["y"])
    y = supervised["y"]

    estimator.fit(X, y)
    train_pred = estimator.predict(X)
    resid_std = float(np.std(y.values - train_pred)) if len(y) > 1 else float(np.std(y.values))
    if np.isnan(resid_std):
        resid_std = 0.0

    history = list(train_series.values)
    preds = []

    for step in range(steps):
        future_date = train_series.index[-1] + pd.Timedelta(days=step + 1)
        next_row = _future_feature_row(history, future_date, n_lags=n_lags)
        next_row = next_row[X.columns]
        pred = float(estimator.predict(next_row)[0])
        pred = max(0.0, pred)
        preds.append(pred)
        history.append(pred)

    preds = np.asarray(preds, dtype=float)
    lower = np.clip(preds - 1.96 * resid_std, 0, None)
    upper = np.clip(preds + 1.96 * resid_std, 0, None)

    return {
        "model": estimator,
        "forecast": preds,
        "lower": lower,
        "upper": upper,
        "residual_std": resid_std,
        "feature_columns": list(X.columns),
        "history_tail": history[-120:],
        "n_lags": n_lags,
    }


def forecast_random_forest(train_series: pd.Series, steps: int, n_lags: int = 21) -> Dict[str, Any]:
    """Forecast with a Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    return _fit_recursive_tree_forecast(train_series, steps, model, n_lags=n_lags)


def forecast_gradient_boosting(train_series: pd.Series, steps: int, n_lags: int = 21) -> Dict[str, Any]:
    """Forecast with a Gradient Boosting model."""
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    return _fit_recursive_tree_forecast(train_series, steps, model, n_lags=n_lags)


def _save_bundle(bundle: Dict[str, Any], artifact_dir: str) -> str:
    """Save a model bundle to disk."""
    os.makedirs(artifact_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(artifact_dir, f"forecast_bundle_{timestamp}.joblib")
    joblib.dump(bundle, path)

    meta = {
        "saved_at": datetime.utcnow().isoformat(),
        "best_model": bundle.get("best_model"),
        "horizon": bundle.get("horizon"),
        "training_end": bundle.get("training_end"),
    }

    meta_path = os.path.join(artifact_dir, f"forecast_bundle_{timestamp}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return path


def compare_forecasting_models(
    series: pd.Series,
    horizon: int = 30,
    artifact_dir: str = "artifacts",
) -> Dict[str, Any]:
    """Compare multiple forecasting models and return the best one."""
    if series is None:
        return {
            "best_model": "None",
            "comparison": [],
            "forecast": [],
            "forecast_sum": 0.0,
            "forecast_average": 0.0,
            "artifact_path": None,
            "message": "No data available for forecasting.",
        }

    series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(series) == 0:
        return {
            "best_model": "None",
            "comparison": [],
            "forecast": [],
            "forecast_sum": 0.0,
            "forecast_average": 0.0,
            "artifact_path": None,
            "message": "No data available for forecasting.",
        }

    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    horizon = int(max(3, min(horizon, max(3, len(series) // 4))))

    if len(series) < 20:
        fallback = _moving_average_forecast(series, steps=horizon)
        future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        forecast_records = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "forecast": float(v),
                "lower": float(l),
                "upper": float(u),
            }
            for d, v, l, u in zip(future_dates, fallback["forecast"], fallback["lower"], fallback["upper"])
        ]
        return {
            "best_model": "Moving Average",
            "comparison": [],
            "forecast": forecast_records,
            "forecast_sum": float(np.sum(fallback["forecast"])),
            "forecast_average": float(np.mean(fallback["forecast"])),
            "artifact_path": None,
            "message": "Not enough data. Used moving average fallback.",
        }

    train_size = len(series) - horizon
    if train_size < 10:
        horizon = max(3, len(series) // 5)
        train_size = len(series) - horizon

    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    comparison: List[Dict[str, Any]] = []
    best_candidate = None

    # SARIMAX
    try:
        sarimax_result = forecast_sarimax(train, steps=horizon)
        sarimax_metrics = _evaluate_forecast(test.values, sarimax_result["forecast"])
        comparison.append(
            {
                "model": "SARIMAX",
                **sarimax_metrics,
                "extra": f"order={sarimax_result['order']}, seasonal={sarimax_result['seasonal_order']}",
            }
        )
        if best_candidate is None or sarimax_metrics["mape"] < best_candidate["metrics"]["mape"]:
            best_candidate = {
                "name": "SARIMAX",
                "metrics": sarimax_metrics,
                "config": {
                    "order": sarimax_result["order"],
                    "seasonal_order": sarimax_result["seasonal_order"],
                },
            }
    except Exception:
        pass

    # ETS
    try:
        ets_result = forecast_ets(train, steps=horizon)
        ets_metrics = _evaluate_forecast(test.values, ets_result["forecast"])
        comparison.append(
            {
                "model": "Exponential Smoothing",
                **ets_metrics,
                "extra": "seasonal=auto",
            }
        )
        if best_candidate is None or ets_metrics["mape"] < best_candidate["metrics"]["mape"]:
            best_candidate = {
                "name": "Exponential Smoothing",
                "metrics": ets_metrics,
                "config": {},
            }
    except Exception:
        pass

    # Random Forest
    try:
        n_lags = min(21, max(7, len(train) // 3))
        rf_result = forecast_random_forest(train, steps=horizon, n_lags=n_lags)
        rf_metrics = _evaluate_forecast(test.values, rf_result["forecast"])
        comparison.append(
            {
                "model": "Random Forest",
                **rf_metrics,
                "extra": f"lags={n_lags}",
            }
        )
        if best_candidate is None or rf_metrics["mape"] < best_candidate["metrics"]["mape"]:
            best_candidate = {
                "name": "Random Forest",
                "metrics": rf_metrics,
                "config": {"n_lags": n_lags},
            }
    except Exception:
        pass

    # Gradient Boosting
    try:
        n_lags = min(21, max(7, len(train) // 3))
        gb_result = forecast_gradient_boosting(train, steps=horizon, n_lags=n_lags)
        gb_metrics = _evaluate_forecast(test.values, gb_result["forecast"])
        comparison.append(
            {
                "model": "Gradient Boosting",
                **gb_metrics,
                "extra": f"lags={n_lags}",
            }
        )
        if best_candidate is None or gb_metrics["mape"] < best_candidate["metrics"]["mape"]:
            best_candidate = {
                "name": "Gradient Boosting",
                "metrics": gb_metrics,
                "config": {"n_lags": n_lags},
            }
    except Exception:
        pass

    if not comparison or best_candidate is None:
        fallback = _moving_average_forecast(train, steps=horizon)
        future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        forecast_records = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "forecast": float(v),
                "lower": float(l),
                "upper": float(u),
            }
            for d, v, l, u in zip(future_dates, fallback["forecast"], fallback["lower"], fallback["upper"])
        ]
        return {
            "best_model": "Moving Average",
            "comparison": comparison,
            "forecast": forecast_records,
            "forecast_sum": float(np.sum(fallback["forecast"])),
            "forecast_average": float(np.mean(fallback["forecast"])),
            "artifact_path": None,
            "message": "All models failed. Used moving average fallback.",
        }

    best_name = best_candidate["name"]

    # Refit the best model on the full series
    artifact_path = None
    try:
        if best_name == "SARIMAX":
            final = forecast_sarimax_with_config(
                series,
                steps=horizon,
                order=best_candidate["config"]["order"],
                seasonal_order=best_candidate["config"]["seasonal_order"],
            )
            bundle = {
                "best_model": best_name,
                "model": final["model"],
                "training_end": series.index[-1].isoformat(),
                "horizon": horizon,
                "config": best_candidate["config"],
            }

        elif best_name == "Exponential Smoothing":
            final = forecast_ets(series, steps=horizon)
            bundle = {
                "best_model": best_name,
                "model": final["model"],
                "training_end": series.index[-1].isoformat(),
                "horizon": horizon,
                "config": {},
                "residual_std": final["residual_std"],
            }

        elif best_name == "Random Forest":
            final = forecast_random_forest(series, steps=horizon, n_lags=best_candidate["config"]["n_lags"])
            bundle = {
                "best_model": best_name,
                "model": final["model"],
                "training_end": series.index[-1].isoformat(),
                "horizon": horizon,
                "config": best_candidate["config"],
                "feature_columns": final["feature_columns"],
                "history_tail": final["history_tail"],
            }

        else:
            final = forecast_gradient_boosting(series, steps=horizon, n_lags=best_candidate["config"]["n_lags"])
            bundle = {
                "best_model": best_name,
                "model": final["model"],
                "training_end": series.index[-1].isoformat(),
                "horizon": horizon,
                "config": best_candidate["config"],
                "feature_columns": final["feature_columns"],
                "history_tail": final["history_tail"],
            }

        artifact_path = _save_bundle(bundle, artifact_dir=artifact_dir)
        forecast_values = final["forecast"]
        lower = final["lower"]
        upper = final["upper"]

    except Exception:
        fallback = _moving_average_forecast(series, steps=horizon)
        forecast_values = fallback["forecast"]
        lower = fallback["lower"]
        upper = fallback["upper"]
        best_name = "Moving Average"

    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_records = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "forecast": float(v),
            "lower": float(l),
            "upper": float(u),
        }
        for d, v, l, u in zip(future_dates, forecast_values, lower, upper)
    ]

    return {
        "best_model": best_name,
        "comparison": comparison,
        "forecast": forecast_records,
        "forecast_sum": float(np.sum(forecast_values)),
        "forecast_average": float(np.mean(forecast_values)),
        "artifact_path": artifact_path,
        "message": f"Best model selected: {best_name}",
    }