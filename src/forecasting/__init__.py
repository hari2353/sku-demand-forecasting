"""Public API surface for the forecasting package."""

from forecasting.backtest import (
    BacktestConfig,
    backtest,
    score_forecaster,
    select_model,
)
from forecasting.forecasters import (
    DEFAULT_FORECASTERS,
    SBA,
    AutoETS,
    Croston,
    LinearTrend,
    SeasonalNaive,
    candidate_pool_for,
    safe_forecast,
)
from forecasting.metrics import bias, rmse, smape
from forecasting.models import (
    BacktestWindow,
    DemandPattern,
    ForecastResult,
    Scorecard,
    SkuPlan,
    SkuSeries,
)
from forecasting.portfolio import (
    PortfolioPlanner,
    PortfolioResult,
    PortfolioSummary,
    pattern_mix,
    plan_to_dict,
)
from forecasting.synthetic import classify, make_portfolio, make_sku

__all__ = [
    "DEFAULT_FORECASTERS",
    "SBA",
    "AutoETS",
    "BacktestConfig",
    "BacktestWindow",
    "Croston",
    "DemandPattern",
    "ForecastResult",
    "LinearTrend",
    "PortfolioPlanner",
    "PortfolioResult",
    "PortfolioSummary",
    "Scorecard",
    "SeasonalNaive",
    "SkuPlan",
    "SkuSeries",
    "backtest",
    "bias",
    "candidate_pool_for",
    "classify",
    "make_portfolio",
    "make_sku",
    "pattern_mix",
    "plan_to_dict",
    "rmse",
    "safe_forecast",
    "score_forecaster",
    "select_model",
    "smape",
]
