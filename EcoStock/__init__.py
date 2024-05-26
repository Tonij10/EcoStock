"""
EcoStock is a Python package that provides a set of functions for analyzing and visualizing financial and macroeconomic data.
This package includes functions for generating candlestick charts, calculating Bollinger Bands, calculating the correlation between stocks and macroeconomic indicators,
and much more. In addition, EcoStock provides a FastAPI-based API to access useful functions for use in no-code programming apps (e.g., via HTTP requests).
The functions provided by EcoStock can be used to generate visualizations and insights that can help users make informed decisions in the financial markets.

"""

# The current version of the EcoStock package
__version__ = "1.4"

# Import the FastAPI application from the api module
from .api import app

# Import the functions provided by the adalo module
from .adalo import (
    candlestick,
    bollinger_bands,
    correlation,
    indicator_for_countries,
    stock_correlation,
    get_news,
    stock_prediction,
    portfolio_return
)

# Import the functions provided by the functions module
from .functions import (
    get_stock_data,
    plot_candlestick,
    moving_avg_stock_data,
    get_world_bank_data,
    plot_correlation,
    plot_bollinger_bands,
    countries_indicator,
    daily_returns,
    arima_prediction,
    cum_returns,
    annual_returns,
    rolling_volatility,
    macd,
    rsi,
    vwap,
    lm_prediction,
    fibonacci_retracement,
    ichimoku_cloud,
    gdp_growth,
    analyse_economy
)