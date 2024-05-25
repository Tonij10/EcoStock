# EcoStock

EcoStock is a comprehensive Python package tailored for finance professionals and economists, offering a suite of powerful tools to analyze stock data and economic indicators. Whether you're a seasoned investor, financial analyst, or economist, EcoStock provides the necessary functionality to extract insights and make informed decisions in today's dynamic markets.

## Key Features:

- **Data Retrieval and Visualization:** Fetch historical stock data, economic indicators, and news articles effortlessly, enabling in-depth analysis and informed decision-making.

- **Correlation Analysis:** Explore correlations between stock prices and economic indicators, uncovering hidden relationships and potential market trends.

- **Technical Analysis Tools:** Utilize a wide range of technical analysis tools, including Bollinger Bands, Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and more, to identify patterns and forecast market movements.

- **Predictive Modeling:** Leverage machine learning techniques such as ARIMA and linear regression to predict stock prices and portfolio returns, empowering you to stay ahead of the curve.

- **Portfolio Management:** Calculate portfolio returns, analyze cumulative and annual returns, and assess rolling volatility to optimize portfolio performance and mitigate risk.

- **Global Economic Analysis:** Conduct comparative analysis of GDP growth and economic performance across multiple countries, gaining valuable insights into global economic trends and opportunities.

- **Interactive Visualization:** Generate interactive plots and charts to visualize stock price trends, economic indicators, and technical analysis metrics, facilitating clear communication of findings and strategies.

- **User-Friendly Interface:** Designed with simplicity and efficiency in mind, EcoStock offers intuitive functions and clear documentation, ensuring ease of use for both novice and experienced users alike.

EcoStock empowers finance professionals and economists with the tools they need to navigate today's complex financial landscape, enabling data-driven decision-making and strategic planning.

## Installation

To install the EcoStock package, use pip, a package manager for Python:

```bash
pip install EcoStock

```

## Usage

EcoStock consists of two modules: **functions** and **adalo**.

- The **functions** module provides various functions for retrieving, analyzing, and visualizing economic and stock data.

Below are examples of how to use all the available functions in **functions** module:

```python
from Ecostock.functions import *

# Fetch stock data for a specific company within a date range
get_stock_data('AAPL', '2022-01-01', '2022-12-31')

# Calculate the moving average of a stock's data within a year
moving_avg_stock_data('AAPL', '2022', '2023')

# Generate a MACD (Moving Average Convergence Divergence) plot for a stock within a date range
macd('AAPL', '2020-01-01', '2023-12-31', 12, 26)

# Generate a RSI (Relative Strength Index) plot for a stock within a date range
rsi('AAPL', '2020-01-01', '2023-12-31', 14)

# Generate a VWAP (Volume Weighted Average Price) plot for a stock within a date range
vwap('AAPL', '2020-01-01', '2023-12-31')

# Generate a candlestick chart for a stock within a date range
plot_candlestick('AAPL', '2022-01-01', '2023-12-31')

# Generate a Bollinger Bands plot for a stock within a date range
plot_bollinger_bands('AAPL', '2022-01-01', '2023-12-31')

# Generate a plot of daily returns for multiple stocks within a date range
daily_returns(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2020-01-31')

# Generate a plot of cumulative returns for a portfolio of stocks within a date range
cum_returns(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2023-12-31')

# Generate a plot of annual returns for a portfolio of stocks within a specific year range
annual_returns(['AAPL', 'MSFT', 'GOOGL'], 2020, 2023)

# Generate a plot of rolling volatility for a stock within a date range
rolling_volatility('AAPL', '2020-01-01', '2023-12-31', 30)

# Generate a stock price prediction plot using the ARIMA model within a date range
arima_prediction('AAPL', '2024-05-01', '2024-12-31')

# Generate a stock price prediction plot using linear regression within a date range
lm_prediction('AAPL', '2023-01-01', '2024-01-31', 30)

# Generate a Fibonacci retracement levels plot for a stock within a date range
fibonacci_retracement('AAPL', '2020-01-01', '2020-12-31')

# Generate an Ichimoku Cloud plot for a stock within a date range
ichimoku_cloud('AAPL', '2020-01-01', '2020-12-31')

# Fetch economic data from the World Bank for a specific country within a year
get_world_bank_data('NY.GDP.MKTP.CD', 'US', '2010', '2020')

# Generate a plot of correlation between a stock and economic data within a date range
plot_correlation('AAPL', 'NY.GDP.MKTP.CD', 'US', '2015-01-01', '2022-12-31')

# Generate a plot of an economic indicator for a list of countries within a year range
countries_indicator('NY.GDP.MKTP.CD', ['US', 'CN', 'JP'], '2010', '2020')

# Generate a comparison plot of GDP growth for multiple countries within a year range
gdp_growth(['US', 'CN', 'JP'], '2010', '2020')

# Generate an economic analysis plot for multiple countries within a year range
analyse_economy(['US', 'CN', 'JP'], '2010', '2020')

```

- The **adalo** module contains functions tailored for use in the Adalo app, a no-code platform for building applications. These functions return data in a format suitable for a no-code environment, such as JSON.

Below are examples of how to use all the available functions in **adalo** module:

```python
from Ecostock.adalo import *

# Fetch stock data for a specific company within a date range
candlestick('AAPL', '2022-01-01', '2022-12-31')

# Fetch economic data and calculate correlation for a specific company and country within a year
correlation('AAPL', 'GDP', 'US', '2010', '2020')

# Generate a Bollinger Bands plot for a stock within a date range
bollinger_bands('AAPL', '2022-01-01', '2022-12-31')

# Generate a plot of an economic indicator for a list of countries within a year
indicator_for_countries('NY.GDP.MKTP.CD', ['US', 'CN'], '2012', '2021')

# Fetch news articles related to a specific company
get_news('AAPL')

# Generate a plot of daily returns for multiple stocks within a date range
stock_correlation(['AAPL', 'GOOG'], '2023-01-01', '2023-01-31')

# Generate a stock price prediction plot for a specific company within a date range
stock_prediction('AAPL', '2023-01-01', '2024-12-31')

# Calculate the return of a portfolio of stocks within a date range
portfolio_return(['AAPL', 'GOOG'], '2018-01-01', '2023-12-31')

```

## Documentation

For basic usage and generic information, please refer to the [README](README.md) file.

### API Documentation

The EcoStock package includes a FastAPI application that serves as the API for interacting with the functionalities provided by the package. Below are the available endpoints:

#### Root Endpoint

- **URL:** `/`
- **Description:** Returns a welcome message indicating the successful setup of the EcoStock API.

#### Candlestick Endpoint

- **URL:** `/candlestick/{ticker}/{start_date}/{end_date}`
- **Description:** Generates a price trend chart for the specified stock within the provided date range.

#### Correlation Endpoint

- **URL:** `/correlation/{ticker}/{indicator}/{country}/{start_date}/{end_date}`
- **Description:** Generates a correlation plot between a stock and an economic indicator for the specified country within the provided date range.

#### Bollinger Bands Endpoint

- **URL:** `/bollinger_bands/{ticker}/{start_date}/{end_date}`
- **Description:** Generates a Bollinger Bands plot for the specified stock within the provided date range.

#### Indicator for Countries Endpoint

- **URL:** `/indicator_for_countries/{indicator}/{countries}/{start_date}/{end_date}`
- **Description:** Generates a plot of an economic indicator for a list of countries within the provided date range.

#### Get News Endpoint

- **URL:** `/get_news/{ticker}`
- **Description:** Retrieves news articles related to the specified stock.

#### Stock Correlation Endpoint

- **URL:** `/stock_correlation/{tickers}/{start_date}/{end_date}`
- **Description:** Generates a plot of correlation between multiple stocks within the provided date range.

#### Stock Prediction Endpoint

- **URL:** `/stock_prediction/{ticker}/{start_date}/{end_date}`
- **Description:** Generates a plot of stock price prediction for the specified stock within the provided date range.

#### Portfolio Return Endpoint

- **URL:** `/portfolio_return/{tickers}/{start_date}/{end_date}`
- **Description:** Generates a plot of portfolio return for the specified stocks within the provided date range.

For more details on how to use these endpoints, please refer to the [API documentation](docs/API.md).

## License

EcoStock is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE.md) file. 