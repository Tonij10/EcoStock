# EcoStock

EcoStock is a comprehensive Python package designed for finance professionals and economists. It offers a suite of powerful tools for analyzing stock data and economic indicators, making it easier to extract insights and make informed decisions in today's dynamic markets.

## Key Features

- **Data Retrieval and Visualization**: Effortlessly fetch historical stock data, economic indicators and news articles.
- **Correlation Analysis**: Explore correlations between stock prices and economic indicators to uncover hidden relationships.
- **Technical Analysis Tools**: Utilize tools like Bollinger Bands, MACD and RSI to identify patterns and forecast market movements.
- **Predictive Modeling**: Leverage ARIMA and linear regression models to predict stock prices and portfolio returns.
- **Portfolio Management**: Calculate and analyze portfolio returns, cumulative returns and rolling volatility.
- **Global Economic Analysis**: Conduct comparative analysis of GDP growth and economic performance across countries.
- **Interactive Visualization**: Create interactive plots and charts for clear communication of findings.
- **User-Friendly Interface**: Intuitive functions and clear documentation for ease of use.

## Installation

To install the EcoStock package, use pip:

```bash
pip install EcoStock

```

## Basic Usage

### Fetch Stock Data

```python
from EcoStock.functions import get_stock_data

# Fetch stock data for Apple Inc. within a date range
get_stock_data('AAPL', '2022-01-01', '2022-12-31')

```
### Calculate Moving Average

```python
from EcoStock.functions import moving_avg_stock_data

# Calculate the moving average of Apple's stock data within a year
moving_avg_stock_data('AAPL', '2022', '2023')

```
### Generate MACD Plot

```python
from EcoStock.functions import macd

# Generate a MACD plot for Apple Inc. within a date range
macd('AAPL', '2020-01-01', '2023-12-31', 12, 26)

```

## Modules

EcoStock consists of two main modules:

- **functions**: Provides various functions for retrieving, analyzing, and visualizing economic and stock data.
- **adalo**: Contains functions tailored for use in the Adalo app, a no-code platform for building applications.

### Functions Module Example

```python
from EcoStock.functions import *

# Generate a Bollinger Bands plot for Apple Inc. within a date range
plot_bollinger_bands('AAPL', '2022-01-01', '2023-12-31')

```
### Adalo Module Example

```python
from EcoStock.adalo import *

# Get news articles of Apple Inc. for no-code programming apps (e.g Adalo)
get_news('AAPL')

```

## Documentation

For more examples and detailed list of available functions, please refer to the [documentation](docs/index.md).

### API Documentation

The EcoStock package includes a FastAPI application that serves as the API for interacting with the functionalities provided by the package.

For more details, please refer to the [API documentation](docs/API.md).

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request on GitHub.

## License

EcoStock is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE.md) file.