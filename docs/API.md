# EcoStock API Documentation

The EcoStock package includes a FastAPI application that serves as the API for interacting with the functionalities provided by the package. The API is deployed on Heroku platform to interact with the EcoStock functions on an Adalo app.

## Root Endpoint

- **URL:** `/`
- **Method:** GET
- **Description:** Returns a welcome message indicating the successful setup of the EcoStock API.

## Candlestick Endpoint

- **URL:** `/candlestick/{ticker}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a price trend chart for the specified stock within the provided date range.

## Correlation Endpoint

- **URL:** `/correlation/{ticker}/{indicator}/{country}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a correlation plot between a stock and an economic indicator for the specified country within the provided date range.

## Bollinger Bands Endpoint

- **URL:** `/bollinger_bands/{ticker}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a Bollinger Bands plot for the specified stock within the provided date range.

## Indicator for Countries Endpoint

- **URL:** `/indicator_for_countries/{indicator}/{countries}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a plot of an economic indicator for a list of countries within the provided date range.

## Get News Endpoint

- **URL:** `/get_news/{ticker}`
- **Method:** GET
- **Description:** Retrieves news articles related to the specified stock.

## Stock Correlation Endpoint

- **URL:** `/stock_correlation/{tickers}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a plot of correlation between multiple stocks within the provided date range.

## Stock Prediction Endpoint

- **URL:** `/stock_prediction/{ticker}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a plot of stock price prediction for the specified stock within the provided date range.

## Portfolio Return Endpoint

- **URL:** `/portfolio_return/{tickers}/{start_date}/{end_date}`
- **Method:** GET
- **Description:** Generates a plot of portfolio return for the specified stocks within the provided date range.

This documentation provides an overview of all the available endpoints in the EcoStock API, along with their URLs, HTTP methods, and descriptions. You can use these endpoints to interact with the EcoStock functionalities programmatically.