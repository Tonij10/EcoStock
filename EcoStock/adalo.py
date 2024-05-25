# This file contains the functions that will be used in the Adalo app. 
# It includes:
# - candlestick: Generate a price trend chart for a given stock ticker and date range.
# - correlation: Generate a correlation plot between a stock and an economic indicator for a given country and date range.
# - bollinger_bands: Generate a Bollinger Bands plot for a given stock ticker and date range.
# - indicator_for_countries: Generate a plot of an economic indicator for multiple countries over a given date range.
# - get_news: Get news articles related to a given stock ticker.
# - stock_correlation: Generate a plot of the daily returns of multiple stocks over a given date range.
# - stock_prediction: Generate a plot of stock price prediction using ARIMA model.
# - portfolio_return: Calculate the cumulative returns of a portfolio of stocks over a given date range.

# Import necessary libraries and modules
import pandas as pd
import requests
import io
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as mticker
from statsmodels.tsa.arima.model import ARIMA
import warnings
import re
from EcoStock.functions import (get_stock_data, moving_avg_stock_data, get_world_bank_data)

# This function generates a stock price trend chart
def candlestick(ticker, start_date, end_date):
    """
    Generate a price trend chart for a given stock ticker and date range.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    """
    try:
        # Fetch the stock data for the given ticker and date range
        stock_data = get_stock_data(ticker, start_date, end_date)

        # If the fetched data is empty, print a message and return None
        if stock_data.empty:
            print(f"No data available for {ticker} from {start_date} to {end_date}")
            return None

        # Create a matplotlib figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10,6))

        # Plot the closing prices against the date
        ax.plot(stock_data.index, stock_data['Close'], label='Close Price')

        # Set up the date locator and formatter for the x-axis
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Set the title of the plot and the labels for the x and y axes
        ax.set_title(f'{ticker} Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price (USD)')

        # Add a grid and a legend to the plot
        ax.grid(True)
        ax.legend()

        # Auto format the date labels on the x-axis
        fig.autofmt_xdate()

        # Save the plot to a BytesIO object in PNG format
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        # Convert the BytesIO object to base64 and return the result
        return base64.b64encode(buf.getbuffer()).decode("ascii")

    except Exception as e:
        # If an exception occurs, print an error message and return None
        print(f"An error occurred: {e}")
        return None

# This function calculates and plots the correlation between stock and economic data
def correlation(ticker, indicator, country, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance and calculate a 1-year moving average, then fetch economic data from the World Bank, 
    and plot the correlation between stock and economic data over time.

    Parameters:
    ticker (str): Ticker symbol of the stock.
    indicator (str): The indicator of interest for the World Bank data.
    country (str): The country ISO code of interest for the World Bank data.
    start_date (str): The start date for the data (format: "YYYY").
    end_date (str): The end date for the data (format: "YYYY").
    """
    # Fetch the 1-year moving average stock data for the given ticker and date range
    stock_data = moving_avg_stock_data(ticker, start_date, end_date)

    # Fetch the economic data for the given indicator, country, and date range from the World Bank
    econ_data = get_world_bank_data(indicator, country, start_date, end_date)
    
    # Extract the label for the economic data from the DataFrame
    econ_label = econ_data['indicator'].iloc[0]

    # Determine the format for the economic data based on the label
    if '$' in econ_label:
        econ_format = 'US$'
    elif '%' in econ_label:
        econ_format = '%'
    else:
        econ_format = None

    # Merge the stock and economic data on the date index
    merged_data = pd.merge(stock_data, econ_data, how='inner', left_index=True, right_index=True)

    # If the merged data is empty, print an error message and return None
    if merged_data.empty:
        print("Error: No matching data found for stock_data and econ_data")
        return None

    # Calculate the correlation between the closing stock price and the economic data
    correlation = merged_data['Close'].corr(merged_data['value'])

    # Set the style for the plot
    sns.set(style="darkgrid")

    # Create a matplotlib figure and axis for the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the stock data on the first y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{ticker} Price', color='tab:red')
    ax1.plot(stock_data.index, stock_data['Close'], color='tab:red', label=f'{ticker} Price')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot the economic data on the second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(econ_label, color='tab:blue')
    ax2.plot(econ_data.index, econ_data['value'], color='tab:blue', label=econ_label)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Format the y-tick labels based on the economic data format
    if econ_format == '%':
        formatter = mticker.FuncFormatter(lambda x, pos: '{:.1f}%'.format(x))
    elif econ_format == 'US$':
        # Define a function to format large numbers in a human-readable way
        def human_format(num):
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return '${:.1f}{}'.format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

        formatter = mticker.FuncFormatter(lambda x, pos: human_format(x))
    else:
        formatter = None

    # If a formatter is defined, apply it to the second y-axis
    if formatter is not None:
        ax2.yaxis.set_major_formatter(formatter)

    # Add a legend and a title to the plot
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
    plt.title(f'Correlation between {ticker} Price and {econ_label}: {correlation:.2f}')

    # Adjust the layout of the plot
    fig.tight_layout()

    # Save the plot to a BytesIO object in PNG format
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to base64 and return the result
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# This function generates a Bollinger Bands plot for a given stock ticker and date range.
def bollinger_bands(ticker, start_date, end_date):
    """
    Generate a Bollinger Bands plot for a given stock ticker and date range.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    """
    
    # Fetch the stock data for the given ticker and date range
    data = get_stock_data(ticker, start_date, end_date)

    # Define the window size for the rolling mean and standard deviation calculations
    window_size = 20
    # Define the number of standard deviations to use for the Bollinger Bands
    num_of_std = 2

    # Calculate the rolling mean of the 'Close' prices over the window size
    rolling_mean = data['Close'].rolling(window=window_size).mean()

    # Calculate the rolling standard deviation of the 'Close' prices over the window size
    rolling_std = data['Close'].rolling(window=window_size).std()

    # Calculate the upper Bollinger Band as the rolling mean plus a number of standard deviations
    upper_band = rolling_mean + (rolling_std * num_of_std)

    # Calculate the lower Bollinger Band as the rolling mean minus a number of standard deviations
    lower_band = rolling_mean - (rolling_std * num_of_std)

    # Create a simple line chart with matplotlib
    fig, ax = plt.subplots(figsize=(10,6))

    # Plot the 'Close' prices, upper band, rolling mean, and lower band
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, upper_band, label='Upper Band', color='red')
    ax.plot(data.index, rolling_mean, label='Rolling Mean', color='blue')
    ax.plot(data.index, lower_band, label='Lower Band', color='green')

    # Set the title and labels for the plot
    ax.set_title('Bollinger Bands for {}'.format(ticker))
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Add a legend to the plot
    ax.legend()

    # Format the date axis with matplotlib's AutoDateLocator and AutoDateFormatter
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Auto format the date labels to avoid overlap
    fig.autofmt_xdate()

    # Save the plot to a BytesIO object in PNG format
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to base64 and return the result
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# This function plots an economic indicator for several countries over a specified date range.
def indicator_for_countries(indicator, countries, start_date, end_date):
    """
    Plot an economic indicator for several countries.

    Parameters:
    indicator (str): The indicator of interest.
    countries (list): The country ISO codes of interest.
    start_date (str): The start date for the data (format: "YYYY").
    end_date (str): The end date for the data (format: "YYYY").
    """
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10,6))

    # Initialize the indicator name to None
    indicator_name = None

    # Loop over each country
    for country in countries:
        # Fetch the World Bank data for the given indicator and country
        df = get_world_bank_data(indicator, country, start_date, end_date)
        # If the DataFrame is not empty, plot the data and set the indicator name
        if not df.empty:
            ax.plot(df.index, df['value'], label=country)
            # Get the indicator name from the first row of the DataFrame
            if indicator_name is None:
                indicator_name = df['indicator'].iloc[0]
    
    # Determine the format for the y-axis labels based on the indicator name
    if '$' in indicator_name:
        econ_format = 'US$'
    elif '%' in indicator_name:
        econ_format = '%'
    else:
        econ_format = None

    # Set the title and y-axis label of the plot based on the indicator name
    if indicator_name is not None:
        ax.set_title(f'{indicator_name} from {start_date} to {end_date}')
        ax.set_ylabel(indicator_name)
    else:
        ax.set_title(f'{indicator} from {start_date} to {end_date}')
        ax.set_ylabel(indicator)

    # Set the x-axis label and add a legend to the plot
    ax.set_xlabel('Year')
    ax.legend()

    # Format the y-tick labels based on the determined format
    if econ_format == '%':
        formatter = mticker.FuncFormatter(lambda x, pos: '{:.1f}%'.format(x))
    elif econ_format == 'US$':
        # Define a function to format large numbers with suffixes like 'K', 'M', 'B', etc.
        def human_format(num):
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return '${:.1f}{}'.format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

        formatter = mticker.FuncFormatter(lambda x, pos: human_format(x))
    else:
        formatter = None

    # If a formatter was defined, set it as the major formatter for the y-axis
    if formatter is not None:
        ax.yaxis.set_major_formatter(formatter)

    # Save the plot to a BytesIO object in PNG format
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Convert the BytesIO object to base64 and return the result
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# This function fetches news data related to a given stock ticker from the GNews API.
def get_news(ticker):
    """
    Get news articles related to a given stock ticker.

    Parameters:
    ticker (str): The stock ticker symbol.
    """
    # Define the API key and the URL for the GNews API
    api_key = '8eb9080b292fef6aa96f1b6a78df86b9'
    url = f'https://gnews.io/api/v4/search?q={ticker}&token={api_key}&lang=en'

    # Try to send a GET request to the GNews API
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        # If an HTTP error occurs, print the error and return None
        print(f'HTTP error occurred: {http_err}')
        return None
    except Exception as err:
        # If any other error occurs, print the error and return None
        print(f'Other error occurred: {err}')
        return None

    # Try to parse the response as JSON
    try:
        data = response.json()
    except ValueError:
        # If the response cannot be parsed as JSON, print an error and return None
        print('Error parsing JSON response')
        return None

    # Prepare a list to hold the articles
    articles = []

    # Loop over each article in the 'articles' field of the JSON data
    for article in data.get('articles', []):
        # If the 'content' field is present in the article
        if 'content' in article:
            # Remove the number inside the brackets and append 'read more...' to the 'content' field
            article['content'] = re.sub(r'\[\d+ chars\]', 'read more...', article['content'])
            # Append the article to the list of articles
            articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content'),
                'url': article.get('url'),
                'image': article.get('image'),
                'publishedAt': article.get('publishedAt'),
                'source': article.get('source', {}).get('name')
            })

    # Return the list of articles
    return articles

# This function generates a plot of the daily returns of multiple stocks over a given date range.
def stock_correlation(tickers, start_date, end_date):
    """
    Generate a plot of the daily returns of multiple stocks over a given date range.
    
    Parameters:
    tickers (list): A list of ticker symbols.
    start_date (str): The start date in format 'YYYY-MM-DD'.
    end_date (str): The end date in format 'YYYY-MM-DD'.
    """
    # Initialize an empty DataFrame to store the stock data
    stock_data = pd.DataFrame()

    # Loop over each ticker symbol
    for ticker in tickers:
        # Fetch the stock data for the ticker over the specified date range
        data = get_stock_data(ticker, start_date, end_date)
        # Add the closing prices to the stock data DataFrame
        stock_data[ticker] = data['Close']
    
    # Calculate the daily returns of the stocks in percentage
    returns = stock_data.pct_change() * 100
    
    # Create a new figure and axes for the plot with a smaller size
    fig, ax = plt.subplots(figsize=(8,4))

    # Loop over each ticker symbol
    for ticker in tickers:
        # Plot the daily returns of the stock with a thinner line and some transparency
        ax.plot(returns.index, returns[ticker], label=ticker, linewidth=0.6, alpha=0.8)
    
    # Set the title, x-axis label, and y-axis label of the plot
    ax.set_title('Daily Returns of Stocks')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return (%)')

    # Add a grid to the plot and a legend with a smaller font size
    ax.grid(True)
    ax.legend(fontsize='x-small')

    # Automatically format the date labels to prevent overlap
    fig.autofmt_xdate()

    # Save the plot to a BytesIO object in PNG format with a lower DPI
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Convert the contents of the BytesIO object to base64 and return the result
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# This function generates a plot of stock price prediction using ARIMA model
def stock_prediction(ticker, start_date, end_date):
    """
    Generate a plot of stock price prediction using ARIMA model.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    """
    # Suppress warnings to keep output clean
    warnings.filterwarnings('ignore')

    # Fetch the stock data from the specified ticker and date range
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Calculate 10-day and 50-day moving averages of the closing prices
    stock_data['MA10'] = stock_data['Close'].rolling(10).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(50).mean()
    
    # Calculate Bollinger Bands, which are a volatility indicator
    stock_data['20 Day MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['20 Day STD'] = stock_data['Close'].rolling(window=20).std()
    stock_data['Upper Band'] = stock_data['20 Day MA'] + (stock_data['20 Day STD'] * 2)
    stock_data['Lower Band'] = stock_data['20 Day MA'] - (stock_data['20 Day STD'] * 2)
    
    # Calculate the Relative Strength Index (RSI), a momentum oscillator
    delta = stock_data['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=14).mean()
    average_loss = abs(down.rolling(window=14).mean())
    rs = average_gain / average_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate the Moving Average Convergence Divergence (MACD), a trend-following momentum indicator
    stock_data['26 EMA'] = stock_data['Close'].ewm(span=26).mean()
    stock_data['12 EMA'] = stock_data['Close'].ewm(span=12).mean()
    stock_data['MACD'] = stock_data['12 EMA'] - stock_data['26 EMA']
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()
    
    # Fit the ARIMA model to the closing prices
    arima_order = (5,1,0)  # Order of the ARIMA model
    model = ARIMA(stock_data['Close'], order=arima_order)
    model_fit = model.fit()
    
    # Use the fitted model to make a prediction
    forecast_result = model_fit.forecast(steps=1)
    prediction = forecast_result.iloc[0]

    # Create a new DataFrame for the prediction
    prediction_data = pd.DataFrame({'Close': [prediction]}, index=[pd.to_datetime(end_date) + pd.DateOffset(days=1)])

    # Concatenate the historical and prediction data
    full_data = pd.concat([stock_data, prediction_data])

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(3, 1, figsize=(14, 8))
    
    # Plot the closing prices, moving averages, Bollinger Bands, and prediction
    ax[0].plot(full_data.index, full_data['Close'], label='Close Price')
    ax[0].plot(stock_data.index, stock_data['MA10'], label='10-day MA')
    ax[0].plot(stock_data.index, stock_data['MA50'], label='50-day MA')
    ax[0].plot(stock_data.index, stock_data['Upper Band'], label='Upper Bollinger Band')
    ax[0].plot(stock_data.index, stock_data['Lower Band'], label='Lower Bollinger Band')
    ax[0].plot(prediction_data.index, prediction_data['Close'], 'ro', label='Predicted Price')
    ax[0].set_title(f'{ticker} Stock Price Prediction: {prediction:.2f}$')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Price')
    ax[0].legend(fontsize='x-small')
    ax[0].grid(True)
    
    # Plot the RSI
    ax[1].plot(stock_data.index, stock_data['RSI'], label='RSI')
    ax[1].set_title('Relative Strength Index')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('RSI')
    ax[1].legend(fontsize='x-small')
    ax[1].grid(True)
    
    # Plot the MACD and its signal line
    ax[2].plot(stock_data.index, stock_data['MACD'], label='MACD')
    ax[2].plot(stock_data.index, stock_data['Signal Line'], label='Signal Line')
    ax[2].set_title('Moving Average Convergence Divergence')
    ax[2].set_xlabel('Date')
    ax[2].set_ylabel('MACD')
    ax[2].legend(fontsize='x-small')
    ax[2].grid(True)
    
    # Adjust the format of the x-axis labels to prevent overlap
    fig.autofmt_xdate()
    fig.tight_layout()

    # Save the plot to a BytesIO object with a lower DPI for smaller file size
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)

    # Convert the buffer contents to base64 and return as a string
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# This function calculates the cumulative return of a portfolio of stocks
def portfolio_return(tickers, start_date, end_date):
    """
    Calculate the cumulative return of a portfolio.

    Parameters:
    tickers (list): A list of ticker symbols.
    start_date (str): The start date in format 'YYYY-MM-DD'.
    end_date (str): The end date in format 'YYYY-MM-DD'.

    Returns:
    str: Base64 encoded string of the plot.
    """
    # Initialize an empty DataFrame to store the return of each stock in the portfolio
    portfolio_return = pd.DataFrame()

    # Loop over each ticker symbol in the portfolio
    for ticker in tickers:
        # Fetch the historical data for the current stock
        stock_data = get_stock_data(ticker, start_date, end_date)

        # Calculate the daily return of the current stock
        stock_data['Return'] = stock_data['Close'].pct_change()

        # If this is the first stock, copy its return data to the portfolio return DataFrame
        if portfolio_return.empty:
            portfolio_return = stock_data[['Return']].copy()
            portfolio_return.columns = [ticker]
        else:
            # If this is not the first stock, join its return data with the existing portfolio return data
            portfolio_return = portfolio_return.join(stock_data['Return'], how='outer', rsuffix=ticker)

    # Calculate the average daily return of the portfolio
    portfolio_return['Average'] = portfolio_return.mean(axis=1)

    # Calculate the cumulative return of the portfolio
    portfolio_return['Cumulative'] = (1 + portfolio_return['Average']).cumprod()

    # Convert the cumulative return to percentage format
    portfolio_return['Cumulative'] = portfolio_return['Cumulative'] * 100

    # Create a new figure for plotting
    fig = plt.figure()

    # Plot the cumulative return of the portfolio over time
    plt.plot(portfolio_return.index, portfolio_return['Cumulative'])
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')

    # Automatically adjust the date labels to prevent overlap
    fig.autofmt_xdate()

    # Save the plot to a BytesIO object with a lower DPI for smaller file size
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)

    # Convert the buffer contents to base64 and return as a string
    return base64.b64encode(buf.getbuffer()).decode("ascii")