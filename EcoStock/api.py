# This is a FastAPI application that serves as the API for the EcoStock package. 
# It contains various endpoints for generating financial and economical data visualizations and fetching news related to stocks. 
# The API is deployed on Heroku platform to interact with the EcoStock functions on an ADALO app.

# Import necessary libraries and modules
from fastapi import FastAPI, HTTPException,Response
import base64
import argparse
import uvicorn
from EcoStock.adalo import (
    candlestick, 
    bollinger_bands,
    correlation, 
    indicator_for_countries, 
    stock_correlation, 
    get_news,
    stock_prediction,
    portfolio_return
    )

# Initialize FastAPI application
app = FastAPI()

# Define root endpoint
@app.get("/")
async def read_root():
    return {"results": {
    "message": "Welcome to EcoStock API!"
  }}

# Define endpoint for generating stock price trend charts
@app.get("/candlestick/{ticker}/{start_date}/{end_date}")
def candlestick_api(ticker: str, start_date: str, end_date: str):
    try:
        image = candlestick(ticker, start_date, end_date)
        image_bytes = base64.b64decode(image)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define endpoint for generating correlation plots
@app.get("/correlation/{ticker}/{indicator}/{country}/{start_date}/{end_date}")
def correlation_api(ticker: str, indicator: str, country: str, start_date: str, end_date: str):
    try:
        image = correlation(ticker, indicator, country, start_date, end_date)
        image_bytes = base64.b64decode(image)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define endpoint for generating bollinger bands
@app.get("/bollinger_bands/{ticker}/{start_date}/{end_date}")
def bollinger_bands_api(ticker: str, start_date: str, end_date: str):
    try:
        image = bollinger_bands(ticker, start_date, end_date)
        image_bytes = base64.b64decode(image)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define endpoint for generating indicator for countries
@app.get('/indicator_for_countries/{indicator}/{countries}/{start_date}/{end_date}')
async def indicator_countries_api(indicator: str, countries: str, start_date: str, end_date: str):
    try:
        countries = countries.split(',')
        image_base64 = indicator_for_countries(indicator, countries, start_date, end_date)
        image_bytes = base64.b64decode(image_base64)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define endpoint for getting news related to a stock
@app.get("/get_news/{ticker}")
async def get_news_api(ticker: str):
    try:
        data = get_news(ticker)
        if data is None:
            raise HTTPException(status_code=400, detail="Error fetching news data")
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Define endpoint for generating stock correlation
@app.get('/stock_correlation/{tickers}/{start_date}/{end_date}')
async def stock_correlation_api(tickers: str, start_date: str, end_date: str):
    try:
        tickers = tickers.split(',')
        image_base64 = stock_correlation(tickers, start_date, end_date)
        image_bytes = base64.b64decode(image_base64)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define endpoint for generating stock prediction
@app.get('/stock_prediction/{ticker}/{start_date}/{end_date}')
async def stock_prediction_api(ticker: str, start_date: str, end_date: str):
    try:
        image_base64 = stock_prediction(ticker, start_date, end_date)
        image_bytes = base64.b64decode(image_base64)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define endpoint for generating portfolio return
@app.get('/portfolio_return/{tickers}/{start_date}/{end_date}')
async def portfolio_return_api(tickers: str, start_date: str, end_date: str):
    try:
        tickers = tickers.split(',')
        image_base64 = portfolio_return(tickers, start_date, end_date)
        image_bytes = base64.b64decode(image_base64)
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Run the FastAPI application
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI application.")
    parser.add_argument("-p", "--port", type=int, default=8000, help="The port to bind.")
    parser.add_argument("-r", "--reload", action="store_true", help="Enable hot reloading.")
    args = parser.parse_args()
    uvicorn.run("api:app", host="127.0.0.1", port=args.port, reload=args.reload)
