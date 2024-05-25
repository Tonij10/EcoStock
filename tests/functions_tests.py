# In this file, the unittest module is used for testing. 
# Each function from the EcoStock.functions module has its own test case. In each test case, the function is run with some parameters. 
# If the function raises an exception, the test case fails and the exception is printed. 
# If the function does not raise an exception, the test case passes. 
# If this script is run directly, it loads all the test cases from this module and runs them.

import unittest
import sys
from EcoStock.functions import (get_stock_data, plot_candlestick, moving_avg_stock_data, get_world_bank_data, 
                                plot_correlation, plot_bollinger_bands, countries_indicator, daily_returns, 
                                arima_prediction, cum_returns, annual_returns, rolling_volatility, macd, rsi, 
                                vwap, lm_prediction, fibonacci_retracement, ichimoku_cloud, gdp_growth, analyse_economy)

class TestFunctions(unittest.TestCase):

    def test_get_stock_data(self):
        try:
            get_stock_data('AAPL', '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"get_stock_data raised an exception: {e}")

    def test_plot_candlestick(self):
        try:
            plot_candlestick('AAPL', '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"plot_candlestick raised an exception: {e}")
    
    def test_moving_avg_stock_data(self):
        try:
            moving_avg_stock_data('AAPL', '2020', '2022')
        except Exception as e:
            self.fail(f"moving_avg_stock_data raised an exception: {e}")
        
    def test_get_world_bank_data(self):
        try:
            get_world_bank_data('NY.GDP.MKTP.CD', 'US', '2020', '2022')
        except Exception as e:
            self.fail(f"get_world_bank_data raised an exception: {e}")
        
    def test_plot_correlation(self):
        try:
            plot_correlation('AAPL', 'NY.GDP.MKTP.CD', 'US', '2020', '2022')
        except Exception as e:
            self.fail(f"plot_correlation raised an exception: {e}")

    def test_plot_bollinger_bands(self):
        try:
            plot_bollinger_bands('AAPL', '2022-01-01', '2023-01-05')
        except Exception as e:
            self.fail(f"plot_bollinger_bands raised an exception: {e}")    

    def test_countries_indicator(self):
        try:
            countries_indicator('NY.GDP.MKTP.CD', ["USA","JPN","CHN"], '2010', '2020')
        except Exception as e:
            self.fail(f"countries_indicator raised an exception: {e}")

    def test_daily_returns(self):
        try:
            daily_returns(['AAPL', 'GOOG'], '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"daily_returns raised an exception: {e}")

    def test_arima_prediction(self):
        try:
            arima_prediction('AAPL', '2022-01-01', '2024-09-05')
        except Exception as e:
            self.fail(f"arima_prediction raised an exception: {e}")

    def test_cum_returns(self):
        try:
            cum_returns(['AAPL', 'GOOG'], '2015-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"cum_returns raised an exception: {e}")

    def test_annual_returns(self):
        try:
            annual_returns(['AAPL', 'GOOG'], 2020, 2023)
        except Exception as e:
            self.fail(f"annual_returns raised an exception: {e}")

    def test_rolling_volatility(self):
        try:
            rolling_volatility('AAPL', '2020-01-01', '2022-01-05', 5)
        except Exception as e:
            self.fail(f"rolling_volatility raised an exception: {e}")

    def test_macd(self):
        try:
            macd('AAPL', '2020-01-01', '2022-01-05', 12, 26)
        except Exception as e:
            self.fail(f"macd raised an exception: {e}")

    def test_rsi(self):
        try:
            rsi('AAPL', '2020-01-01', '2022-01-05', 14)
        except Exception as e:
            self.fail(f"rsi raised an exception: {e}")

    def test_vwap(self):
        try:
            vwap('AAPL', '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"vwap raised an exception: {e}")

    def test_lm_prediction(self):
        try:
            lm_prediction('AAPL', '2015-01-01', '2023-01-05', 16)
        except Exception as e:
            self.fail(f"lm_prediction raised an exception: {e}")

    def test_fibonacci_retracement(self):
        try:
            fibonacci_retracement('AAPL', '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"fibonacci_retracement raised an exception: {e}")

    def test_ichimoku_cloud(self):
        try:
            ichimoku_cloud('AAPL', '2021-01-01', '2024-01-05')
        except Exception as e:
            self.fail(f"ichimoku_cloud raised an exception: {e}")

    def test_gdp_growth(self):
        try:
            gdp_growth(['US','CHN','DEU'], '2015', '2022')
        except Exception as e:
            self.fail(f"gdp_growth raised an exception: {e}")

    def test_analyse_economy(self):
        try:
            analyse_economy(['US','CHN','DEU'], '2015', '2022')
        except Exception as e:
            self.fail(f"analyse_economy raised an exception: {e}")

    if __name__ == '__main__':
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        unittest.TextTestRunner().run(suite)