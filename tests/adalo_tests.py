# In this file, the unittest module is used for testing. 
# Each function from the EcoStock.adalo module has its own test case. In each test case, the function is run with some parameters. 
# If the function raises an exception, the test case fails and the exception is printed. 
# If the function does not raise an exception, the test case passes. 
# If this script is run directly, it loads all the test cases from this module and runs them.

import unittest
import sys
from EcoStock.adalo import (candlestick, bollinger_bands, correlation, indicator_for_countries, stock_correlation, 
                            get_news, stock_prediction, portfolio_return)

class TestFunctions(unittest.TestCase):

    def test_candlestick(self):
        try:
            candlestick('AAPL', '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"candlestick raised an exception: {e}")

    def test_bollinger_bands(self):
        try:
            bollinger_bands('AAPL', '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"bollinger_bands raised an exception: {e}")
            
    def test_correlation(self):
        try:
            correlation('AAPL', 'NY.GDP.MKTP.CD', 'US', '2020', '2022')
        except Exception as e:
            self.fail(f"correlation raised an exception: {e}")

    def test_indicator_for_countries(self):
        try:
            indicator_for_countries('NY.GDP.MKTP.CD', ["USA","JPN","CHN"], '2010', '2020')
        except Exception as e:
            self.fail(f"countries_indicator raised an exception: {e}")

    def test_stock_correlation(self):
        try:
            stock_correlation(['AAPL', 'GOOG'], '2020-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"daily_returns raised an exception: {e}")

    def test_stock_prediction(self):
        try:
            stock_prediction('AAPL', '2022-01-01', '2024-09-05')
        except Exception as e:
            self.fail(f"stock_prediction raised an exception: {e}")

    def test_portfolio_return(self):
        try:
            portfolio_return(['AAPL', 'GOOG'], '2015-01-01', '2022-01-05')
        except Exception as e:
            self.fail(f"portfolio_return raised an exception: {e}")

    def test_get_news(self):
        try:
            get_news('AAPL')
        except Exception as e:
            self.fail(f"get_news raised an exception: {e}")

    if __name__ == '__main__':
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        unittest.TextTestRunner().run(suite)