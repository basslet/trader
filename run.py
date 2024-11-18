import pandas as pd
import matplotlib.pyplot as plt
import os, glob

from trader import TradingStrategySimulator


params = {
    'sell_high': True,
    'buy_low': True,
    'relative_sell': 30,
    'relative_buy': 50,
    'cash_interest': 0.12/252,
    'high_sell_factor': 1.015, # how much above 52 week high in order to sell
    'low_sell_factor': 1, # how much below 52 week low in order to sell
    'keep_minimum': 0.5,
    'invest_cap': -250000,
    'total_cap': -500000,
    'hi_lo_weeks': 52,
    'verbose': False,
}


source_dir = "history"
output_dir = "output"


# Use glob to find all CSV files in the specified directory
file_list = glob.glob(os.path.join(source_dir, '*.csv'))
file_list = [os.path.basename(file) for file in file_list]

simulator = TradingStrategySimulator(initial_invest=5000,
                                    start_date="2018-07-03",
                                    period_months = 60,
                                    plan=1,
                                    source_dir = source_dir,
                                    output_dir = output_dir,
                                    params=params)
simulator.run(file_list)
