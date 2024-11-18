import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from helper import frange
from trader import TradingStrategySimulator

# Define the parameters
params = {
    'sell_high': True,
    'buy_low': True,
    'relative_sell': 30,
    'relative_buy': 50,
    'cash_interest': 0.12 / 252,
    'high_sell_factor': 1.015,  # how much above 52 week high in order to sell
    'low_sell_factor': 0.96,  # placeholder, will vary in the loop
    'keep_minimum': 0.5,
    'invest_cap': -250000,
    'total_cap': -500000,
    'hi_lo_weeks': 52,
    'verbose': False,
}

# Define directories
source_dir = "history"
output_dir = "output"

# Get the list of files
file_list = glob.glob(os.path.join(source_dir, '*.csv'))
file_list = [os.path.basename(file) for file in file_list]


# Define the start date range
start_date_range = pd.date_range(start="2010-01-03", end="2022-01-03", freq="12M")

# Prepare a DataFrame to store results
results = []

# Iterate through start dates
for start_date in start_date_range:
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Iterate through low_sell_factor from 0.95 to 1.0 in 0.01 steps
    for low_sell_factor in [round(x, 2) for x in list(frange(0.95, 1.01, 0.01))]:
        params['low_sell_factor'] = low_sell_factor

        # Update the simulator with the current start date and low_sell_factor
        # Initialize the simulator with fixed parameters
        simulator = TradingStrategySimulator(
            initial_invest=5000,
            start_date=start_date_str,
            period_months = 60,
            plan=1,
            source_dir=source_dir,
            output_dir=output_dir,
            params=params
    )
        simulator.create_plots = False

        # Run the simulation
        results.extend(simulator.run(file_list))


# Save results to a CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "parameter_sweep_results.csv"), index=False)

print("Parameter sweep completed. Results saved to parameter_sweep_results.csv")
