import pandas as pd
import matplotlib.pyplot as plt
import os, glob


class TradingStrategySimulator:
    def __init__(self, initial_invest, start_date, period_months, plan, source_dir, output_dir, params):
        """Initialize the simulator with initial parameters."""
        # files
        self.filename = None
        self.source_dir = source_dir
        self.output_dir = output_dir
        # portfolio
        self.initial_invest = initial_invest
        self.initial_shares = 0
        self.start_date = pd.to_datetime(start_date)
        self.period_months = period_months
        self.end_date = self.start_date + pd.DateOffset(months=period_months)
        self.plan = plan
        # parameter
        self.params = params
        self.start_price = 0
        # output data
        self.metrics_data = []
        self.total_cash = {}

    def load_data(self, filename):
        """Load a CSV file and prepare it for simulation."""
        self.filename = filename
        filepath = os.path.join(self.source_dir, filename)
        print(f"Loading file: {filepath}")
        try:
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(by='Date').reset_index(drop=True)
            return data
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return None

    def validate_date(self, meta_data):
        # Ensure meta_data['Date'] is in datetime format
        meta_data['Date'] = pd.to_datetime(meta_data['Date'])

        # Update start_date if the first date in meta_data['Date'] is later
        first_date_in_meta = meta_data['Date'].iloc[0]  # First date in meta_data

        start_date = self.start_date
        end_date = self.end_date

        # Ensure start_date aligns with available dates
        if self.start_date < first_date_in_meta:
            start_date = first_date_in_meta
        else:
            # Find the next available date >= self.start_date
            idx = meta_data['Date'].searchsorted(self.start_date)
            start_date = meta_data['Date'].iloc[idx] if idx < len(meta_data) else first_date_in_meta

        # Calculate the end_date based on the new start_date
        end_date = start_date + pd.DateOffset(months=self.period_months)

        # Ensure end_date doesn't exceed the last available date
        last_date_in_meta = meta_data['Date'].iloc[-1]
        if end_date > last_date_in_meta:
            end_date = last_date_in_meta

            print(f"## Updated start_date to {start_date}")
        return start_date, end_date


    def apply_strategy(self, meta_data):
        """Apply the trading strategy with interest"""
        portfolio = {'cash': -self.initial_shares * self.start_price,
                     'shares': self.initial_shares, 'transactions': []}
        outside_range = False
        last_transaction_month = None

        week_high_str = f'{self.params["hi_lo_weeks"]}_Week_High'
        week_low_str = f'{self.params["hi_lo_weeks"]}_Week_Low'

        for idx, row in meta_data.iterrows():
            current_month = row['Date'].month
            current_year = row['Date'].year

            if not outside_range and (last_transaction_month != (current_year, current_month)):
                if self.params['sell_high'] \
                    and row['High'] >= row[week_high_str] \
                    and portfolio['shares']*self.params['keep_minimum'] > self.initial_shares:

                    sell_amount = portfolio['shares'] * self.params['relative_sell'] / 100
                    portfolio['cash'] += sell_amount * row[week_high_str]
                    portfolio['shares'] -= sell_amount
                    outside_range = True
                    last_transaction_month = (current_year, current_month)
                elif self.params['buy_low'] \
                    and row['Low'] < row[week_low_str] \
                    and portfolio['cash'] > self.params['invest_cap']:

                    buy_amount = portfolio['shares'] * self.params['relative_buy'] / 100
                    buy_value = buy_amount * row[week_low_str]
                    available_cash = None

                    # make sure we don't exceed the cash limits
                    if self.check_balance(row['Date']) - buy_value < self.params['total_cap']:
                        available_cash = self.params['total_cap'] - self.check_balance(row['Date'])
                    elif buy_amount * row[week_low_str] + portfolio['cash'] < self.params['invest_cap']:
                        available_cash = self.params['invest_cap'] - portfolio['cash']
                        print('stock limit')
                    if available_cash != None:
                        buy_amount = abs(int(available_cash / row[week_low_str]))
                        buy_value = buy_amount * row[week_low_str]


                    portfolio['cash'] -= buy_value
                    portfolio['shares'] += buy_amount
                    outside_range = True
                    last_transaction_month = (current_year, current_month)

            if row[week_low_str] <= row['Low'] <= row['High'] <= row[week_high_str]:
                outside_range = False

            meta_data.loc[idx, 'Cash_S'] = portfolio['cash']
            meta_data.loc[idx, 'Portfolio_S'] = portfolio['shares']
            portfolio['cash'] = portfolio['cash'] + portfolio['cash']*self.params['cash_interest']
            self.accumulate_cash_balance(row['Date'],portfolio['cash'])

    def apply_monthly_plan(self, meta_data):
        """Apply a monthly buy/sell plan with interest """
        portfolio = {'cash': -self.initial_shares * self.start_price,
                     'shares': self.initial_shares, 'transactions': []}

        for idx, row in meta_data.iterrows():
            if row['Date'].day == 15:
                if self.plan < 0 and portfolio['shares'] >= abs(self.plan):
                    portfolio['cash'] += abs(self.plan) * row['Close']
                    portfolio['shares'] += self.plan
                elif self.plan > 0:
                    portfolio['cash'] -= self.plan * row['Close']
                    portfolio['shares'] += self.plan
            meta_data.loc[idx, 'Cash_M'] = portfolio['cash']
            meta_data.loc[idx, 'Portfolio_M'] = portfolio['shares']
            portfolio['cash'] = portfolio['cash'] + portfolio['cash']*self.params['cash_interest']

        # Calculate the final value of the portfolio
        final_closing_price = meta_data.iloc[-1]['Close']
        portfolio_value = portfolio['cash'] + portfolio['shares'] * final_closing_price
        return {
            'Final Portfolio Value': portfolio_value,
            'Transactions': portfolio['transactions']
        }




    def apply_hold_interest(self, meta_data):
        """
        Apply the hold with interest
        """
        portfolio = {'cash': self.initial_shares * self.start_price*-1, 'shares': self.initial_shares, 'transactions': []}
        for idx, row in meta_data.iterrows():

            meta_data.loc[idx, 'Cash_H'] = portfolio['cash']
            portfolio['cash'] = portfolio['cash'] + portfolio['cash']*self.params['cash_interest']
        return meta_data

    def accumulate_cash_balance(self, date, transaction):
        """Accumulate total cash"""
        if date in self.total_cash:
            self.total_cash[date] += transaction
        else:
            self.total_cash[date] = transaction

    def check_balance(self, date):
        """Accumulate total cash"""
        if date in self.total_cash:
            return max(self.total_cash[date], self.params['total_cap'])
        else:
            return 0

    def calculate_metrics(self, meta_data):
        """Calculate portfolio metrics like returns and final value."""
        # Calculate the final portfolio value


        metrics_dict = {}
        metrics_dict['name'] = os.path.splitext(self.filename)[0]
        metrics_dict["final_closing_price"] = meta_data.iloc[-1]['Close']
        metrics_dict["initial_invest"] = self.initial_shares*self.start_price

        metrics_dict["portfolio_value_hold"] = meta_data.iloc[-1]['Portfolio_H'] * metrics_dict["final_closing_price"]
        metrics_dict["portfolio_value_monthly"] = meta_data.iloc[-1]['Portfolio_M'] * metrics_dict["final_closing_price"]
        metrics_dict["portfolio_value_strategy"] = meta_data.iloc[-1]['Portfolio_S'] * metrics_dict["final_closing_price"]

        metrics_dict["cash_hold"] = meta_data.iloc[-1]['Cash_H']
        metrics_dict["cash_monthly"] = meta_data.iloc[-1]['Cash_M']
        metrics_dict["cash_strategy"] = meta_data.iloc[-1]['Cash_S']


        # Calculate returns
        metrics_dict['return_hold'] = (metrics_dict['portfolio_value_hold'] + metrics_dict['cash_hold']) / (self.initial_shares*self.start_price)*100
        metrics_dict['return_monthly'] = (metrics_dict['portfolio_value_monthly'] + metrics_dict['cash_monthly']) / (self.initial_shares*self.start_price)*100
        metrics_dict['return_strategy'] = (metrics_dict['portfolio_value_strategy'] + metrics_dict['cash_strategy']) / (self.initial_shares*self.start_price)*100
        metrics_dict['max_invest'] = meta_data['Cash_S'].min()

        return metrics_dict

    def save_results(self, meta_data):
        """Save results to a CSV file."""
        output_filename = os.path.join(self.output_dir, os.path.splitext(self.filename)[0] + '_Output.csv')
        meta_data.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")

    def save_metrics_to_csv(self, metrics_list):
        """Save metrics from multiple files into a single CSV."""
        # Convert the list of dictionaries to a DataFrame

        output_filename = os.path.join(self.output_dir, '_metrics.csv')
        metrics_df = pd.DataFrame(metrics_list)
        # Save to CSV
        metrics_df.to_csv(output_filename, index=False)

        output_filename = os.path.join(self.output_dir, '_cashflow.csv')
        df = pd.DataFrame(list(self.total_cash.items()), columns=['Date', 'Value'])
        # Save to CSV
        df.to_csv(output_filename, index=False)


    def print_data(self,mdict):
        message = ""

        message += f"Final Portfolio Results for %{self.filename}\n"

        message += f"Initial Investment: ${mdict['initial_invest']:,.0f}\n"
        message += f"Final Portfolio Value (Hold): ${mdict['portfolio_value_hold']:,.0f} (Cash: ${mdict['cash_hold']:.0f})\n"
        message += f"Final Portfolio Value (Monthly): ${mdict['portfolio_value_monthly']:,.0f} (Cash: ${mdict['cash_monthly']:.0f})\n"
        message += f"Final Portfolio Value (Strategy): ${mdict['portfolio_value_strategy']:,.0f} (Cash: ${mdict['cash_strategy']:.0f})\n"

        message += f"Return (%) (Hold): {mdict['return_hold']:.2f}%\n"
        message += f"Return (%) (Monthly): {mdict['return_monthly']:.2f}%\n"
        message += f"Return (%) (Strategy): {mdict['return_strategy']:.2f}%\n"

        message += f"Cash Gain (Hold): {mdict['portfolio_value_hold'] + mdict['cash_hold']:,.0f}\n"
        message += f"Cash Gain (Monthly): {mdict['portfolio_value_monthly'] + mdict['cash_monthly']:,.0f}\n"
        message += f"Cash Gain (Strategy): {mdict['portfolio_value_strategy'] + mdict['cash_strategy']:,.0f}\n"

        # The message variable now contains the complete output
        print(message)  # Optionally print the final message if needed
        return message



    def draw_plots(self, meta_data, message):
        """Generate and save plots."""
        meta_data['Total_Value_S'] = meta_data['Cash_S'] + meta_data['Portfolio_S'] * meta_data['Close']

        file_root, file_ext = os.path.splitext(self.filename)

        # Calculate Total Values
        meta_data['Total_Value_H'] = meta_data['Cash_H'] + meta_data['Portfolio_H'] * meta_data['Close']
        meta_data['Total_Value_M'] = meta_data['Cash_M'] + meta_data['Portfolio_M'] * meta_data['Close']
        meta_data['Total_Value_S'] = meta_data['Cash_S'] + meta_data['Portfolio_S'] * meta_data['Close']

        # Get the maximum cash value for Cash_S
        max_cash = meta_data['Cash_S'].max()
        max_date = meta_data.loc[meta_data['Cash_S'].idxmax(), 'Date']  # Corresponding date

        plt.figure(figsize=(20, 12))
        plt.plot(meta_data['Date'], meta_data['Cash_H'], label='Cash Hold')
        plt.plot(meta_data['Date'], meta_data['Cash_M'], label='Cash Monthly')
        plt.plot(meta_data['Date'], meta_data['Cash_S'], label='Cash Strategy')
        plt.title(f'{self.filename}: Cash Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cash ($)')
        plt.legend()
        plt.figtext(
            0.7, 0.8,  # x and y positions in figure coordinates (0 to 1 scale)
            message,
            wrap=False, horizontalalignment='left',
            fontsize=10, bbox=dict(facecolor='lightgrey', alpha=0.8, pad=10),
            fontdict={'family': 'monospace'}
        )

        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir,'_plots', file_root + 'cash.png'))
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(meta_data['Date'], meta_data['Total_Value_H'], label='Total Value Hold')
        plt.plot(meta_data['Date'], meta_data['Total_Value_M'], label='Total Value Monthly')
        plt.plot(meta_data['Date'], meta_data['Total_Value_S'], label='Total Value Strategy')
        plt.title(f'{self.filename}: Total Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir,'_plots', file_root + 'value.png'))
        plt.close()

    def run(self, file_list):
        """Run the simulation for all files in the file list."""
        for filename in file_list:
            meta_data = self.load_data(filename)
            if meta_data is None:
                continue

            start_date, end_date = self.validate_date(meta_data)

            meta_data = meta_data[
                                (meta_data['Date'] >= start_date) & (meta_data['Date'] <= end_date)
                                ].reset_index(drop=True)

            week_high_str = f'{self.params["hi_lo_weeks"]}_Week_High'
            week_low_str = f'{self.params["hi_lo_weeks"]}_Week_Low'
            weeks_as_days = self.params["hi_lo_weeks"]*5

            meta_data[week_high_str] = meta_data['High'].shift(20).rolling(weeks_as_days).max() * self.params['high_sell_factor']
            meta_data[week_low_str] = meta_data['Low'].shift(20).rolling(weeks_as_days).min()

            if start_date in meta_data['Date'].values:
              self.start_price = meta_data.loc[meta_data['Date'] == start_date, 'Close'].values[0]
              self.initial_shares = int(self.initial_invest/self.start_price)
            else:
                print(f"Start date {self.start_date} not found in meta_data.")
                continue

            meta_data['Cash_H'] = float(self.initial_shares * self.start_price *-1)
            meta_data['Portfolio_H'] = float(self.initial_shares)
            meta_data['Cash_M'] = float(self.initial_shares * self.start_price *-1)
            meta_data['Portfolio_M'] = float(self.initial_shares)
            meta_data['Cash_S'] = float(self.initial_shares * self.start_price *-1)
            meta_data['Portfolio_S'] = float(self.initial_shares)

            self.apply_strategy(meta_data)
            self.apply_monthly_plan(meta_data)
            self.apply_hold_interest(meta_data)
            metrics_dict = self.calculate_metrics(meta_data)
            self.metrics_data.append(metrics_dict)

            message = self.print_data(metrics_dict)
            self.save_results(meta_data)
            self.draw_plots(meta_data, message)

        self.save_metrics_to_csv(self.metrics_data)



params = {
    'sell_high': True,
    'buy_low': True,
    'relative_sell': 30,
    'relative_buy': 50,
    'cash_interest': 0.12/252,
    'high_sell_factor': 1.015, # how much above 52 week high in order to sell
    'keep_minimum': 0.5,
    'invest_cap': -250000,
    'total_cap': -500000,
    'hi_lo_weeks': 52
}


ci_test = True

source_dir = "history"
output_dir = "output"


if ci_test:
    source_dir = "ci/input"
    output_dir = "ci/output"
    plot_dir = "ci/plots"


# Use glob to find all CSV files in the specified directory
file_list = glob.glob(os.path.join(source_dir, '*.csv'))
file_list = [os.path.basename(file) for file in file_list]

simulator = TradingStrategySimulator(initial_invest=5000,
                                    start_date="2022-07-03",
                                    period_months = 60,
                                    plan=1,
                                    source_dir = source_dir,
                                    output_dir = output_dir,
                                    params=params)
simulator.run(file_list)
