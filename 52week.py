import pandas as pd
import os

# Global variable to define monthly transaction plan for "Hold" strategy
plan = 1  # Number of shares to buy (positive) or sell (negative) monthly

# Global variables to control the trading strategy
sell_high = True  # Defines if we sell when reaching above 52-week high
buy_low = True    # Defines if we buy when reaching below 52-week low
sell_amount = 5  # Number of shares to sell when conditions are met
buy_amount = 10   # Number of shares to buy when conditions are met

relative_transaction = True
relative_sell = 10
relative_buy = 50
cash_interest = 0.12/252 # Assume that free money can generate up to 12% elsewhere
tax_rate = 0.8 # 20% tax on capital gains

# Initialize portfolio simulation
initial_shares = 100
start_date = pd.to_datetime('2018-01-03')



# Define a function to load "Meta_Dataset.csv" from the disk
def load_meta_dataset(filename):
    """
    Load the Meta historical dataset from the same directory as the script.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data from "Meta_Dataset.csv".
    """
    filepath = os.path.join("history/", filename )
    print("Loading file:", filepath)
    try:
        # Attempt to load the dataset
        dataset = pd.read_csv(filepath)
        # Convert the 'Date' column to datetime format
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        # Sort the dataset by the 'Date' column in ascending order
        dataset = dataset.sort_values(by='Date', ascending=True).reset_index(drop=True)

        return dataset
    except FileNotFoundError:
        print("Error: ' " + filename + ".csv' not found in the script's directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None



def apply_monthly_plan(meta_data, initial_shares, plan):
    """
    Adjust the "Hold" portfolio based on the monthly plan.
    Buys or sells the defined number of shares on the 15th of each month.

    Args:
        meta_data (pd.DataFrame): The stock data.
        initial_shares (int): Initial number of shares in the portfolio.
        plan (int): Number of shares to buy (positive) or sell (negative) monthly.

    Returns:
        dict: Final portfolio value and transactions.
    """
    portfolio = {'cash': initial_shares * start_price*-1, 'shares': initial_shares, 'transactions': []}
    for idx, row in meta_data.iterrows():
        if row['Date'].day == 15:  # Execute the plan on the 15th of each month
            if plan < 0 and portfolio['shares'] >= abs(plan):  # Sell shares
                portfolio['cash'] += abs(plan) * row['Close']
                portfolio['shares'] += plan  # Subtract from shares
                portfolio['transactions'].append({'Date': row['Date'], 'Action': 'Sell', 'Price': row['Close'], 'Amount': abs(plan)})
            elif plan > 0:  # Buy shares
                portfolio['cash'] -= plan * row['Close']
                portfolio['shares'] += plan  # Add to shares
                portfolio['transactions'].append({'Date': row['Date'], 'Action': 'Buy', 'Price': row['Close'], 'Amount': plan})
        meta_data.loc[idx, 'Cash_M'] = portfolio['cash']
        meta_data.loc[idx, 'Portfolio_M'] = portfolio['shares']
        portfolio['cash'] = portfolio['cash'] + portfolio['cash']*cash_interest
    # Calculate the final value of the portfolio
    final_closing_price = meta_data.iloc[-1]['Close']
    portfolio_value = portfolio['cash'] + portfolio['shares'] * final_closing_price
    return {
        'Final Portfolio Value': portfolio_value,
        'Transactions': portfolio['transactions']
    }

def apply_hold_interest(meta_data):
    """

    """
    portfolio = {'cash': initial_shares * start_price*-1, 'shares': initial_shares, 'transactions': []}
    for idx, row in meta_data.iterrows():

        meta_data.loc[idx, 'Cash_H'] = portfolio['cash']
        portfolio['cash'] = portfolio['cash'] + portfolio['cash']*cash_interest



def apply_strategy(meta_data):
    """
    Apply strategic plan

    """
    # Simulate the trading strategy
    outside_range = False  # Flag to track if we're waiting for the price to return inside the range

    # Initialize a variable to track the last transaction month
    last_transaction_month = None

    for idx, row in meta_data.iterrows():
        current_month = row['Date'].month
        current_year = row['Date'].year

        if not outside_range and (last_transaction_month != (current_year, current_month)):
            if sell_high and row['High'] > row['52_Week_High'] and portfolio['shares'] >= sell_amount:
                # Sell 'sell_amount' shares at the day's high
                if relative_transaction:
                    sell_amount = portfolio['shares'] * relative_sell / 100
                portfolio['cash'] += sell_amount * row['High'] * tax_rate
                portfolio['shares'] -= sell_amount
                portfolio['transactions'].append({'Date': row['Date'], 'Action': 'Sell', 'Price': row['High'], 'Amount': sell_amount})
                outside_range = True
                last_transaction_month = (current_year, current_month)
            elif buy_low and row['Low'] < row['52_Week_Low']:
                # Buy 'buy_amount' shares at the day's low
                if relative_transaction:
                    buy_amount = portfolio['shares'] * relative_buy / 100
                portfolio['cash'] -= buy_amount * row['Low']
                portfolio['shares'] += buy_amount
                portfolio['transactions'].append({'Date': row['Date'], 'Action': 'Buy', 'Price': row['Low'], 'Amount': buy_amount})
                outside_range = True
                last_transaction_month = (current_year, current_month)

        else:
            # Reset the flag once the price moves back into the range
            if row['52_Week_Low'] <= row['Low'] <= row['High'] <= row['52_Week_High']:
                outside_range = False

        # Update the portfolio and cash data in meta_data
        meta_data.loc[idx, 'Cash_S'] = portfolio['cash']
        meta_data.loc[idx, 'Portfolio_S'] = portfolio['shares']

        # Apply cash interest to the portfolio's cash
        portfolio['cash'] = portfolio['cash'] + portfolio['cash'] * cash_interest


def calculate_output(meta_data):
    # Calculate the final portfolio value
    final_closing_price = meta_data.iloc[-1]['Close']
    # Integrate the monthly adjustment for the "Hold" approach
    hold_monthly_results = apply_monthly_plan(meta_data, initial_shares, plan)
    apply_hold_interest(meta_data)

    portfolio_value_hold = meta_data.iloc[-1]['Portfolio_H'] * final_closing_price
    portfolio_value_monthly = meta_data.iloc[-1]['Portfolio_M'] * final_closing_price
    portfolio_value_strategy = meta_data.iloc[-1]['Portfolio_S'] * final_closing_price

    cash_hold = meta_data.iloc[-1]['Cash_H']
    cash_monthly = meta_data.iloc[-1]['Cash_M']
    cash_strategy = meta_data.iloc[-1]['Cash_S']


    # Calculate returns
    return_hold = (portfolio_value_hold + cash_hold) / (initial_shares*start_price)*100
    return_monthly = (portfolio_value_monthly + cash_monthly) / (initial_shares*start_price)*100
    return_strategy = (portfolio_value_strategy + cash_strategy) / (initial_shares*start_price)*100

def print_data(meta_data):
    print("Final Portfolio Results:")

    print(f"Initial Investment: ${initial_shares*start_price:,.0f}")
    print(f"Final Portfolio Value (Hold): ${portfolio_value_hold:,.0f} (Cash: ${cash_hold:.0f})")
    print(f"Final Portfolio Value (Monthly): ${portfolio_value_monthly:,.0f} (Cash: ${cash_monthly:.0f})")
    print(f"Final Portfolio Value (Strategy): ${portfolio_value_strategy:,.0f} (Cash: ${cash_strategy:.0f})")

    print(f"Return (%) (Hold): {return_hold:.2f}%")
    print(f"Return (%) (Monthly): {return_monthly:.2f}%")
    print(f"Return (%) (Strategy): {return_strategy:.2f}%")

    print(f"Cash Gain (Hold): {portfolio_value_hold + cash_hold:,.0f}")
    print(f"Cash Gain (Monthly): {portfolio_value_monthly + cash_monthly:,.0f}")
    print(f"Cash Gain (Strategy): {portfolio_value_strategy + cash_strategy:,.0f}")


def draw_plots(meta_data):
    import matplotlib.pyplot as plt
    # Calculate Total Values
    meta_data['Total_Value_H'] = meta_data['Cash_H'] + meta_data['Portfolio_H'] * meta_data['Close']
    meta_data['Total_Value_M'] = meta_data['Cash_M'] + meta_data['Portfolio_M'] * meta_data['Close']
    meta_data['Total_Value_S'] = meta_data['Cash_S'] + meta_data['Portfolio_S'] * meta_data['Close']

    # Plot Cash
    plt.figure(figsize=(12, 6))
    plt.plot(meta_data['Date'], meta_data['Cash_H'], label='Cash_H')
    plt.plot(meta_data['Date'], meta_data['Cash_M'], label='Cash_M')
    plt.plot(meta_data['Date'], meta_data['Cash_S'], label='Cash_S')
    plt.title('Cash Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cash ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Total Value
    plt.figure(figsize=(12, 6))
    plt.plot(meta_data['Date'], meta_data['Total_Value_H'], label='Total Value H')
    plt.plot(meta_data['Date'], meta_data['Total_Value_M'], label='Total Value M')
    plt.plot(meta_data['Date'], meta_data['Total_Value_S'], label='Total Value S')
    plt.title('Total Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()




file_list = ["AAPL-history.csv",
"DIS-history.csv",
"DUOL-history.csv",
"GOOG-history.csv",
"MSFT-history.csv",
"NKE-history.csv",
"NVDA-history.csv",
"RDDT-history.csv",
"TSLA-history.csv",]

for filename in file_list:


    # Example usage
    # Uncomment the following line to load the dataset:
    meta_data = load_meta_dataset(filename)
    # Convert the 'Date' column to datetime for easier filtering
    meta_data['Date'] = pd.to_datetime(meta_data['Date'])

    # Initialize the Cash and Portfolio columns in meta_data
    if start_date in meta_data['Date'].values:
        start_price = meta_data.loc[meta_data['Date'] == start_date, 'Close'].values[0]
    else:
        print(f"Start date {start_date} not found in meta_data.")
        exit()


    meta_data['Cash_H'] = initial_shares * start_price *-1
    meta_data['Portfolio_H'] = initial_shares
    meta_data['Cash_M'] = initial_shares * start_price *-1
    meta_data['Portfolio_M'] = initial_shares
    meta_data['Cash_S'] = initial_shares * start_price *-1
    meta_data['Portfolio_S'] = initial_shares


    # Filter data starting from start_date
    meta_data = meta_data[meta_data['Date'] >= start_date].reset_index(drop=True)

    # Calculate rolling 52-week high and low for the *previous day*
    meta_data['52_Week_High'] = meta_data['High'].shift(1).rolling(window=260, min_periods=1).max()
    meta_data['52_Week_Low'] = meta_data['Low'].shift(1).rolling(window=260, min_periods=1).min()


    cash = initial_shares * start_price*-1  # Start with no additional cash
    portfolio = {
        'cash': cash,
        'shares': initial_shares,
        'transactions': [],  # To record buy/sell actions
    }

    # Add a flag to indicate if we're outside the 52-week range
    meta_data['Outside_Range'] = (
        (meta_data['High'] > meta_data['52_Week_High']) |
        (meta_data['Low'] < meta_data['52_Week_Low'])
    )

    apply_strategy(meta_data)
    calculate_output(meta_data)
    print_data(meta_data)

    meta_data.to_csv(filename + "_Output.csv", index=False)

    draw_plots(meta_data)
