import numpy as np

def simulate_trades(predicted_close_prices, actual_close_prices, balance=10000.00, transaction_fee_percentage=0.1 / 100,
                    bid_ask_spread_percentage=0.1 / 100, slippage_percentage=0.05 / 100, stop_loss_percentage=5 / 100,
                    take_profit_percentage=10 / 100, cooldown_period_days=5):
    """
    Simulates trades based on predicted and actual prices with transaction fees, bid-ask spread, slippage,
    stop-loss, and take-profit.

    :param predicted_close_prices: array of predicted close prices
    :param actual_close_prices: array of actual close prices
    :param balance: initial balance for trading
    :param transaction_fee_percentage: percentage of transaction fees (default: 0.1%)
    :param bid_ask_spread_percentage: percentage of bid-ask spread (default: 0.1%)
    :param slippage_percentage: percentage of slippage (default: 0.05%)
    :param stop_loss_percentage: percentage for stop-loss (default: 5%)
    :param take_profit_percentage: percentage for take-profit (default: 10%)
    :param cooldown_period_days: days to wait between trades (default: 5 days)

    :return: final balance after trades, total profit or loss
    """

    # Initialize variables
    shares_held = 0
    last_trade_day = 0
    entry_price = None

    # Define helper functions
    def apply_transaction_fee(balance, transaction_value):
        fee = transaction_value * transaction_fee_percentage
        return balance - fee

    def apply_bid_ask_spread(price, is_buy):
        if is_buy:
            return price * (1 + bid_ask_spread_percentage)
        else:
            return price * (1 - bid_ask_spread_percentage)

    def apply_slippage(price):
        slippage = price * np.random.uniform(-slippage_percentage, slippage_percentage)
        return price + slippage

    def should_exit_trade(entry_price, current_price):
        if entry_price is None:
            return 'hold'
        percentage_change = (current_price - entry_price) / entry_price * 100
        if percentage_change <= -stop_loss_percentage:
            return 'stop_loss'
        elif percentage_change >= take_profit_percentage:
            return 'take_profit'
        return 'hold'

    def can_trade(current_day, last_trade_day):
        return current_day - last_trade_day >= cooldown_period_days

    # Simulate trades
    for day in range(len(predicted_close_prices)):
        current_price = predicted_close_prices[day]
        actual_price = actual_close_prices[day]

        # Simulate a buy based on cooldown period
        if shares_held == 0 and can_trade(day, last_trade_day):
            # Buy as many shares as possible
            shares_to_buy = balance // current_price
            if shares_to_buy > 0:
                shares_held += shares_to_buy
                balance -= shares_to_buy * apply_bid_ask_spread(current_price, is_buy=True)
                balance = apply_transaction_fee(balance, shares_to_buy * current_price)
                entry_price = actual_price
                last_trade_day = day
                print(f"Day {day + 1}: Buying {shares_to_buy} shares at {float(current_price):.2f}. Balance: {float(balance):.2f}, Shares held: {shares_held}")

        # If holding shares, check exit condition
        if shares_held > 0:
            exit_signal = should_exit_trade(entry_price, current_price)
            if exit_signal in ['stop_loss', 'take_profit']:
                # Sell all shares
                sell_price = apply_bid_ask_spread(apply_slippage(current_price), is_buy=False)
                balance += shares_held * sell_price
                balance = apply_transaction_fee(balance, shares_held * sell_price)
                print(f"Day {day + 1}: Selling {shares_held} shares at {float(sell_price):.2f}. Balance: {float(balance):.2f}, Shares held: {shares_held}")
                shares_held = 0
                last_trade_day = day

    # Final summary
    final_balance = balance
    total_profit_loss = final_balance - balance
    print(f"\nInitial balance: {float(balance):.2f}")
    print(f"Final balance: {float(final_balance):.2f}")
    print(f"Total profit or loss: {float(total_profit_loss):.2f}")

    return final_balance, total_profit_loss