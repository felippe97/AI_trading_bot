# execution/risk_manager.py
from config import RISK_PER_TRADE, MAX_DAILY_DRAWDOWN

class RiskManager:
    def __init__(self):
        self.initial_balance = None
        self.daily_drawdown_limit = MAX_DAILY_DRAWDOWN
        self.risk_per_trade = RISK_PER_TRADE
    
    def set_initial_balance(self, balance):
        self.initial_balance = balance
    
    def check_daily_drawdown(self, current_equity):
        if self.initial_balance is None:
            return False
        drawdown_pct = ((self.initial_balance - current_equity) / self.initial_balance) * 100
        return drawdown_pct >= self.daily_drawdown_limit

def calculate_position_size(balance, risk_percent, stop_loss_points, pip_value):
    risk_amount = balance * (risk_percent / 100)
    risk_per_pip = pip_value * stop_loss_points
    return round(risk_amount / risk_per_pip, 2)