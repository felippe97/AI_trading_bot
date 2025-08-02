# monitoring/drawdown_alert.py
import smtplib
from email.mime.text import MIMEText
from config import MAX_DAILY_DRAWDOWN
import smtplib
from email.mime.text import MIMEText
def send_alert_email(drawdown_pct, max_drawdown=MAX_DAILY_DRAWDOWN):
    if drawdown_pct < max_drawdown:
        return
    
    # Nastavenie emailu
    sender = "alert@yourdomain.com"
    receivers = ["your_email@example.com"]
    subject = f"ALERT: Drawdown dosiahol {drawdown_pct}%"
    body = f"Denný drawdown vášho účtu dosiahol {drawdown_pct}%, čo je viac ako povolených {max_drawdown}%."
    
    message = MIMEText(body, 'plain')
    message['Subject'] = subject
    message['From'] = sender
    message['To'] = ", ".join(receivers)
    
    try:
        with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
            server.starttls()
            server.login(sender, "your_password")
            server.send_message(message)
        print("Alert email odoslaný")
    except Exception as e:
        print(f"Chyba pri odosielaní emailu: {e}")