# monitoring/performance_dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import sqlite3

def create_dashboard():
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("AI Intraday Trading Dashboard"),
        dcc.Graph(id='equity-curve'),
        dcc.Interval(id='interval', interval=60*1000),  # Update každú minútu
        html.Div([
            html.Div(id='current-stats', className='row'),
            dcc.Graph(id='symbol-performance')
        ])
    ])
    
    @app.callback(
        Output('equity-curve', 'figure'),
        [Input('interval', 'n_intervals')]
    )
    def update_equity_curve(n):
        conn = sqlite3.connect('trading_db.sqlite')
        df = pd.read_sql_query("SELECT * FROM equity_history", conn)
        conn.close()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['equity'],
            mode='lines',
            name='Equity'
        ))
        fig.update_layout(title='Vývoj účtu', yaxis_title='Equity')
        return fig
    
    @app.callback(
        Output('current-stats', 'children'),
        [Input('interval', 'n_intervals')]
    )
    def update_stats(n):
        conn = sqlite3.connect('trading_db.sqlite')
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC LIMIT 1", conn)
        conn.close()
        
        if df.empty:
            return html.Div("No trades yet")
        
        last_trade = df.iloc[0]
        return html.Div([
            html.Div(f"Posledný obchod: {last_trade['symbol']} {'BUY' if last_trade['direction'] == 1 else 'SELL'}"),
            html.Div(f"Veľkosť: {last_trade['volume']}"),
            html.Div(f"Výsledok: ${last_trade['profit']:.2f}")
        ])
    
    app.run_server(debug=True)

if __name__ == "__main__":
    create_dashboard()