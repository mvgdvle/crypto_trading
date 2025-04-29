from gym_trading_env.downloader import download
import datetime


download(
    exchange_names = ['binance'],
    symbols = ['BTC/USDT'],
    timeframe = '1h',
    dir = 'data',
    since= datetime.datetime(year=2020, month=1, day=1),
    until = datetime.datetime(year= 2024, month=12, day=31)
)
