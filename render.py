from gym_trading_env.renderer import Renderer
import pandas as pd

render_logs_dir = "render_logs/basic/no_fee/A2C/A2C50"

renderer = Renderer(render_logs_dir=render_logs_dir)
renderer.add_line( name= "SMA 48", function= lambda df : df["close"].rolling(48).mean(), line_options ={"width" : 1, "color": "black"})
renderer.add_line( name= "SMA 168", function= lambda df : df["close"].rolling(168).mean(), line_options ={"width" : 1, "color": "blue"})

renderer.run()
