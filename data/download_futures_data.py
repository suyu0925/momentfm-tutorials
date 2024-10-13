# %%
import os  # noqa
import sys  # noqa

# add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# %%
from config.config import config  # noqa

print(config['jqdata'])

# %%
import jqdatasdk as jq  # noqa

jq.auth(config["jqdata"]["username"], config["jqdata"]["password"])

# 获取收盘价
df = jq.get_price(['RB9999.XSGE', 'I9999.XDCE'], start_date='2023-01-01', end_date='2023-12-31', frequency='1m', fields=[
    'close'], skip_paused=False, fq='pre', panel=False)

jq.logout()

# %%
df = df.pivot(index='time', columns='code', values='close')
df['iossr'] = df['RB9999.XSGE'] / df['I9999.XDCE']  # Iron Ore to Steel Scrap Ratio
df.to_csv('../data/iossr.csv')
df
