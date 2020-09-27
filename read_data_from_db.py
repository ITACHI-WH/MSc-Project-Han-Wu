%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import os

import json
import pandas as pd

from sqlalchemy.types import TIMESTAMP as typeTIMESTAMP

from dotenv import load_dotenv

load_dotenv(verbose=True)

import math
import logging
logger = logging.getLogger(__name__)

import dateutil
from sqlalchemy import create_engine

import os
REMOTE_HOST=os.getenv('REMOTE_HOST')
REMOTE_DB_PASS=os.getenv('REMOTE_DB_PASS')
REMOTE_USER=os.getenv('REMOTE_USER')
DB_NAME=os.getenv('DB_NAME')
print(REMOTE_HOST, REMOTE_DB_PASS, REMOTE_USER, DB_NAME)

REMOTE_HOST='mqtt.spe-hub.net'
REMOTE_DB_PASS='o+Al<FqDANu_'
#REMOTE_DB_PASS='aaa'
REMOTE_USER='pm'
DB_NAME='ha'

engine = create_engine(f'postgresql+psycopg2://{REMOTE_USER}:{REMOTE_DB_PASS}@{REMOTE_HOST}/{DB_NAME}', server_side_cursors=True)
last_hours = 24
df = pd.read_sql_query(f"""
SELECT
       to_timestamp(event_data::json -> 'new_state' ->> 'last_changed','YYYY-MM-DD"T"HH24:MI:SS.US') as last_changed,
       event_data::json -> 'new_state' -> 'entity_id' as entity_id,
       event_data::json -> 'new_state' -> 'attributes' -> 'node_id' as node_id,
       event_data::json -> 'new_state' -> 'attributes' -> 'power_consumption' as power_consumption,
       event_data::json -> 'new_state' -> 'attributes' -> 'unit_of_measurement' as unit
FROM  events
WHERE event_type like 'state_changed'
AND event_data::json ->> 'entity_id' = 'sensor.aeon_labs_zw096_smart_switch_6_power_2'
-- ORDER BY last_changed DESC limit 3600 * {last_hours}
;
""", con=engine)
df['last_changed'] = pd.to_datetime(df['last_changed'], errors='coerce')
df.dropna(subset=['last_changed'], inplace=True)
df.set_index('last_changed', inplace=True)
start_date = '2020-09-05'
end_date = '2020-10-01'
#start_time = '00:00'
#end_time = '23:59'
df
subset_df = df.loc[start_date:end_date].between_time(start_time, end_time)
subset_df.to_csv('power.csv')