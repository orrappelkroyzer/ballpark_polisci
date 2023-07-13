
import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(add_date=False)
import pandas as pd
from datetime import datetime, date


def nc_curr_to_short_nc_curr():
    df = pd.read_csv(config['db_dir'] /  "db_creation_input" / "Dobbs"/ "Voter registration DB Raw" / "NC" / "ncvoter_Statewide.txt", delimiter="\t", encoding="latin-1")
    df[['county_desc', 'ncid', 'last_name', 'first_name', 'middle_name', 'status_cd', 'zip_code', 'registr_dt', 'race_code', 
        'ethnic_code', 'party_cd', 'gender_code', 'age_at_year_end', 'birth_state', 'drivers_lic']]\
            .to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'current.csv')

def nc_hist_to_short_nc_hist():
    df = pd.read_csv(config['db_dir'] /  "db_creation_input" / "Dobbs"/ "Voter registration DB Raw" / "NC" / "ncvhis_Statewide.txt", delimiter="\t", encoding="latin-1")
    df[['election_desc', 'voting_method', 'voted_party_cd', 'ncid']]\
        .to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist.csv')

def create_hist_party_df():
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist.csv').drop(columns=['Unnamed: 0'])
    df = df.join(df['election_desc'].str.split(" ", n=1, expand=True).rename(columns = {0 : 'date', 1 : 'type'}))
    df = df[df['type'] == 'GENERAL']
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
    df = df.set_index(['ncid', 'date'])['voted_party_cd'].unstack()
    df.columns.name = None
    df.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party.csv')
    return df

def create_switches_df():
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party.csv')
    df = df.set_index('ncid')
    df = df.apply(lambda s: s.fillna(method='ffill'), axis=1)
    df1 = df.apply(lambda s: s[s.notnull() & s.shift().notnull()] != s.shift()[s.notnull() & s.shift().notnull()], axis=1)
    df1 = df1.fillna(False)
    df1.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches.csv')
    t = {}
    for year in range(2014, 2023, 2):
        logger.info(f"Processing year {year}")
        t[year] = df1[[x for x in df1.columns if date(year-1,1,1) <= x <= date(year, 12, 31)]].sum(axis=1)
    df2 = pd.DataFrame(t)
    df2.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches_by_election_year.csv')
    t = {}
    for year in range(2016, 2021, 4):
        logger.info(f"Processing year {year}")
        t[year] = df1[[x for x in df1.columns if date(year-3,1,1) <= x <= date(year, 12, 31)]].sum(axis=1)
    df3 = pd.DataFrame(t)
    df3.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches_by_presidential_election_year.csv')

