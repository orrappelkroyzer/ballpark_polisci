
import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(add_date=False)
import pandas as pd
from datetime import datetime


def nc_curr_to_short_nc_curr():
    df = pd.read_csv(config['db_dir'] / "Dobbs" / "Voter registration DB Raw" / "NC" / "ncvoter_Statewide.txt", delimiter="\t", encoding="latin-1")
    df[['county_desc', 'ncid', 'last_name', 'first_name', 'middle_name', 'status_cd', 'zip_code', 'registr_dt', 'race_code', 
        'ethnic_code', 'party_cd', 'gender_code', 'age_at_year_end', 'birth_state', 'drivers_lic']]\
            .to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'current.csv')

def nc_hist_to_short_nc_hist():
    df = pd.read_csv(config['db_dir'] / "Voter registration DB Raw" / "NC" / "ncvhis_Statewide.txt", delimiter="\t", encoding="latin-1")
    df[['election_desc', 'voting_method', 'voted_party_cd', 'ncid']]\
        .to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist.csv')

def create_db():
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist.csv').drop(columns=['Unnamed: 0'])
    df = df.join(df['election_desc'].str.split(" ", n=1, expand=True).rename(columns = {0 : 'date', 1 : 'type'}))
    df = df[df['type'] == 'GENERAL']
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
    df = df.set_index(['ncid', 'date'])['voted_party_cd'].unstack()
    df.columns.name = None