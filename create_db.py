
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
    logger.info("Reading hist.csv")
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist.csv').drop(columns=['Unnamed: 0'])
    logger.info("Splitting election_desc into date and type")
    df = df.join(df['election_desc'].str.split(" ", n=1, expand=True).rename(columns = {0 : 'date', 1 : 'type'}))
    logger.info("Dropping primary elections")
    df = df[df['type'] == 'GENERAL']
    logger.info("Fixing date")
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
    party = df.set_index(['ncid', 'date'])['voted_party_cd'].unstack()
    party.columns.name = None
    party.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party.csv')

def create_hist_vm_df():
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist.csv').drop(columns=['Unnamed: 0'])
    df = df.join(df['election_desc'].str.split(" ", n=1, expand=True).rename(columns = {0 : 'date', 1 : 'type'}))
    logger.info("Dropping primary elections")
    df = df[df['type'] == 'GENERAL']
    logger.info("Fixing date")
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').date())
    logger.info("removing unrecognised values")
    vm = df[df['voting_method'].isin(['ABSENTEE ONESTOP', 'IN-PERSON', 'ABSENTEE BY MAIL', 'ABSENTEE CURBSIDE', 'CRBSIDE'])]
    vm = vm.drop_duplicates(['ncid', 'date'])
    vm = vm.set_index(['ncid', 'date'])['voting_method'].unstack()
    vm.columns.name = None
    logger.info("Writing to file")
    vm.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_voting_method.csv')

def create_filled_hist_party_df():
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party.csv')
    df = df.set_index('ncid')
    df = df.apply(lambda s: s.fillna(method='ffill'), axis=1)
    df.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party_filled.csv')

def create_filled_hist_vm_df():
    logger.info("Reading hist_voting_method.csv")
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_voting_method.csv')
    logger.info("Setting index to ncid")
    df = df.set_index('ncid')
    logger.info("Filling in missing values")
    df = df.apply(lambda s: s.fillna(method='ffill'), axis=1)
    logger.info(f"Writing to file {config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_voting_method_filled.csv'}")
    df.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_voting_method_filled.csv')


def create_switches_df():
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party_filled.csv')
    df = df.replace({'DEM' : -1, 'REP' : 1, 'LIB' : 1, 'GRE' : -1, 'CST' : 0, 'UNA' : 0})
    df1 = df.apply(lambda row: row-row.shift(), axis=1)
    df1 = df1.fillna(0)
    df1.columns = [date.fromisoformat(x) for x in df1.columns]
    t = {}
    for year in range(2016, 2021, 4):
        logger.info(f"Processing year {year}")
        t[year] = df1[[x for x in df1.columns if date(year-3,1,1) <= x <= date(year, 12, 31)]].sum(axis=1)
    df3 = pd.DataFrame(t)
    df3.to_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches_by_presidential_election_year.csv')

    
counties_dict = {37001: 'Piedmont',
 37003: 'Piedmont',
 37005: 'Mountains',
 37007: 'Piedmont',
 37009: 'Mountains',
 37011: 'Mountains',
 37013: 'Coastal Plains',
 37015: 'Coastal Plains',
 37017: 'Coastal Plains',
 37019: 'Coastal Plains',
 37021: 'Mountains',
 37023: 'Mountains',
 37025: 'Piedmont',
 37027: 'Mountains',
 37029: 'Coastal Plains',
 37031: 'Coastal Plains',
 37033: 'Piedmont',
 37035: 'Piedmont',
 37037: 'Piedmont',
 37039: 'Mountains',
 37041: 'Coastal Plains',
 37043: 'Mountains',
 37045: 'Piedmont',
 37047: 'Coastal Plains',
 37049: 'Coastal Plains',
 37051: 'Coastal Plains',
 37053: 'Coastal Plains',
 37055: 'Coastal Plains',
 37057: 'Piedmont',
 37059: 'Piedmont',
 37061: 'Coastal Plains',
 37063: 'Research Triangle',
 37065: 'Coastal Plains',
 37067: 'Greensboro-Winston-Salem',
 37069: 'Piedmont',
 37071: 'Piedmont',
 37073: 'Coastal Plains',
 37075: 'Coastal Plains',
 37077: 'Piedmont',
 37079: 'Coastal Plains',
 37081: 'Greensboro-Winston-Salem',
 37083: 'Coastal Plains',
 37085: 'Coastal Plains',
 37087: 'Mountains',
 37089: 'Mountains',
 37091: 'Coastal Plains',
 37093: 'Coastal Plains',
 37095: 'Coastal Plains',
 37097: 'Piedmont',
 37099: 'Mountains',
 37101: 'Coastal Plains',
 37103: 'Coastal Plains',
 37105: 'Piedmont',
 37107: 'Coastal Plains',
 37109: 'Piedmont',
 37111: 'Mountains',
 37113: 'Mountains',
 37115: 'Mountains',
 37117: 'Coastal Plains',
 37119: 'Charlotte',
 37121: 'Mountains',
 37123: 'Piedmont',
 37125: 'Piedmont',
 37127: 'Coastal Plains',
 37129: 'Coastal Plains',
 37131: 'Coastal Plains',
 37133: 'Coastal Plains',
 37135: 'Piedmont',
 37137: 'Coastal Plains',
 37139: 'Coastal Plains',
 37141: 'Coastal Plains',
 37143: 'Coastal Plains',
 37145: 'Piedmont',
 37147: 'Coastal Plains',
 37149: 'Mountains',
 37151: 'Piedmont',
 37153: 'Piedmont',
 37155: 'Coastal Plains',
 37157: 'Piedmont',
 37159: 'Piedmont',
 37161: 'Mountains',
 37163: 'Coastal Plains',
 37165: 'Coastal Plains',
 37167: 'Piedmont',
 37169: 'Piedmont',
 37171: 'Piedmont',
 37173: 'Mountains',
 37175: 'Mountains',
 37177: 'Coastal Plains',
 37179: 'Piedmont',
 37181: 'Piedmont',
 37183: 'Research Triangle',
 37185: 'Piedmont',
 37187: 'Coastal Plains',
 37189: 'Mountains',
 37191: 'Coastal Plains',
 37193: 'Mountains',
 37195: 'Coastal Plains',
 37197: 'Piedmont',
 37199: 'Mountains'
}
    
    
def read_switches():
    logger.info("Reading switches")
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches_by_presidential_election_year.csv')
    return df.rename(columns={x : f'switch_{x}' for x in ['2016', '2020']})
    
def read_current_voter_roles():
    logger.info("Reading currrent voter rolls")
    df1 = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'current.csv')
    df1 = df1.drop(columns='Unnamed: 0')
    df1 = df1[df1['zip_code'].notnull()]
    df1['zip_code'] = df1['zip_code'].apply(int)
    df1.loc[df1['ethnic_code'] == 'HL', 'race_code'] = 'H'
    
    logger.info("Merging historic and current voter rolls")
    return df1

def read_fips():
    logger.info("Processing fips to name")
    fips2name = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'fips2name.csv')
    fips2name = fips2name[fips2name.State == 'North Carolina']
    fips2name['county_desc'] = fips2name['County'].str.upper()
    logger.info("Adding FIPS and region")
    fips2name = fips2name.join(pd.Series(counties_dict, name='Region'), on='FIPS') 
    return fips2name[['county_desc', 'FIPS', 'Region']]

def read_education(year=2016):
    ed = pd.read_csv(config['db_dir'] / 'db_creation_input' / "ACS" / "education" / f"ACSST5Y{year}.S1501-Data.csv")
    ed = ed.drop(0)
    ed['zip_code'] = ed['NAME'].str.split(" ").str[1].apply(int)
    races = ['W', 'B', 'I', 'A', 'P', 'O', 'M', 'H']
    t = []
    for gender, x in {'M' : '3', 'F' : '5'}.items():
        t1 = [pd.DataFrame({'College Educated' : ed[f'S1501_C0{x}_0{y+2}E'].apply(int), 
                            'Total' : ed[f'S1501_C0{x}_0{y}E'].apply(int), 
                            'zip_code' :ed['zip_code'],
                            'race_code' : races[i],  'gender_code' : gender}) 
                 for i, y in enumerate(range(31, 53, 3))]
            
        t1 += [pd.concat(t1).groupby('zip_code')[['College Educated', 'Total']].sum().reset_index().assign(race_code='U').assign(gender_code=gender)]
        t += [pd.concat(t1).reset_index(drop=True)]
    
    t += [pd.concat([(t[0]+t[1])[['College Educated', 'Total']], 
                   t[0][['race_code', 'zip_code']]], axis=1)\
        .groupby(['zip_code', 'race_code'])[['College Educated', 'Total']].sum().reset_index().assign(gender_code='U')]

    ed = pd.concat(t).reset_index(drop=True)
    ed[f'%_college_educated_{year}'] = (ed['College Educated']/ed['Total']).fillna(0)
    ed['zip_code'] = ed['zip_code'].apply(int)
    return ed[['zip_code', 'race_code', 'gender_code', f'%_college_educated_{year}']]

def read_median_household_income(year):
    logger.info("Adding zip code level median household income")
    mhh = pd.read_csv(config['db_dir'] / 'db_creation_input' / "ACS" / "median household income" / f"ACSST5Y{year}.S1901-Data.csv")
    mhh = mhh.drop(0)
    mhh['zip_code'] = mhh['NAME'].str.split(" ").str[1]
    field = f"median_household_income_{year}"
    mhh = mhh[['zip_code', 'S1901_C01_012E']].rename(columns={'S1901_C01_012E' : field})
    mhh = mhh[mhh[field] != 'N']
    mhh = mhh[mhh[field] != '-']
    mhh[field] = mhh[field].apply(int)
    mhh['zip_code'] = mhh['zip_code'].apply(int)
    return mhh

def read_voting_history():
    logger.info("Adding voting history")
    df1 = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_voting_method.csv')
    df1 = df1.set_index('ncid')[['2014-11-04', '2016-11-08', '2018-11-06', '2020-11-03', '2022-11-08']]
    df1 = pd.concat([df1.rename(columns={x: f'voting_method_{x.split("-")[0]}' for x in df1.columns}),  df1.notnull().applymap(int).rename(columns={x: f'voted_{x.split("-")[0]}' for x in df1.columns})], axis=1)
    return df1

def read_party_registration():
    logger.info("Adding historic party affiliation")
    df1 = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'hist_party_filled.csv')
    df1 = df1.set_index('ncid')[['2014-11-04', '2016-11-08', '2018-11-06', '2020-11-03', '2022-11-08']]
    df1.rename(columns={x: f'party_{x.split("-")[0]}' for x in df1.columns})
    df1 = df1.rename(columns={x: f'party_{x.split("-")[0]}' for x in df1.columns})
    df1 = df1.replace({'DEM' : -1, 'REP' : 1, 'LIB' : 1, 'GRE' : -1, 'CST' : 0, 'UNA' : 0})
    return df1

def create_db():
    df = read_current_voter_roles()
    df = df.merge(read_switches(), on='ncid')
    df = df.merge(read_fips(), on='county_desc')
    df = df.merge(read_education(2016), on=['zip_code', 'race_code', 'gender_code'])
    df = df.merge(read_education(2020), on=['zip_code', 'race_code', 'gender_code'])
    df = df.merge(read_median_household_income(2016), on='zip_code')   
    df = df.merge(read_median_household_income(2020), on='zip_code')   
    df = df.join(read_voting_history(), on='ncid')
    df = df.join(read_party_registration(), on='ncid')
    for year in ['2016', '2020']:
        for direction, switch in [('left', -1), ('right', 1)]:
            df.loc[:, f'switch_{year}_{direction}'] = (df[f'switch_{year}'] == switch).astype(int)
        df[f'D_{year}'] = (df[f'party_{year}'] != 1).apply(int)
        df[f'R_{year}'] = (df[f'party_{year}'] == 1).apply(int)
    logger.info("Writing to file")
    df.to_csv(config['db_dir'] / 'ballpark.csv', index=False)
    return df

if __name__ == '__main__':
    # create_hist_vm_df()
    # create_filled_hist_vm_df()
    create_db()