from ballpark_yhat_reg import solve_w_y

import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger, DATETIME
logger = get_logger(__name__)
config = load_config(add_date=DATETIME)
import pandas as pd
from datetime import datetime, date
import pandas as pd
from create_db_tools.processor_functions.static_and_demography import process_fips_to_name
import random

def read_voter_rolls():
    logger.info("Reading historic voter rolls")
    df = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches_by_presidential_election_year.csv')
    logger.info("Reading currrent voter rolls")
    df1 = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'current.csv')
    df1 = df1.drop(columns='Unnamed: 0')
    logger.info("Merging historic and current voter rolls")
    df = df1.merge(df, on='ncid')
    logger.info("Processing fips to name")
    fips2name = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'fips2name.csv')
    fips2name = fips2name[fips2name.State == 'North Carolina']
    fips2name['county_desc'] = fips2name['County'].str.upper()
    logger.info("Adding FIPS")
    df = df.merge(fips2name[['county_desc', 'FIPS']], on='county_desc')
    logger.info("Adding zip code level median household income")
    mhh = pd.read_csv(config['db_dir'] / 'db_creation_input' / "ACS", "median household income" , "ACSST5Y2021.S1901-Data.csv")
    mhh = mhh.drop(0)
    mhh['zip_code'] = mhh['NAME'].str.split(" ").str[1]
    mhh = mhh[['zip_code', 'S1901_C01_013E']].rename(columns={'S1901_C01_013E' : 'zip_median_household_income'})
    mhh = mhh[mhh['zip_median_household_income'] != 'N']
    mhh = mhh[mhh['zip_median_household_income'] != '-']
    mhh['zip_median_household_income'] = mhh['zip_median_household_income'].apply(int)
    mhh['zip_code'] = mhh['zip_code'].apply(int)
    df = df[df['zip_code'].notnull()]
    df['zip_code'] = df['zip_code'].apply(int)
    df = df.merge(mhh, on='zip_code')
    df.to_csv(config['db_dir'] / 'ballpark.csv', index=False)
    return df


def build_features_table(df):
    logger.info("creating race features")
    race_dict = {
        'A' : 'is_asian',
        'B' : 'is_black',
        'I' : 'is_native_american',
        'M' : 'is_multiracial',
        'O' : 'is_of_other_race',
        'P' : 'is_pacific_islander',
        'U' : 'is_undesignated_race',
        'W' : 'is_white'
    }
    race_features = pd.get_dummies(df['race_code']).drop(columns=['W', ' '], errors='ignore').rename(columns=race_dict)
    race_features['latino'] = df['ethnic_code'] == 'HL'
    logger.info("creating gender features")
    gender_dict = {
        'M' : 'is_male',
        'F' : 'is_female',
    }
    gender_features = pd.get_dummies(df['gender_code']).drop(columns=['U'],errors='ignore').rename(columns=gender_dict)
    logger.info("creating region features")
    state2reagion = {'NC' : 'North Carolina', 'NY' : 'North East', 'OC' : 'Rest of the World', 'VA' : 'South', 'CA' : 'West', 'FL' : 'South', 'PA' : 'North East', 'NJ' : 'North East',
                     'OH' : 'Mid West', 'SC' : 'South', 'IL' : 'Mid West', 'GA' : 'South', 'TX' : 'South', 'MI' : 'Mid West', 'MD' : 'North East', 'MA' : 'North East',
                     'CT' : 'North East', 'TN' : 'South', 'IN' : 'Mid West', 'WV' : 'South', 'PR' : 'Non-Continental US', 'AL' : 'South', 'MO' : 'Mid West', 'DC' : 'North East',
                     'WI' : 'Mid West', 'KY' : 'South', 'CO' : 'West', 'LA' : 'South', 'WA' : 'West', 'MN' : 'Mid West', 'AZ' : 'West', 'IA' : 'Mid West', 'MS' : 'South',
                     'KS' : 'West', 'OK' : 'West', 'UT' : 'West', 'NH' : 'North East', 'ME' : 'North East', 'RI' : 'North East', 'OR' : 'West', 'DE' : 'North East', 'AR' : 'South',
                     'NE' : 'West', 'HI' : 'Non-Continental US', 'VT' : 'North East', 'NV' : 'West', 'NM' : 'West', 'AK' : 'Non-Continental US', 'MT' : 'West', 'ID' : 'West',
                     'ND' : 'West', 'SD' : 'West', 'WY' : 'West', 'VI' : 'Non-Continental US', 'GU' : 'Non-Continental US', 'AS' : 'Non-Continental US', 'MP' : 'Non-Continental US', 'NO' : 'Non-Continental US'}
    df['Birth Region'] = df['birth_state'].replace(state2reagion).fillna('Unspecified')
    birth_regions = pd.get_dummies(df['Birth Region'])
    birth_regions.columns = [f"{x.lower()} origin".replace(" ", "_") for x in birth_regions.columns]
    df['drivers_lic'] = df['drivers_lic'].replace('Y', 1).replace('N', 0)
    X = race_features.join(gender_features).join(birth_regions).join(df[['drivers_lic', 'age_at_year_end']])
    logger.info("normalizing")
    X = X.applymap(int)
    return X.values
    
def build_bags(df):
    bag_list = df.groupby('FIPS').groups
    bag_list = {k : list(v) for k, v in bag_list.items()}
    return bag_list

def build_pairwise_constraints_indices(df):
    logger.info("building pairwise constraints")
    switches = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / 'switches_by_presidential_election_year.csv')
    switches = switches.merge(df[['county_desc', 'ncid']], on='ncid')
    fips2name = process_fips_to_name()
    fips2name = fips2name[fips2name.State == 'North Carolina']
    fips2name['county_desc'] = fips2name['County'].str.upper()
    switches = switches.merge(fips2name[['county_desc', 'FIPS']], on='county_desc').drop(columns=['county_desc'])
    probs = pd.DataFrame({2016: df.groupby('FIPS')['2016'].apply(lambda s: s.value_counts(normalize=True)[1]),
                          2020 : df.groupby('FIPS')['2020'].apply(lambda s: s.value_counts(normalize=True)[1]),
                          '2016 abs': df.groupby('FIPS')['2016'].apply(lambda s: s.value_counts()[1]),
                          '2020 abs': df.groupby('FIPS')['2020'].apply(lambda s: s.value_counts()[1])}).reset_index()
    probs['temp'] = 1
    probs.merge(probs, on='temp', how='outer')
    prob_pairs = probs.merge(probs, on='temp', how='outer')
    prob_pairs
    prob_pairs['diff'] = prob_pairs['2016_x']- prob_pairs['2016_y']
    pairwise_constraints_indices = prob_pairs.loc[prob_pairs.sort_values('diff', ascending=False).index[:200], ['FIPS_x', 'FIPS_y']].values.tolist()
    return probs, pairwise_constraints_indices

def create_upper_p_bounds(probs, pairwise_constraints_indices):
    return {}

def create_lower_p_bounds(probs, pairwise_constraints_indices):
    return {}

def create_upper_diff_bounds(probs, pairwise_constraints_indices):
    return {}

def create_lower_diff_bounds(probs, pairwise_constraints_indices):
    return {k[0] : 0 for k in pairwise_constraints_indices}

def solve_ballpark():

    # df = read_voter_rolls()
    df = pd.read_csv(config['db_dir'] / 'ballpark.csv')
    #df = df.sample(500000)
    indices = df.index
    df = df.reset_index(drop=True)

    X_features = build_features_table(df)
    bags = build_bags(df)
    probs, pairwise_constraints = build_pairwise_constraints_indices(df)
    upper_p_bound_bags = create_upper_p_bounds(probs, pairwise_constraints)
    lower_p_bound_bags = create_lower_p_bounds(probs, pairwise_constraints)
    upper_diff_bound_bags = create_upper_diff_bounds(probs, pairwise_constraints)
    lower_diff_bound_bags = create_lower_diff_bounds(probs, pairwise_constraints)
    # logger.info(f"X_features shape: {X_features.shape},  {len(bags)} bags")
    # logger.info(f"pairwise constraints: {pairwise_constraints}")
    # logger.info(f"upper_p_bound_bags: {upper_p_bound_bags}")
    # logger.info(f"lower_p_bound_bags: {lower_p_bound_bags}")
    # logger.info(f"diff_upper_bound_pairs: {upper_diff_bound_bags}")
    # logger.info(f"diff_lower_bound_pairs: {lower_diff_bound_bags}")

    logger.info("solving")
    w_t,y_t,loss_bp = solve_w_y(X=X_features, 
                            pairwise_constraints_indices=pairwise_constraints,
                            bag_list=bags,
                            upper_p_bound_bags=upper_p_bound_bags,
                            lower_p_bound_bags=lower_p_bound_bags,
                            diff_upper_bound_pairs=upper_diff_bound_bags,
                            diff_lower_bound_pairs=lower_diff_bound_bags)
    y_t = pd.Series(y_t, index=indices, name='y_t')
    df = df.join(y_t)
    df['y_t_hat'] = df['y_t'].apply(lambda x: int (random.uniform(0, 1)<x))
    df = df[['2020', 'y_t_hat']].rename(columns={'2020' : 'actual', 'y_t_hat' : 'predicted'})
    df['actual'] = (df['actual'] >= 1).apply(int)
    error_matrix =  df[['predicted', 'actual']].value_counts(normalize=True).unstack()*100
    return pd.Series(w_t),  y_t, error_matrix, loss_bp

if __name__ == "__main__":             
    w_t, y_t, error_matrix, loss_bp = solve_ballpark()

    logger.info("saving results")
    w_t.to_csv(config['output_dir'] / 'w_t.csv')
    y_t.to_csv(config['output_dir'] / 'y_t.csv')
    error_matrix.to_csv(config['output_dir'] / 'error_matrix.csv')
    logger.info(f"Ballpark loss: {loss_bp}")
    
