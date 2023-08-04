from ballpark_yhat_reg import solve_w_y

import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger, DATETIME
logger = get_logger(__name__)

IS_WHITE = 'is_white'
SWITCH_2020 = 'switch_2020'
VOTE_2016 = 'vote_2016'
SWITCH_2016 = 'switch_2016'
scenario = VOTE_2016

config = load_config(output_dir_suffix=scenario, add_date=DATETIME)
import pandas as pd
from datetime import datetime, date
import pandas as pd
from create_db_tools.processor_functions.static_and_demography import process_fips_to_name
import random
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)




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
    df.loc[:, 'Birth Region'] = df['birth_state'].replace(state2reagion).fillna('Unspecified')
    birth_regions = pd.get_dummies(df['Birth Region'])
    birth_regions.columns = [f"{x.lower()} origin".replace(" ", "_") for x in birth_regions.columns]
    df.loc[:, 'drivers_lic'] = df['drivers_lic'].replace('Y', 1).replace('N', 0)
    X = race_features.join(gender_features).join(birth_regions).join(df[['drivers_lic', 'age_at_year_end', 'zip_median_household_income', 'voted_2016', 'voted_2018']])
    X['urban'] = df['FIPS'].isin([37183, 37119, 37081, 37067, 37067, 37063])
    X['drivers_lic'] = X['drivers_lic'].apply(int)
    X['D_2016'] = (df['party_2016'] == 1).apply(int)
    X['R_2016'] = (df['party_2016'] == -1).apply(int)
    
    return X
 
    
    
def build_bags(df):
    bag_list = df.groupby('Region').groups
    return bag_list

def build_pairwise_constraints_indices(df):
    if scenario == IS_WHITE:
        df['is_white'] = df['race_code'] == 'W'
        probs = df.groupby('Region')['is_white'].mean().reset_index()
        probs = probs.sort_values('is_white')
    elif scenario == VOTE_2016:
        probs = df.groupby('Region')[['voted_2016', 'voted_2018', 'voted_2020']].mean().reset_index()
        probs = probs.sort_values('voted_2016')
    else:
        #df = df[['2016', '2020', 'Region']]
        
        probs = df.groupby('Region')[['switch_2016', 'switch_2020', 'switch_2016_right', 'switch_2016_left', 'switch_2020_right', 'switch_2020_left']].mean().reset_index()
        if scenario == SWITCH_2020:
            probs = probs.sort_values('switch_2020_right')
        elif scenario == SWITCH_2016:
            probs = probs.sort_values('switch_2016_right')
        else:
            assert False
    
    probs['temp'] = 1
    probs['order'] = range(len(probs))
    prob_pairs = probs.merge(probs, on='temp', how='outer')
    prob_pairs = prob_pairs[prob_pairs['order_x'] > prob_pairs['order_y']].drop(columns=['order_x', 'order_y', 'temp'])
    if scenario == VOTE_2016:
        prob_pairs['diff_2016'] = prob_pairs['voted_2016_x'] - prob_pairs['voted_2016_y']
        pairwise_constraints_indices = [tuple(x) for x in prob_pairs.loc[prob_pairs['diff_2016'] >= 0.019, ['Region_x', 'Region_y']].values.tolist()]
    else:
        pairwise_constraints_indices = [tuple(x) for x in prob_pairs[['Region_x', 'Region_y']].values.tolist()]
    return probs, pairwise_constraints_indices

def create_upper_p_bounds(bags):
    if scenario in [SWITCH_2020, SWITCH_2016]:
        return {bag : 0.03 for bag in bags.keys()}
    return {}
    

def create_lower_p_bounds(bags):
    return {}

def create_upper_diff_bounds(probs, pairwise_constraints_indices):
    if scenario == IS_WHITE:
        probs_by_region = probs.set_index('Region')
        return {pair: (int((probs_by_region.loc[pair[0]]-probs_by_region.loc[pair[1]])['is_white']*100)+1)/100 for pair in pairwise_constraints_indices}
    return {}

def create_lower_diff_bounds(probs, pairwise_constraints_indices):
    if scenario == VOTE_2016:
        probs_by_region = probs.set_index('Region')
        constraints = {}
        for pair in pairwise_constraints_indices:
            if (probs_by_region.loc[pair[0]] - probs_by_region.loc[pair[1]])['voted_2016'] > 0.027:
                constraints[pair] = 0.014
        return constraints
    return {}

def create_upper_ratio_bounds(probs, pairwise_constraints_indices):
    if scenario == SWITCH_2016:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2016_right'] +0.15 for pair in pairwise_constraints_indices}
    if scenario == SWITCH_2020:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2020_right'] +0.1 for pair in pairwise_constraints_indices}
    return {}
    
def create_lower_ratio_bounds(probs, pairwise_constraints_indices):
    if scenario == SWITCH_2016:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2016_right']-0.4 for pair in pairwise_constraints_indices}
    if scenario == SWITCH_2020:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2020_right']-0.1 for pair in pairwise_constraints_indices}
    return {}

def solve_ballpark():

    # df = read_voter_rolls()
    logger.info('Reading data')
    df_orig = pd.read_csv(config['db_dir'] / 'ballpark.csv')
    df = df_orig.sample(500000) #pd.concat([df[df['2016'] == 0].sample(df['2016'].sum()), df[df['2016'] == 1]])
    indices = df.index
    df = df.reset_index(drop=True)
    for year in ['2016', '2020']:
        for direction, switch in [('left', -1), ('right', 1)]:
            df.loc[:, f'switch_{year}_{direction}'] = (df[f'switch_{year}'] == switch).astype(int)

    X_df = build_features_table(df).join(df[['switch_2016', 'switch_2020']])
    X_features = X_df.values
    bags = build_bags(df)
    probs, pairwise_constraints = build_pairwise_constraints_indices(df_orig)
    upper_p_bound = create_upper_p_bounds(bags)
    lower_p_bound = create_lower_p_bounds(bags)
    diff_upper_bound_pairs = create_upper_diff_bounds(probs, pairwise_constraints)
    diff_lower_bound_pairs = create_lower_diff_bounds(probs, pairwise_constraints)
    ratio_upper_bound_pairs = create_upper_ratio_bounds(probs, pairwise_constraints)
    ratio_lower_bound_pairs = create_lower_ratio_bounds(probs, pairwise_constraints)
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
                            upper_p_bound=upper_p_bound,
                            lower_p_bound=lower_p_bound,
                            diff_upper_bound_pairs=diff_upper_bound_pairs,
                            diff_lower_bound_pairs=diff_lower_bound_pairs,
                            ratio_upper_bound_pairs=ratio_upper_bound_pairs,
                            ratio_lower_bound_pairs=ratio_lower_bound_pairs)
    y_t = pd.Series(y_t, index=indices, name='y_t')
    df = df.join(y_t)
    df = df[df['y_t'].notnull()]
    if scenario == SWITCH_2016:
        var = 'switch_2016_right'
    elif scenario == SWITCH_2020:
        var = 'switch_2020_right'
    elif scenario == VOTE_2016:
        var = 'voted_2016'
    elif scenario == IS_WHITE:
        var = 'is_white'
    error_matrix = pd.Series({False: df[df[var] == 0]['y_t'].mean(), True: df[df[var] == 1]['y_t'].mean()}, name='error')
    return pd.Series(w_t),  y_t, error_matrix, loss_bp

if __name__ == "__main__":   
    w_t, y_t, error_matrix, loss_bp = solve_ballpark()
    logger.info("saving results")
    w_t.to_csv(config['output_dir'] / f'w_t.csv')
    y_t.to_csv(config['output_dir'] / f'y_t.csv')
    error_matrix.to_csv(config['output_dir'] / f'error_matrix.csv')
    logger.info(f"Ballpark loss: {loss_bp}")

