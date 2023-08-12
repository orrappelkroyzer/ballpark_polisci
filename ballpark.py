from ballpark_yhat_reg import solve_w_y

import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger, DATETIME
logger = get_logger(__name__)
import argparse

IS_WHITE = 'is_white'
SWITCH_2016 = 'switch_2016'
SWITCH_2020 = 'switch_2020'
SWITCH_2020_STRICT = 'switch_2020_strict'
VOTE_2016 = 'vote_2016'
VOTE_2016_STRICT = 'vote_2016_strict'
D_REG_2016 = 'D_reg_2016'
D_REG_2016_WO_UNA = 'D_reg_2016_no_una'
D_VOTE_2020 = 'D_vote_2020'

scenarios = [IS_WHITE, SWITCH_2016, SWITCH_2020, SWITCH_2020_STRICT, VOTE_2016, VOTE_2016_STRICT, D_REG_2016, D_REG_2016_WO_UNA, D_VOTE_2020]
import pandas as pd
from datetime import datetime, date
import pandas as pd
from create_db_tools.processor_functions.static_and_demography import process_fips_to_name
import random
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)



def build_features_table(df, scenario):
    logger.info("creating race features")
    race_dict = {
        'A' : 'is_asian',
        'B' : 'is_black',
        'I' : 'is_native_american',
        'M' : 'is_multiracial',
        'O' : 'is_of_other_race',
        'P' : 'is_pacific_islander',
        'U' : 'is_undesignated_race',
        'W' : 'is_white',
        'H' : 'is_latino'
    }
    race_features = pd.get_dummies(df['race_code']).drop(columns=['W', ' '], errors='ignore').rename(columns=race_dict)
    logger.info("creating gender features")
    gender_dict = {
        'M' : 'is_male',
        'F' : 'is_female',
    }
    gender_features = pd.get_dummies(df['gender_code']).drop(columns=['U', ' '],errors='ignore').rename(columns=gender_dict)
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
    X = race_features.\
        join(gender_features).\
            join(birth_regions).\
                join(df[['drivers_lic', 'age_at_year_end']])
    X['urban'] = df['FIPS'].isin([37183, 37119, 37081, 37067, 37067, 37063])
    X['drivers_lic'] = X['drivers_lic'].apply(int)
    voting_method_dict_2016 = {
        'ABSENTEE BY MAIL' : 'voted_absentee_by_mail_2016',
        'ABSENTEE CURBSIDE' : 'voted_absentee_curbside_2016',
        'ABSENTEE ONESTOP' : 'voted_absentee_onestop_2016'
    }
    voting_method_dict_2020 = {
        'ABSENTEE BY MAIL' : 'voted_absentee_by_mail_2020',
        'ABSENTEE CURBSIDE' : 'voted_absentee_curbside_2020',
        'ABSENTEE ONESTOP' : 'voted_absentee_onestop_2020'
    }

    if scenario == IS_WHITE:
        pass
    elif scenario == SWITCH_2016:
        df = df.join(pd.get_dummies(df['voting_method_2020']).drop(columns=['IN-PERSON'],errors='ignore').rename(columns=voting_method_dict_2020))
        df = df.join(pd.get_dummies(df['voting_method_2016']).drop(columns=['IN-PERSON'],errors='ignore').rename(columns=voting_method_dict_2016))
        columns = ['D_2016', 'R_2016',  'voted_2016', 'voted_2018', 'median_household_income_2016', '%_college_educated_2016']
    elif scenario in [SWITCH_2020, SWITCH_2020_STRICT]:
        columns = ['D_2020', 'R_2020', 'median_household_income_2020', '%_college_educated_2020']
        df = df.join(pd.get_dummies(df['voting_method_2020']).drop(columns=['IN-PERSON'],errors='ignore').rename(columns=voting_method_dict_2020))
    elif scenario in [VOTE_2016, VOTE_2016_STRICT]: 
        columns = ['D_2016', 'R_2016', 'median_household_income_2016', '%_college_educated_2016']
    elif scenario in [D_REG_2016, D_REG_2016_WO_UNA]:
        columns = ['voted_2016', 'voted_2018', 'median_household_income_2016', '%_college_educated_2016']
        df = df.join(pd.get_dummies(df['voting_method_2020']).drop(columns=['IN-PERSON'],errors='ignore').rename(columns=voting_method_dict_2020))
    elif scenario == D_VOTE_2020:
        df = df.join(pd.get_dummies(df['voting_method_2020']).drop(columns=['IN-PERSON'],errors='ignore').rename(columns=voting_method_dict_2020))
        columns = ['D_2020', 'R_2020', 'median_household_income_2020', '%_college_educated_2020']
    else:
        assert False
    for col in columns:
        X[col] = df[col]

    return X
 
    
    
def build_bags(df, scenario):
    if scenario in [SWITCH_2020_STRICT, VOTE_2016_STRICT, D_REG_2016, D_REG_2016_WO_UNA, D_VOTE_2020]:
        return df.groupby('FIPS').groups
    return df.groupby('Region').groups
    

def build_pairwise_constraints_indices(df, scenario):
    if scenario == IS_WHITE:
        df['is_white'] = df['race_code'] == 'W'
        probs = df.groupby('Region')['is_white'].mean().reset_index()
        probs = probs.sort_values('is_white')
    elif scenario == VOTE_2016:
        probs = df.groupby('Region')[['voted_2016', 'voted_2018', 'voted_2020']].mean().reset_index()
        probs = probs.sort_values('voted_2016')
    elif scenario == VOTE_2016_STRICT:
        probs = df.groupby('FIPS')[['voted_2016', 'voted_2018', 'voted_2020']].mean().reset_index()
        probs = probs.sort_values('voted_2016')
    elif scenario in [D_REG_2016, D_REG_2016_WO_UNA]:
        probs = df.groupby('FIPS')[['D_2016']].mean().reset_index()
        probs = probs.sort_values('D_2016')
    elif scenario == D_VOTE_2020:
        probs = pd.read_csv(config['db_dir'] / 'db_creation_input' / 'ballpark' / '2020_election_results.csv')[['FIPS', 'D_2020']]
        probs = probs.sort_values('D_2020')
    elif scenario == SWITCH_2020_STRICT:
        probs = df.groupby('FIPS')[['switch_2016', 'switch_2020', 'switch_2016_right', 'switch_2016_left', 'switch_2020_right', 'switch_2020_left']].mean().reset_index()
        probs = probs.sort_values('switch_2020_right')
    else:
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
    if scenario in [VOTE_2016_STRICT, D_REG_2016, D_REG_2016_WO_UNA, D_VOTE_2020, SWITCH_2020_STRICT]:
        pairwise_constraints_indices = [] # [tuple(x) for x in prob_pairs[['FIPS_x', 'FIPS_y']].values.tolist()]
    else:
        pairwise_constraints_indices = [tuple(x) for x in prob_pairs[['Region_x', 'Region_y']].values.tolist()]
    return probs, pairwise_constraints_indices

def create_upper_p_bounds(bags, probs, scenario):
    if scenario in [SWITCH_2020, SWITCH_2016]:
        return {bag : 0.03 for bag in bags.keys()}
    if scenario == SWITCH_2020_STRICT:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['switch_2020'] + 0.0005 for bag in bags.keys()}
    if scenario == VOTE_2016_STRICT:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['voted_2016'] + 0.0005 for bag in bags.keys()}
    if scenario in [D_REG_2016, D_REG_2016_WO_UNA]:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['D_2020'] + 0.0005 for bag in bags.keys()}
    if scenario in [D_VOTE_2020]:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['D_2020'] + 0.0005 for bag in bags.keys()}
    

    return {}
    

def create_lower_p_bounds(bags, probs, scenario):
    if scenario == SWITCH_2020_STRICT:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['switch_2020'] - 0.005 for bag in bags.keys()}
    if scenario == VOTE_2016_STRICT:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['voted_2016'] - 0.005 for bag in bags.keys()}
    if scenario in [D_REG_2016, D_REG_2016_WO_UNA]:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['D_2020'] - 0.0005 for bag in bags.keys()}
    if scenario in [D_VOTE_2020]:
        probs_by_region = probs.set_index('FIPS')
        return {bag: probs_by_region.loc[bag]['D_2020'] - 0.0005 for bag in bags.keys()}
    
    return {}

def create_upper_diff_bounds(probs, pairwise_constraints_indices, scenario):
    if scenario == VOTE_2016:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]-probs_by_region.loc[pair[1]])['voted_2016']*2 for pair in pairwise_constraints_indices}   

def create_lower_diff_bounds(probs, pairwise_constraints_indices, scenario):
    if scenario == IS_WHITE:
        probs_by_region = probs.set_index('Region')
        return {pair: (int((probs_by_region.loc[pair[0]]-probs_by_region.loc[pair[1]])['is_white']*100)+1)/100 for pair in pairwise_constraints_indices}
    if scenario == VOTE_2016:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]-probs_by_region.loc[pair[1]])['voted_2016']/2 for pair in pairwise_constraints_indices}   
    return {}

def create_upper_ratio_bounds(probs, pairwise_constraints_indices, scenario):
    if scenario == SWITCH_2016:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2016_right'] +0.15 for pair in pairwise_constraints_indices}
    if scenario == SWITCH_2020:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2020_right'] +0.01 for pair in pairwise_constraints_indices}
    return {}
    
def create_lower_ratio_bounds(probs, pairwise_constraints_indices, scenario):
    if scenario == SWITCH_2016:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2016_right']-0.4 for pair in pairwise_constraints_indices}
    if scenario == SWITCH_2020:
        probs_by_region = probs.set_index('Region')
        return {pair: (probs_by_region.loc[pair[0]]/probs_by_region.loc[pair[1]])['switch_2020_right']-0.01 for pair in pairwise_constraints_indices}
    return {}

def solve_ballpark(scenario):

    logger.info('Reading data')
    df_orig =  pd.read_csv(config['db_dir'] / 'ballpark.csv')
    df = df_orig.sample(200000) #pd.concat([df[df['2016'] == 0].sample(df['2016'].sum()), df[df['2016'] == 1]])
    indices = df.index
    df = df.reset_index(drop=True)
    X_df = build_features_table(df, scenario).join(df[['switch_2016', 'switch_2020']])
    X_features = X_df.values
    X_df.index = indices
    bags = build_bags(df, scenario=scenario)
    probs, pairwise_constraints = build_pairwise_constraints_indices(df_orig, scenario=scenario)
    upper_p_bound = create_upper_p_bounds(bags, probs, scenario=scenario)
    lower_p_bound = create_lower_p_bounds(bags, probs, scenario=scenario)
    diff_upper_bound_pairs = create_upper_diff_bounds(probs, pairwise_constraints, scenario=scenario)
    diff_lower_bound_pairs = create_lower_diff_bounds(probs, pairwise_constraints, scenario=scenario)
    ratio_upper_bound_pairs = create_upper_ratio_bounds(probs, pairwise_constraints, scenario=scenario)
    ratio_lower_bound_pairs = create_lower_ratio_bounds(probs, pairwise_constraints, scenario=scenario)
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
    return pd.Series(w_t),  pd.Series(y_t, index=indices), loss_bp, X_df


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Receive a scenario from a predefined list.")

    # Add an argument for the scenario
    parser.add_argument(
        'scenario',
        help="Specify a scenario",
        choices=scenarios,
        metavar='scenario'
    )

    # Parse the provided arguments
    args = parser.parse_args()
    scenario = args.scenario
    global config
    config = load_config(output_dir_suffix=scenario, add_date=DATETIME)
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Output dir: {config['output_dir']}")
    return scenario

if __name__ == "__main__":   
    
    scenario = parse_args()    
    w_t, y_t, loss_bp, X_df = solve_ballpark(scenario=scenario)
    logger.info("saving results")
    X_df.to_csv(config['output_dir'] / f'X_df.csv')
    w_t.to_csv(config['output_dir'] / f'w_t.csv')
    y_t.to_csv(config['output_dir'] / f'y_t.csv')
    logger.info(f"Ballpark loss: {loss_bp}")

