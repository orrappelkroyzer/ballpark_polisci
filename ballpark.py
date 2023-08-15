from ballpark_yhat_reg import solve_w_y

import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger, DATETIME
logger = get_logger(__name__)
import argparse
# import plotly.express as px

VOTE_2016 = 'vote_2016'
VOTE_2016_STRICT = 'vote_2016_strict'
TURNOUT_2020_STRICT = 'turnout_2020_strict'
D_REG_2016 = 'D_reg_2016'
D_REG_2016_WO_UNA = 'D_reg_2016_no_una'
D_REG_2016_STRICT = 'D_reg_2016_strict'
D_REG_2016_WO_UNA_STRICT = 'D_reg_2016_no_una_strict'


scenarios = [VOTE_2016, VOTE_2016_STRICT, D_REG_2016, D_REG_2016_WO_UNA,  D_REG_2016_STRICT,
              D_REG_2016_WO_UNA_STRICT,  TURNOUT_2020_STRICT]
strict_scenarios = [VOTE_2016_STRICT, D_REG_2016_STRICT, D_REG_2016_WO_UNA_STRICT, TURNOUT_2020_STRICT]
import pandas as pd
from datetime import datetime, date
import pandas as pd
from create_db_tools.processor_functions.static_and_demography import process_fips_to_name
import random
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
import geopandas as gpd
# from utils.plotly_utils import fix_and_write

def read_data(config, scenario):
    logger.info("reading data")
    df =  pd.read_csv(config['db_dir'] / 'ballpark.csv')
    if scenario in [D_REG_2016_STRICT, D_REG_2016]:
        df = df[df['party_2016'].notnull()]
    elif scenario in [D_REG_2016_WO_UNA, D_REG_2016_WO_UNA_STRICT]:
        df = df[df['party_2016'].notnull() & (df['party_2016'] != 0)]
    else:
        df = df
    df = df[df['age_at_year_end'] >= 23]
    df['turned_out_2020'] = df['voted_2020'] * (1-df['voted_2016'])
    df.loc[df['age_at_year_end'].between(18, 30), 'age_group'] = 'below_30'
    df.loc[df['age_at_year_end'].between(30, 50), 'age_group'] = '30_to_50'
    df.loc[df['age_at_year_end'].between(50, 65), 'age_group'] = '50_to_65'
    df.loc[df['age_at_year_end'].between(65, 100), 'age_group'] = 'above_65'
    for year in range(2014, 2023, 2):
        df[f'voted_{year}'] = df[f'voted_{year}'].fillna(0)
    return df

race_dict = {
        'A' : 'asian',
        'B' : 'black',
        'I' : 'native_american',
        'M' : 'multiracial',
        'O' : 'of_other_race',
        'P' : 'pacific_islander',
        'U' : 'undesignated_race',
        'W' : 'white',
        'H' : 'latino'
    }
gender_dict = {
        'M' : 'male',
        'F' : 'female',
        'U' : 'undesignated_gender'
    }
 
def scenario2field(scenario):
    if scenario in [VOTE_2016, VOTE_2016_STRICT]:
        field = 'voted_2016'
    elif scenario == TURNOUT_2020_STRICT:
        field = 'turned_out_2020'
    elif scenario in [D_REG_2016, D_REG_2016_WO_UNA, D_REG_2016_STRICT, D_REG_2016_WO_UNA_STRICT]:
        field = 'D_2016'
    else:
        assert False
    return field

def build_features_table(df, scenario):
    logger.info("creating race features")
    race_features = pd.get_dummies(df['race_code']).rename(columns={k: f"is_{v}" for k, v in race_dict.items()})
    logger.info("creating gender features")
    gender_features = pd.get_dummies(df['gender_code']).rename(columns={k: f"is_{v}" for k, v in gender_dict.items()})
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
    logger.info("creating driver_liscence features")
    df.loc[:, 'drivers_lic'] = df['drivers_lic'].replace('Y', 1).replace('N', 0)
    logger.info("merging")
    X = race_features.\
        join(gender_features).\
            join(birth_regions).\
                join(df[['drivers_lic', 'age_at_year_end']])
    X['urban'] = df['FIPS'].isin([37183, 37119, 37081, 37067, 37067, 37063])
    X['drivers_lic'] = X['drivers_lic'].apply(int)
    
    voting_method_dict_2020 = {
        'ABSENTEE BY MAIL' : 'voted_absentee_by_mail_2020',
        'ABSENTEE CURBSIDE' : 'voted_absentee_curbside_2020',
        'ABSENTEE ONESTOP' : 'voted_absentee_onestop_2020'
    }

    if scenario in [VOTE_2016, VOTE_2016_STRICT, TURNOUT_2020_STRICT]: 
        columns = ['D_2016', 'R_2016', 'median_household_income_2016', '%_college_educated_2016']
    elif scenario in [D_REG_2016, D_REG_2016_WO_UNA, D_REG_2016_STRICT, D_REG_2016_WO_UNA_STRICT]:
        # columns = ['median_household_income_2016', '%_college_educated_2016']
        columns = ['voted_2016', 'voted_2018', 'median_household_income_2016', '%_college_educated_2016']
        df = df.join(pd.get_dummies(df['voting_method_2020']).rename(columns=voting_method_dict_2020))
    else:
        assert False
    for col in columns:
        X[col] = df[col]
    if '%_college_educated_2020' in X.columns:
        X['whites_no_college_degree'] = (df['race_code'] == 'W').apply(int) * df['%_college_educated_2020'].apply(lambda x: 1 - x)
    if '%_college_educated_2016' in X.columns:
        X['whites_no_college_degree'] = (df['race_code'] == 'W').apply(int) * df['%_college_educated_2016'].apply(lambda x: 1 - x)
    logger.info(f"X_features columns:\n{X.columns.to_list()}")
    return X
    
def build_bags(df, scenario):
    if scenario in strict_scenarios:
        bags = {f'FIPS_{k}' : v for k, v in df.groupby('FIPS').groups.items()}
    else:
        bags = {f'Region_{k}' : v for k, v in df.groupby('Region').groups.items()}

    bags.update({f'Race_{race_dict[k]}' : df.groupby('race_code').groups[k].tolist() for k in ['A', 'B', 'H', 'I', 'U', 'W']})
    bags.update({f'Gender_{gender_dict[k]}' : v.tolist() for k, v in df.groupby('gender_code').groups.items()})
    bags.update({f'Age_{k}' : v.tolist() for k, v in df.groupby('age_group').groups.items()})
    bags['All'] = df.index.tolist()
    return bags
   
def build_probs(df, scenario):
    logger.info(f"building probabilities for {scenario}")
    if scenario in strict_scenarios:
        grouper = 'FIPS'
    else:
        grouper = 'Region'
    field = scenario2field(scenario)    
    probs = df.groupby(grouper)[[field]].mean().reset_index().rename(columns={grouper : 'bag'})
    probs['bag'] = probs['bag'].apply(lambda x: f"{grouper}_{x}")
    if scenario  in strict_scenarios:
        return probs
    race_probs = df.groupby('race_code')[[field]].mean().loc[['A', 'B', 'H', 'I', 'U', 'W']].reset_index().rename(columns={'race_code' : 'bag'}).replace(race_dict)
    race_probs['bag'] = race_probs['bag'].apply(lambda x: f"Race_{x}")
    gender_probs = df.groupby('gender_code')[[field]].mean().reset_index().rename(columns={'gender_code' : 'bag'}).replace(gender_dict)
    gender_probs['bag'] = gender_probs['bag'].apply(lambda x: f"Gender_{x}")
    age_probs = df.groupby('age_group')[[field]].mean().reset_index().rename(columns={'age_group' : 'bag'}).replace(gender_dict)
    age_probs['bag'] = age_probs['bag'].apply(lambda x: f"Age_{x}")
    probs = pd.concat([probs, race_probs, gender_probs, age_probs])

    

    return probs

def build_pairwise_constraints(probs, scenario):
    logger.info(f"building pairwise constraints for {scenario}")
    if scenario in strict_scenarios:
        return []
    field = scenario2field(scenario)
    probs = probs.sort_values(field)
    probs['temp'] = 1
    probs['order'] = range(len(probs))
    pairwise_constraints_indices = []
    county_region_probs = probs[probs['bag'].str.startswith('FIPS') | probs['bag'].str.startswith('Region')]
    race_probs = probs[probs['bag'].str.startswith('Race')]
    gender_probs = probs[probs['bag'].str.startswith('Gender')]
    prob_pairs_list = []
    for t_probs in [county_region_probs, race_probs, gender_probs]:
        t_prob_pairs = t_probs.merge(t_probs, on='temp', how='outer')
        prob_pairs_list += [t_prob_pairs[t_prob_pairs['order_x'] > t_prob_pairs['order_y']].drop(columns=['order_x', 'order_y', 'temp'])]
    prob_pairs = pd.concat(prob_pairs_list)
    pairwise_constraints_indices += [tuple(x) for x in prob_pairs[['bag_x', 'bag_y']].values.tolist()]
    return pairwise_constraints_indices
    
def create_upper_p_bounds(probs, scenario):
    county_region_probs = probs[probs['bag'].str.startswith('FIPS') | probs['bag'].str.startswith('Region')].set_index('bag')
    constraints = {}
    if scenario in strict_scenarios:
        constraints.update((county_region_probs[scenario2field(scenario)] + 0.0005).to_dict())
    if scenario == VOTE_2016:
        constraints.update((county_region_probs[scenario2field(scenario)] + 0.2).to_dict())
        constraints.update({'Race_undesignated_race'    :   0.5,
                            'Race_asian'                :   0.7,
                            'Race_latino'               :   0.8,
                            'Race_native_american'      :   0.9,
                            'Race_white'                :   1,
                            'Race_black'                :   1})
        constraints.update({'Gender_male'               :   1,
                            'Gender_female'             :   1})
        constraints.update({'Age_below_30'              :   0.6,
                            'Age_30_to_50'              :   0.8,
                            'Age_50_to_65'              :   0.9,
                            'Age_above_65'              :   0.9})
        constraints['All'] = 1
    if scenario == D_REG_2016:
        constraints.update((county_region_probs[scenario2field(scenario)] + 0.1).to_dict())
        constraints.update({'Race_undesignated_race'    :   0.5,
                            'Race_asian'                :   0.3,
                            'Race_latino'               :   0.5,
                            'Race_native_american'      :   0.8,
                            'Race_white'                :   0.25,
                            'Race_black'                :   1})
        constraints.update({'Gender_male'               :   0.5,
                            'Gender_female'             :   0.6})
        constraints['All'] = 0.45
        pass
        
    if scenario == D_REG_2016_WO_UNA:
        constraints.update((county_region_probs[scenario2field(scenario)] + 0.05).to_dict())
        constraints.update({'Race_undesignated_race'    :   0.7,
                            'Race_asian'                :   0.75,
                            'Race_latino'               :   0.85,
                            'Race_native_american'      :   0.85,
                            'Race_white'                :   0.4,
                            'Race_black'                :   1})
        constraints.update({'Gender_male'               :   0.5,
                            'Gender_female'             :   0.6,
                            'Gender_undesignated_gender':   0.6})
        constraints['All'] = 0.6
        pass

    return constraints

def create_lower_p_bounds(probs, scenario):
    county_region_probs = probs[probs['bag'].str.startswith('FIPS') | probs['bag'].str.startswith('Region')].set_index('bag')
    probs[probs['bag'].str.startswith('Race')].set_index('bag')
    probs[probs['bag'].str.startswith('Gender')].set_index('bag')
    constraints = {}
    if scenario in strict_scenarios:
        constraints.update((county_region_probs[scenario2field(scenario)] - 0.0005).to_dict())
    if scenario == VOTE_2016:
        constraints.update((county_region_probs[scenario2field(scenario)] - 0.2).to_dict())
        constraints.update({'Race_undesignated_race'    :   0.2,
                            'Race_asian'                :   0.4,
                            'Race_latino'               :   0.5,
                            'Race_native_american'      :   0.6,
                            'Race_white'                :   0.75,
                            'Race_black'                :   0.75})
        constraints.update({'Gender_male'               :   0.7,
                            'Gender_female'             :   0.9})
        constraints['All'] = 0.8
    if scenario == D_REG_2016:
        constraints.update((county_region_probs[scenario2field(scenario)] - 0.1).to_dict())
        constraints.update({'Race_undesignated_race'    :   0.3,
                            'Race_asian'                :   0.55,
                            'Race_latino'               :   0.6,
                            'Race_native_american'      :   0.6,
                            'Race_white'                :   0.25,
                            'Race_black'                :   0.95})
        constraints.update({'Gender_male'               :   0.1,
                            'Gender_female'             :   0.2})
        constraints['All'] = 0.35
        pass
    if scenario == D_REG_2016_WO_UNA:
        constraints.update((county_region_probs[scenario2field(scenario)] - 0.05).to_dict())
        constraints.update({'Race_undesignated_race'    :   0.5,
                            'Race_asian'                :   0.3,
                            'Race_latino'               :   0.5,
                            'Race_native_american'      :   0.8,
                            'Race_white'                :   0.25,
                            'Race_black'                :   0.9})
        constraints.update({'Gender_male'               :   0.4,
                            'Gender_female'             :   0.5,
                            'Gender_undesignated_gender':   0.4})
        constraints['All'] = 0.45
        pass
    return constraints

def create_upper_diff_bounds(probs, pairwise_constraints_indices, scenario):
    if scenario in strict_scenarios:
        return {}
    county_region_probs = probs[probs['bag'].str.startswith('FIPS') | probs['bag'].str.startswith('Region')].set_index('bag')
    race_probs = probs[probs['bag'].str.startswith('Race')].set_index('bag')
    gender_probs = probs[probs['bag'].str.startswith('Gender')].set_index('bag')
    county_region_pairwise_constraints = [x for x in pairwise_constraints_indices if x[0] in county_region_probs.index]
    race_pairwise_constraints = [x for x in pairwise_constraints_indices if x[0] in race_probs.index]
    gender_pairwise_constraints = [x for x in pairwise_constraints_indices if x[0] in gender_probs.index]
    
    constraints = {}
    if scenario == VOTE_2016:
        # constraints.update({pair: (county_region_probs.loc[pair[0]]-county_region_probs.loc[pair[1]])['voted_2016']*2 
        #                     for pair in county_region_pairwise_constraints})
        # constraints.update({pair: 0.15
        #                     for pair in race_pairwise_constraints})
        # constraints[('Race_black', 'Race_white')] = 0.03
        # constraints[('Gender_female', 'Gender_male')] = 0.01
        pass
    if scenario == D_REG_2016:
        # constraints.update({pair: (county_region_probs.loc[pair[0]]-county_region_probs.loc[pair[1]])['D_2016']*2.5 
        #                     for pair in county_region_pairwise_constraints})
        # constraints.update({pair: 0.15
        #                     for pair in race_pairwise_constraints})
        # constraints[('Gender_female', 'Gender_male')] = 0.1
        pass
    if scenario == D_REG_2016_WO_UNA:
        # constraints.update({pair: (county_region_probs.loc[pair[0]]-county_region_probs.loc[pair[1]])['D_2016']+0.03 
        #                     for pair in county_region_pairwise_constraints})
        # constraints.update({('Race_black', 'Race_latino'): 0.3,
        #                     ('Race_black', 'Race_white'): 0.8,
        #                     ('Race_latino', 'Race_white'): 0.4})
        # constraints[('Gender_female', 'Gender_male')] = 0.15
        pass
    return constraints

def create_lower_diff_bounds(probs, pairwise_constraints_indices, scenario):
    if scenario in strict_scenarios:
        return {}
    county_region_probs = probs[probs['bag'].str.startswith('FIPS') | probs['bag'].str.startswith('Region')].set_index('bag')
    race_probs = probs[probs['bag'].str.startswith('Race')].set_index('bag')
    gender_probs = probs[probs['bag'].str.startswith('Gender')].set_index('bag')
    county_region_pairwise_constraints = [x for x in pairwise_constraints_indices if x[0] in county_region_probs.index]
    race_pairwise_constraints = [x for x in pairwise_constraints_indices if x[0] in race_probs.index]
    gender_pairwise_constraints = [x for x in pairwise_constraints_indices if x[0] in gender_probs.index]
    constraints = {}
    if scenario == VOTE_2016:
        # constraints.update({pair: (county_region_probs.loc[pair[0]]-county_region_probs.loc[pair[1]])['voted_2016']*0.2 
        #                     for pair in county_region_pairwise_constraints})
        # constraints.update({pair: -0.05
        #                     for pair in race_pairwise_constraints})
        # constraints[('Gender_female', 'Gender_male')] = -0.05
        pass
    if scenario == D_REG_2016:
        # constraints.update({pair: (county_region_probs.loc[pair[0]]-county_region_probs.loc[pair[1]])['D_2016']*0.2
        #                     for pair in county_region_pairwise_constraints})
        # constraints.update({pair: 0
        #                     for pair in race_pairwise_constraints})
        # constraints[('Gender_female', 'Gender_male')] = -0.01
        pass
    if scenario == D_REG_2016_WO_UNA:
        # constraints.update({pair: (county_region_probs.loc[pair[0]]-county_region_probs.loc[pair[1]])['D_2016']-0.03 
        #                     for pair in county_region_pairwise_constraints})
        # constraints.update({('Race_black', 'Race_latino'): 0.2,
        #                     ('Race_black', 'Race_white'): 0.55,
        #                     ('Race_latino', 'Race_white'): 0.3})
        # constraints[('Gender_female', 'Gender_male')] = 0.08
        pass
    return constraints

def create_upper_ratio_bounds(probs, pairwise_constraints_indices, scenario):
    return {}
    
def create_lower_ratio_bounds(probs, pairwise_constraints_indices, scenario):
    return {}

def prepare_data(scenario, config):

    logger.info('Reading data')
    df_orig = read_data(config, scenario)
    df = df_orig#.sample(100000) 
    indices = df.index
    df = df.reset_index(drop=True)
    X_df = build_features_table(df, scenario).join(df[['switch_2016', 'switch_2020']])
    X_features = X_df.values
    X_df.index = indices
    bags = build_bags(df, scenario=scenario)
    probs = build_probs(df_orig, scenario=scenario)
    pairwise_constraints = build_pairwise_constraints(probs=probs, scenario=scenario)
    upper_p_bound = create_upper_p_bounds(probs, scenario=scenario)
    lower_p_bound = create_lower_p_bounds(probs, scenario=scenario)
    diff_upper_bound_pairs = create_upper_diff_bounds(probs, pairwise_constraints, scenario=scenario)
    diff_lower_bound_pairs = create_lower_diff_bounds(probs, pairwise_constraints, scenario=scenario)
    # ratio_upper_bound_pairs = create_upper_ratio_bounds(probs, pairwise_constraints, scenario=scenario)
    # ratio_lower_bound_pairs = create_lower_ratio_bounds(probs, pairwise_constraints, scenario=scenario)
    return indices, probs,  dict(X=X_features, 
                            pairwise_constraints_indices=pairwise_constraints,
                            bag_list=bags,
                            upper_p_bound=upper_p_bound,
                            lower_p_bound=lower_p_bound,
                            diff_upper_bound_pairs=diff_upper_bound_pairs,
                            diff_lower_bound_pairs=diff_lower_bound_pairs)
                            # ratio_upper_bound_pairs=ratio_upper_bound_pairs,
                            # ratio_lower_bound_pairs=ratio_lower_bound_pairs)

def solve(indices, params):
    logger.info("solving")
    w_t,y_t,loss_bp = solve_w_y(**params)
    return pd.Series(w_t),  pd.Series(y_t, index=indices), loss_bp

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
    config = load_config(output_dir_suffix=scenario, add_date=DATETIME)
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Output dir: {config['output_dir']}")
    return scenario, config

def plot_probs(probs, scenario):
    field = scenario2field(scenario)
    probs = probs[probs['bag'].str.startswith('FIPS') | probs['bag'].str.startswith('Region')]
    if probs['bag'].iloc[0].startswith('Region'):
        probs['Region'] = probs['bag'].str.split('_').str[1]
        gdf = gpd.read_file(config['db_dir'] / 'GIS' / "NC_regions" / "NC_regions.shp")
        gdf = gdf.set_index('Region')
        locations_field = 'Region'
    elif probs['bag'].iloc[0].startswith('FIPS'):
        probs['FIPS'] = probs['bag'].str.split('_').str[1]
        gdf = gpd.read_file(config['db_dir'] / 'GIS' / "counties.shp")
        gdf = gdf[gdf['state'] == 'North Carolina'].set_index('FIPS')[['geometry']]
        locations_field = 'FIPS'
    else:
        assert False
    fig = px.choropleth(probs,
                        geojson=gdf,
                        locations=locations_field,
                        color=field,
                        projection='albers usa',
                        title=field)
    fig.update_geos(fitbounds='locations', scope='usa')
    fix_and_write(fig, output_dir=config['output_dir'], filename=scenario)    
        

if __name__ == "__main__":   
    
    scenario, config = parse_args()    
    indices, probs, params = prepare_data(scenario, config)
    # plot_probs(probs=probs, scenario=scenario)
    w_t, y_t, loss_bp = solve(indices=indices, params=params)
    logger.info("saving results")
    # X_df.to_csv(config['output_dir'] / f'X_df.csv')
    w_t.to_csv(config['output_dir'] / f'w_t.csv')
    y_t.to_csv(config['output_dir'] / f'y_t.csv')
    logger.info(f"Ballpark loss: {loss_bp}")

