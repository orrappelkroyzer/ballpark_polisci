import os, sys
from pathlib import Path
local_python_path = str(Path(__file__).parent)
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config()
import pandas as pd
from datetime import datetime, date
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from utils.plotly_utils import fix_and_write
import ballpark as bp
from sklearn.linear_model import LinearRegression


def post_process(df_orig, subdir, fields):
   global scenario
   scenario = Path(subdir).parts[-2]
   input_dir = config['output_dir'].parent / subdir
   
   logger.info(f"Reading y_t")
   y_t = pd.read_csv(input_dir / "y_t.csv")
   y_t = pd.Series(y_t['0'].values, index=y_t['Unnamed: 0'].values, name="y_t")
   df = df_orig.join(y_t)
   df = df[df['y_t'].notnull()]
   logger.info(f"calculating ROC curve")
   title = ' '.join(word.capitalize() for word in scenario.split('_'))
   fig = go.Figure()
   for field in fields:
      ROC_curve = pd.DataFrame([{'x' : x, 
                           'TPR' : ((df['y_t'] > x) & (df[field] == 1)).sum()/(df[field] == 1).sum(),
                           'FPR' : ((df['y_t'] > x) & (df[field] != 1)).sum()/(df[field] != 1).sum()} 
                           for x in np.arange(0,1.1, .01)])
   
      fig.add_trace(go.Scatter(x=ROC_curve['FPR'], y=ROC_curve['TPR'],  name=field))
   logger.info(f"calculating X_df for LR")
   field = fields[0]
   df = df_orig[df_orig[field].notnull()]
   X_df = bp.build_features_table(df, scenario=scenario).astype(float)
   logger.info(f"running LR")
   df['y_t'] = LinearRegression().fit(X_df, df[field]).predict(X_df)
   logger.info(f"calculating ROC curve")
   for field in fields:
      ROC_curve = pd.DataFrame([{'x' : x, 
                                 'TPR' : ((df['y_t'] > x) & (df[field] == 1)).sum()/(df[field] == 1).sum(),
                                 'FPR' : ((df['y_t'] > x) & (df[field] != 1)).sum()/(df[field] != 1).sum()} 
                                 for x in np.arange(0,1.1, .01)])
      fig.add_trace(go.Scatter(x=ROC_curve['FPR'], y=ROC_curve['TPR'],  name=f"{field} Linear Regession"))
   

   fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='random',
                           line=dict(color='black', dash='dash')))
   fig.update_layout(title=title)
   fix_and_write(fig=fig,
                  filename=scenario,
                  output_dir=config['output_dir'])

def main():
   logger.info(f"Reading df_orig")
   df_orig = pd.read_csv(config['db_dir'] / 'ballpark.csv')
   post_process(df_orig=df_orig, subdir=r"vote_2016\2023-08-14_07-47", fields=['voted_2016', 'voted_2020'])
   
if __name__ == '__main__':
   main()