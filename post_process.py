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
from sklearn import metrics


def post_process(df_orig, subdir, fields):
   
   input_dir = config['output_dir'].parent / subdir
   logger.info(f"Reading y_t")
   y_t = pd.read_csv(input_dir / "y_t.csv")
   y_t = pd.Series(y_t['0'].values, index=y_t['Unnamed: 0'].values, name="y_t")
   df = df_orig.join(y_t)
   df = df[df['y_t'].notnull()]
   logger.info(f"calculating ROC curve")
   title = ' '.join(word.capitalize() for word in scenario.split('_'))
   auc_scores = {}
   fig = go.Figure()
   for field in fields:
      y = df[field].replace({1: 2, 0: 1})
      fpr, tpr, thresholds = metrics.roc_curve(y, df['y_t'], pos_label=2)
      fig.add_trace(go.Scatter(x=fpr, y=tpr,  name=field))
      auc_scores[field] = metrics.auc(fpr, tpr)
   logger.info(f"calculating X_df for LR")
   field = fields[0]
   df = df_orig[df_orig[field].notnull()]
   X_df = bp.build_features_table(df, scenario=scenario).astype(float)
   logger.info(f"running LR")
   df['y_t'] = LinearRegression().fit(X_df, df[field]).predict(X_df)
   logger.info(f"calculating ROC curve")
   
   for field in fields:
      y = df[field].replace({1: 2, 0: 1})
      fpr, tpr, thresholds = metrics.roc_curve(y, df['y_t'], pos_label=2)
      fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{field} baseline"))
      auc_scores[f"{field} baseline"] = metrics.auc(fpr, tpr)

   fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='random',
                           line=dict(color='black', dash='dash')))
   logger.info(f"writing output")
   fig.update_layout(title=title,
                     xaxis_title="FPR",
                     yaxis_title="TPR")
   fix_and_write(fig=fig,
                  filename=f"{scenario}_{'_'.join(fields)}",
                  output_dir=config['output_dir'])
   pd.Series(auc_scores).to_csv(config['output_dir'] / f"{scenario}_{'_'.join(fields)}_auc.csv")
   

def main():
   logger.info(f"Reading df_orig")
   subdir = r"D_reg_2016_no_una\2023-08-16_09-14"
   global scenario
   scenario = Path(subdir).parts[-2]
   df_orig = bp.read_data(config, scenario)
   post_process(df_orig=df_orig, subdir=subdir, fields=['D_2020'])
   post_process(df_orig=df_orig, subdir=subdir, fields=['D_2016', 'D_2020'])
   
if __name__ == '__main__':
   main()