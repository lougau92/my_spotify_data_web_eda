import pandas as pd
import numpy as np
from datetime import datetime
import plotly
import plotly.express as px

def build_figures(data):
    df = pd.DataFrame(data)
    figures = []
    # for column in df.columns:
    #     if df[column].dtype in ["int64", "float64"]:
    #         fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    #         figures.append(plotly.offline.plot(fig, output_type="div"))

    return figures
