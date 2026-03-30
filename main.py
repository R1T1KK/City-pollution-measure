
# FINAL PROJECT - COMPLETE (GROUP + RESULT + DENDROGRAM)


!pip install pandas numpy matplotlib seaborn xy ipywidgets openpyxl

from google.colab import output
output.enable_custom_widget_manager()


# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram


# STEP 1: UPLOAD DATASET


from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]


# STEP 2: LOAD DATA

df = pd.read_excel(file_name)

print("Dataset Loaded ✅")
display(df.head())


# STEP 3: STANDARDIZATION

features = df[['PM2.5','PM10','NO2']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)


# STEP 4: FULL DENDROGRAM (ALL CITIES)

plt.figure(figsize=(12,5))
linked = linkage(scaled_data, method='ward')
dendrogram(linked, labels=df['Area_Name'].values, leaf_rotation=90)
plt.title("Full Dendrogram (All Cities)")
plt.show()
