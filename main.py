
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


# STEP 5: CLUSTERING

agglo = AgglomerativeClustering(n_clusters=3)
df['Agglo_Cluster'] = agglo.fit_predict(scaled_data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Div_Cluster'] = kmeans.fit_predict(scaled_data)


# STEP 6: LABEL CLUSTERS (LOW / MEDIUM / HIGH)

def label_clusters(cluster_col):
    avg = df.groupby(cluster_col)[['PM2.5','PM10','NO2']].mean()
    sorted_avg = avg.mean(axis=1).sort_values()

    labels = {}
    labels[sorted_avg.index[0]] = "Low"
    labels[sorted_avg.index[1]] = "Medium"
    labels[sorted_avg.index[2]] = "High"

    return df[cluster_col].map(labels)

df['Agglo_Level'] = label_clusters('Agglo_Cluster')
df['Div_Level'] = label_clusters('Div_Cluster')


# STEP 7: SELECT GROUPS (3 + 3)
group1 = widgets.SelectMultiple(options=df['Area_ID'], description="Group 1")
group2 = widgets.SelectMultiple(options=df['Area_ID'], description="Group 2")

display(group1, group2)


# STEP 8: PRINT FUNCTION
def print_level(name, level):
    if level == "High":
        print(f"{name} → 🔴 High")
    elif level == "Medium":
        print(f"{name} → 🟡 Medium")
    else:
        print(f"{name} → 🟢 Low")

# STEP 9: MAIN FUNCTION

def compare_groups(b):

    if len(group1.value) != 3 or len(group2.value) != 3:
        print("⚠️ Select exactly 3 cities in each group")
        return

    g1 = df[df['Area_ID'].isin(group1.value)]
    g2 = df[df['Area_ID'].isin(group2.value)]

    # ================= GROUP 1 =================
    print("\n===== GROUP 1 =====")
    for i, row in g1.iterrows():
        print_level(row['Area_Name'], row['Agglo_Level'])

    g1_scaled = scaler.transform(g1[['PM2.5','PM10','NO2']])

    plt.figure(figsize=(5,4))
    dendrogram(linkage(g1_scaled, method='ward'),
               labels=g1['Area_Name'].values)
    plt.title("Dendrogram - Group 1")
    plt.show()

    # ================= GROUP 2 =================
    print("\n===== GROUP 2 =====")
    for i, row in g2.iterrows():
        print_level(row['Area_Name'], row['Agglo_Level'])

    g2_scaled = scaler.transform(g2[['PM2.5','PM10','NO2']])

    plt.figure(figsize=(5,4))
    dendrogram(linkage(g2_scaled, method='ward'),
               labels=g2['Area_Name'].values)
    plt.title("Dendrogram - Group 2")
    plt.show()

    # ================= GROUP vs GROUP =================
    print("\n===== GROUP 1 vs GROUP 2 RESULT =====")

    # Mean values
    g1_mean = np.mean(g1_scaled, axis=0)
    g2_mean = np.mean(g2_scaled, axis=0)

    # Level decision (simple method)
    g1_avg = np.mean(g1[['PM2.5','PM10','NO2']].values)
    g2_avg = np.mean(g2[['PM2.5','PM10','NO2']].values)

    def group_level(val):
        if val < 60:
            return "Low"
        elif val < 110:
            return "Medium"
        else:
            return "High"

    g1_level = group_level(g1_avg)
    g2_level = group_level(g2_avg)

    print(f"Group 1 Level → {g1_level}")
    print(f"Group 2 Level → {g2_level}")

    # Similarity
    dist = np.linalg.norm(g1_mean - g2_mean)# linalg.norm used to claculate magnitude
    max_dist = np.linalg.norm(np.max(scaled_data,axis=0) - np.min(scaled_data,axis=0))
    similarity = round((1 - dist/max_dist)*100,2)

    print(f"Similarity → {similarity}%")

display(btn)
