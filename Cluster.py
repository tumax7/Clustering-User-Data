import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, OPTICS
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# Importing data
raw_data = pd.read_csv('data.csv',
                       parse_dates=['date'],
                       converters={"cookie_id": str.strip, "session_id": str.strip, 'event_type': str.strip,
                                   "domain_id": str.strip},
                       usecols=["date", "cookie_id", "session_id", 'event_type', "domain_id"])

# Getting time for each session
pivoted_data = pd.pivot_table(raw_data, index=['cookie_id', 'domain_id', 'session_id'],
                              aggfunc={'date': [max, min]})

average_session_length = (pivoted_data['date']['max'] - pivoted_data['date']['min'])
average_session_length = average_session_length.apply(lambda x: x.total_seconds())

# There is a typo in the event types, such as 'click' as 'clcik' and 'Choose_flat' and 'choose_flat'
raw_data['event_type'] = raw_data['event_type'].str.lower()
raw_data['event_type'] = raw_data['event_type'].str.replace('clcik', 'click')

# Also let us group actions, which have start and finish columns into the instance of the action
raw_data['event_type'] = raw_data['event_type'].str.replace('_start', '')
raw_data['event_type'] = raw_data['event_type'].str.replace('_exit', '')

# Also whatsup, Viber and chat columns are not well defined, hence lets group them together
raw_data['event_type'] = raw_data['event_type'].str.replace('whatsup', 'chat')
raw_data['event_type'] = raw_data['event_type'].str.replace('viber', 'chat')

# As some columns are mostly empty I decided to group them into 'misc' column, I picked these columns as they had
# less than 30000 values

raw_data['event_type'] = raw_data['event_type'].str.replace('change_slider', 'misc')
raw_data['event_type'] = raw_data['event_type'].str.replace('click_scbform_button_callrequest', 'misc')
raw_data['event_type'] = raw_data['event_type'].str.replace('submit_flat_applicaiton_click', 'misc')
raw_data['event_type'] = raw_data['event_type'].str.replace('chat', 'misc')
raw_data['event_type'] = raw_data['event_type'].str.replace('taxi', 'misc')
raw_data['event_type'] = raw_data['event_type'].str.replace('choose_dream_flat_click_podrobnee', 'misc')

# Constructing a new table
pivoted_data = pd.pivot_table(raw_data, index=['cookie_id', 'domain_id', 'session_id'], columns='event_type',
                              aggfunc='count')

# adding session length and removing excess levels
pivoted_data.columns = pivoted_data.columns.droplevel()
pivoted_data['activity'] = pivoted_data.sum(axis=1)
pivoted_data['average_session_length'] = average_session_length
tot_col = pivoted_data.columns

# Removing sessions, which were not registered properly ( 0 seconds) and removing very short sessions as they are
# performed by bots, also removing very large lengths, as it is highly unlikely that people will stay on the website
# for longer than 10 minutes

pivoted_data = pivoted_data[pivoted_data['average_session_length'] < 600]
pivoted_data = pivoted_data[10 < pivoted_data['average_session_length']]
# Testing our average, which should be between 2-3 minutes, according to google analytics

# print(np.mean(pivoted_data['average_session_length']))

# After removing outliers and performing log transformations we can see that our session_length, is becoming normal

plt.hist(np.log(pivoted_data['average_session_length']))
plt.show()

# Constructing dictionary for the aggregation function as we don not want to sum all of them
# Our column with time must be last
func = [np.sum] * len(pivoted_data.columns) + [np.mean]
aggrfunction = dict(zip(pivoted_data.columns, func))

# Pivoting data again and filling NAs appearing as a result
clean_data = pd.pivot_table(pivoted_data, index='cookie_id', columns='domain_id', values=pivoted_data.columns,
                            aggfunc=aggrfunction)
clean_data.fillna(0, inplace=True)

# Removing the extra column level of the pivot table and converting to DataFrame
clean_data = pd.DataFrame(clean_data.to_records())
clean_data.set_index('cookie_id', inplace=True)

# Splitting Columns by events
x = [[] for _ in range(len(tot_col))]
for i in clean_data.columns:
    for c, j in enumerate(tot_col):
        if j in i:
            x[c].append(i)

# Generating columns and summing across events across multiple domains
for c, i in enumerate(x):
    clean_data['total_{}'.format(tot_col[c])] = [0] * clean_data.shape[0]
    for j in i:
        clean_data['total_{}'.format(tot_col[c])] += clean_data[j]

# Scaling Data with np.log and plotting the results, we can see that without the 0 results, which means person did
# not spend time on the website, the data is normal.

for i in clean_data.columns:
    if 'average_session_length' in i:
        clean_data[i] += 1
        clean_data[i] = np.log(clean_data[i])

# Now we have a lot of features, lets remove features which have very little change in them (mostly 0),
# This has removed 5 columns
var_df = pd.DataFrame(clean_data.var(), index=clean_data.columns, columns=['Variance'])
clean_data.drop(var_df[var_df['Variance'] < 0.01].index, axis=1, inplace=True)
print(clean_data.shape)

# Doing initial fit of the data for K-Means for further feature selection 7 Clusters
# kelbow_visualizer(KMeans(random_state=1), clean_data, k=(2, 15))
# plt.show()

# Initial K-Means initialisation for feature selection, by fitting classification model
km = KMeans(n_clusters=7, random_state=1).fit(clean_data)
prim = permutation_importance(km, clean_data, None, n_repeats=10, random_state=1, n_jobs=-1)
feature_importance = pd.DataFrame((prim.importances_mean / sum(prim.importances_mean)), index=clean_data.columns,
                                  columns=['Importance'])

# Dropping features barely used by the model, we are dropping 61 features in this process, hence selecting only
# informative features
clean_data.drop(feature_importance[feature_importance['Importance'] < 0.0001].index, axis=1, inplace=True)

km = KMeans(n_clusters=7, random_state=1).fit_predict(clean_data)

# Visualising clusters in a 2 dimensional space using PCA
pca = PCA(n_components=2, random_state=1).fit_transform(clean_data)
sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=km, palette='deep', size=1)
plt.show()

# From the shape we can see that the data is not spherical, hence K-means has a tough time, clustering it,
# therefore another algorithm should be used such as optics or dbscan

# But we can see that one group formed clearly only contains the outliers, hence we can just ignore this group and
# count it as the error group, assigning it -1 label, and carrying on
noise_dict = {0: 1, 1: -1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
km = list(map(lambda x: noise_dict.get(x), km))

# Saving results to dataframe
results = pd.DataFrame({'Cookie_Id': clean_data.index, 'Cluster Number': km})
results.to_csv('results.csv', index=False)

# Removing noise we can see that only 130 points were removed
clean_data['KM'] = km
clean_data = clean_data[clean_data.KM != -1]
data_no_km = clean_data.loc[:, clean_data.columns != 'KM']

# Plotting data without the noise
pca = PCA(n_components=2, random_state=1).fit_transform(data_no_km)
sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=clean_data['KM'], palette='deep')
plt.show()

# Let us now attempt to apply OPTICS algorithm on our cleaned dataset without noise

# First let us remove the K-means clusters from the dataset
# clean_data.drop(['KM'], axis=1, inplace=True)

# min_samples=82 is set as the data after feature selection has 41 dimensions
# eps=18.498287329980066 determined using k-neighbours plot and imported to increase speed
opt = OPTICS(min_samples=82, eps=18.498287329980066, n_jobs=-1).fit_predict(data_no_km)
clean_data['OPT'] = opt


# OPTICS algorithm has rejected most of the data (100000) and treated it as noise, while the rest 30000,
# it split into various groups of small sizes ranging from 1626 to 84 members. It can be seen on the plot bellow
print(clean_data['OPT'].value_counts())

# Therefore we will discard OPTICS and keep K-Means

# Interpreting clusters using Random Forest and extracting the most important features

# And graphically showing how the clusters were sorted based on important features #(0.5, 1.3)
rf = RandomForestClassifier(random_state=1, n_jobs=-1).fit(data_no_km, clean_data.KM)

selected_columns = list(
    pd.DataFrame(np.array([rf.feature_importances_, data_no_km.columns]).T, columns=['Importance', 'Feature'])
    .sort_values("Importance", ascending=False)
    .head(5)
    .Feature
    .values)

top_features = clean_data[selected_columns + ['KM']].melt(id_vars='KM')
sns.barplot(x='KM', y='value', hue='variable', data=top_features, palette='Set3')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
           ncol=2, fancybox=True, shadow=True)
plt.subplots_adjust(top=0.78)
plt.ylabel('Values')
plt.xlabel('KM Cluster')
plt.show()

# So our top features are
# total activity,
# total read,
# activity on 08c251e8-e56b-485b-b024-d5e82917ee6b (which I will call activity on domain 1)
# activity and read on 997d6180-cc73-4e0b-b02f-5e1e07211841 ((which I will call activity and read on domain 2)

# K-means has roughly clustered the data in these groups:

# 1 : average activity in total and domain 1, average reading event, low read and activity on domain 2
# 2 : high activity in total and domain 1, high reading event, low read and activity on domain 2
# 3 : everything about average apart from activity on domain 1
# 4 : everything very high (reaching maximum values) apart from activity on domain 1
# 5 : quite high total activity and total read, but everything else is low
# 6 : everything low, representing not avid computer users
