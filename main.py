# %%
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from functools import partial
from geopy.geocoders import Nominatim
from IPython import get_ipython


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pycountry
import pycountry_convert as pc
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# %%

# %% [markdown]
# # Data load

# %%
data = pd.read_csv('ramen-ratings.csv')
data.head()

# %% [markdown]
# # Preprocessing

# %%
data.shape


# %%
data.info()

# %% [markdown]
# Two missing values in column 'Style'.

# %%
data.dtypes


# %%
data[['Brand', 'Variety', 'Style', 'Country', 'Top Ten']].nunique()


# %%
# To lower case
data.Brand = data.Brand.str.lower()
data.Variety = data.Variety.str.lower()
data.Style = data.Style.str.lower()
data.Country = data.Country.str.lower()

# %% [markdown]
# ### Stars

# %%
data.Stars.sort_values().unique()


# %%
print('Procentage of `Unrated` ramens : {}%'.format(
    round(np.sum(data.Stars.isin(['Unrated']))*100/len(data), 2)))


# %%
# Setting all ratings to float and filling with zeros if 'Unrated' or missed value
data['Stars'] = pd.to_numeric(data.Stars, errors='coerce')
data['Stars'] = data.Stars.fillna(0)
data['Stars'] = data.Stars.astype('float')

# Rounding ratings to equal amount of decimal places
data['Stars'] = np.around(data.Stars, decimals=1)

# %% [markdown]
# ### Style

# %%
data.Style.unique()


# %%
data[data.Style.isnull()]


# %%
# Filling 2 missing ramen styles after online check
data['Style'] = np.where(data.Style.isnull(), 'pack', data.Style)

# %% [markdown]
# ### Country

# %%
data.Country.unique()


# %%
geolocator = Nominatim(timeout=20, user_agent="pg-ramen-ppir/1.0")
geocode = partial(geolocator.geocode, language="en")


# %%
# Unify country names
countries = list(map(lambda c: geocode(c).address.split(', ')
                     [-1], list(data.Country.unique())))

fix_dict = {k: v for k, v in zip(
    list(data.Country.unique()), countries) if k != v}


# %%
data.Country.replace(fix_dict, inplace=True)


# %%
data.Country.unique()


# %%
data.head()

# %% [markdown]
# # Data Analysis

# %%
data.Stars.describe()

# %% [markdown]
# ## Ratings

# %%
rat = plt.subplots(figsize=(7, 5))
rat = data.Stars.hist(color='limegreen')
plt.xlabel('Rating')
plt.ylabel('Number of Ramen')
plt.title('Rating Histogram')
plt.show()


# %%
data['Stars'].median()


# %%
round(data['Stars'].mean(), 2)

# %% [markdown]
# #### Ramen has average rating of 3.7 star
# %% [markdown]
# ## Style
# %% [markdown]
# #### Most eaten Style of Ramen

# %%
styles = dict(data['Style'].value_counts())
plt.bar(range(len(styles)), list(styles.values()),
        align='center', color='green')
plt.xticks(range(len(styles)), list(styles.keys()))
plt.ylabel('Number of Eaten Ramens')
plt.xlabel('Ramen Style')
plt.title('Most Eaten Style of Ramen')
plt.show()

# %% [markdown]
# ### Ratings in each ramen style

# %%
pack_freq = data['Stars'].loc[data['Style'] == 'pack'].astype(float)
pack_freq.plot(kind='hist', color='lightpink')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
plt.title('Pack Style')
plt.show()


# %%
bowl_freq = data['Stars'].loc[data['Style'] == 'bowl'].astype(float)
bowl_freq.plot(kind='hist', color='lightblue')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
plt.title('Bowl Style')
plt.show()


# %%
cup_freq = data['Stars'].loc[data['Style'] == 'cup'].astype(float)
cup_freq.plot(kind='hist', color='wheat')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
_ = plt.title('Cup Style')
plt.show()


# %%
tray_freq = data['Stars'].loc[data['Style'] == 'tray'].astype(float)
tray_freq.plot(kind='hist', color='lavender')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
plt.title('Tray Style')
plt.show()


# %%
box_freq = data['Stars'].loc[data['Style'] == 'box'].astype(float)
box_freq.plot(kind='hist', color='powderblue')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
plt.title('Box Style')
plt.show()


# %%
bar_freq = data['Stars'].loc[data['Style'] == 'bar'].astype(float)
bar_freq.plot(kind='hist', color='seashell')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
plt.title('Bar Style')
plt.show()


# %%
can_freq = data['Stars'].loc[data['Style'] == 'can'].astype(float)
can_freq.plot(kind='hist', color='slategray')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Ratings')
plt.title('Can Style')
plt.show()

# %% [markdown]
# ## Country
# %% [markdown]
# #### Frequency of stars given in each *country*

# %%
c_dicts = []
c_names = []
countries = data.Country.unique()
for c in countries:
    c_dicts.append(dict(data.Style.loc[data.Country == c].value_counts()))
    c_names.append(c)


# %%
def get_x(country_dict):
    x = []
    for key, value in country_dict.items():
        x.append(key)
    return x


def get_y(country_dict):
    y = []
    for key, value in country_dict.items():
        y.append(value)
    return y


colors = ['aquamarine', 'palegreen', 'plum']

plt.figure(figsize=(15, 25), facecolor='white')
plot_number = 1
for i in range(18):
    ax = plt.subplot(6, 3, plot_number)
    ax.bar(get_x(c_dicts[i]), get_y(c_dicts[i]), color=colors[i % 3])
    ax.set_title(c_names[i])
    plot_number = plot_number + 1

plt.figure(figsize=(15, 25), facecolor='white')
plot_number = 1
for i in range(19, len(data.Country.unique())):
    ax = plt.subplot(6, 3, plot_number)
    ax.bar(get_x(c_dicts[i]), get_y(c_dicts[i]), color=colors[i % 3])
    ax.set_title(c_names[i])
    plot_number = plot_number + 1

# %% [markdown]
# #### Number of participants in each *country*

# %%
country_counts = dict(data.Country.value_counts())
countries = get_x(country_counts)
counts = get_y(country_counts)


# %%
plt.figure(figsize=(15, 20))
plt.barh(countries, counts, color='mediumorchid')
plt.xlabel('Number of Participants')
plt.ylabel('Countries')
plt.title('Number of Participants in each country')
plt.show()

# %% [markdown]
# ## Brands
# %% [markdown]
# #### Top 10 best *Brands*

# %%
best_brands = dict(
    data.Brand.loc[data.Stars.astype(float) > 3.0].value_counts())
brand_names = list(best_brands.keys())
brand_apperance_count = list(best_brands.values())
plt.figure(figsize=(10, 5))
plt.barh(brand_names[:10], brand_apperance_count[:10], color='limegreen')
plt.xlabel('Number of Good Reviews Recieved')
plt.ylabel('Brand name')
plt.grid(True)
plt.title('Top10 best brands')
plt.show()

# %% [markdown]
# #### Top 10 worst *Brands*

# %%
worst_brands = dict(
    data.Brand.loc[data.Stars.astype(float) <= 3.0].value_counts())
brand_names = list(worst_brands.keys())
brand_apperance_count = list(worst_brands.values())
plt.figure(figsize=(10, 5))
plt.barh(brand_names[:10], brand_apperance_count[:10], color='firebrick')
plt.xlabel('Number of Bad Reviews Recieved')
plt.ylabel('Brand name')
plt.grid(True)
plt.title('Top10 worst brands')
plt.show()

# %% [markdown]
# #### Procentage of good and bad ramen

# %%
x = [len(data.Brand.loc[data.Stars.astype(float) <= 3.0].value_counts()), len(
    data.Brand.loc[data.Stars.astype(float) > 3.0].value_counts())]
plt.pie(x, labels=['Bad', 'Good'], autopct='%1.2f',
        startangle=90, colors=['firebrick', 'limegreen'])
plt.axis('equal')
plt.title('Procentage of good & bad ramen')
plt.show()

# %% [markdown]
# #### Number of Bad and Good ratings in each serving style

# %%
ax = plt.subplot(111)
good = ax.bar(get_x(data.Style.loc[data.Stars.astype(float) > 3.0].value_counts()), get_y(
    data.Style.loc[data.Stars.astype(float) > 3.0].value_counts()), color="limegreen")
bad = ax.bar(get_x(data.Style.loc[data.Stars.astype(float) <= 3.0].value_counts()), get_y(
    data.Style.loc[data.Stars.astype(float) <= 3.0].value_counts()), color="firebrick")
ax.legend((good, bad), ('Good', 'Bad'))
plt.xlabel('Style')
plt.ylabel('Number of ratings')
plt.title('Bad & Good ratings in each serving style')
plt.show()

# %% [markdown]
# #### Number of brands serving ramen in specyfic style

# %%
choices = {style: len(
    data.Brand.loc[data.Style == style].unique()) for style in data.Style.unique()}

plt.barh(get_x(choices), get_y(choices), color='mediumpurple')
plt.xlabel('Number of Brands')
plt.ylabel('Style')
plt.show()

# %% [markdown]
# ## World map

# %%
world_df = pd.DataFrame()
world_df['Country'] = data.Country.sort_values().unique()
world_df['Style'] = pd.Series(map(lambda x: data[data.Country.isin([x])].groupby(
    'Style').Stars.mean().sort_values(ascending=False).idxmax(), data.Country.sort_values().unique()))
world_df['Avg rating'] = pd.Series(map(lambda x: round(data[data.Country.isin([x])].groupby(
    'Style').Stars.mean().sort_values(ascending=False).max(), 1), data.Country.sort_values().unique()))
world_df['Popular style'] = pd.Series(map(lambda x: data.Style.where(
    data.Country == x).value_counts().idxmax(), data.sort_values(by='Country').Country.unique()))
world_df['ISO'] = pd.Series(map(lambda x: pc.country_name_to_country_alpha3(
    x), data.Country.sort_values().unique()))
world_df['Continent'] = pd.Series(map(lambda x: pc.convert_continent_code_to_continent_name(
    pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(x))), data.Country.sort_values().unique()))


# %%

fig = px.choropleth(world_df, locations="ISO", color="Style", hover_name='Country', hover_data=[
                    'Style', 'Avg rating', 'Popular style'], projection="natural earth",)
fig.update_layout(title_text='Highest rated ramen style of the world')
fig.show()

# %% [markdown]
# # Machine Learning
# %% [markdown]
# ## PyTorch

# %%
data['Stars'] = data['Stars'].astype(float)

Country = pd.get_dummies(data['Country'], prefix='Country', drop_first=True)
Brand = pd.get_dummies(data['Brand'], prefix='Brand', drop_first=True)
Style = pd.get_dummies(data['Style'], prefix='Style', drop_first=True)

datadf = pd.concat([Country, Brand, Style], axis=1)

X = np.array(datadf, dtype=np.float32)
y = np.array(data[['Stars']], dtype=np.float32)
m = nn.Linear(391, 1)

loss = nn.MSELoss()

opt = torch.optim.SGD(m.parameters(), lr=0.6)
for dataepoch in range(1000):
    datain = torch.from_numpy(X)
    datatrgs = torch.from_numpy(y)

    dataout = m(datain)
    cost = loss(dataout, datatrgs)

    opt.zero_grad()
    cost.backward()
    opt.step()

    if (dataepoch+1) % 100 == 0:
        print(
            'Epoch for ramen-ratings.csv [{}/{}]\nLoss: {:.4f}'.format(dataepoch+1, 1000, cost.item()))

y_predict = m(torch.from_numpy(X)).data.numpy()

print(y_predict[0:5])
print(y[0:5])

metrics.mean_squared_error(y, y_predict)


# %%
datadf.head()

# %% [markdown]
# ## AdaBoost, Elbow Method, k-means clustering

# %%
dummies = pd.get_dummies(data[["Brand", "Variety", "Style", "Country"]])

stars = data[["Stars"]]
data3 = pd.concat((dummies, stars), axis=1)


# %%
res = list()
n_cluster = range(2, 15)
for n in n_cluster:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data3)
    res.append(np.average(
        np.min(cdist(data3, kmeans.cluster_centers_, 'euclidean'), axis=1)))

plt.plot(n_cluster, res)
plt.title('Elbow Curve')

kmeans_model = KMeans(n_clusters=7, init='k-means++',
                      max_iter=300, n_init=10, random_state=0).fit(data3)
results = kmeans_model.predict(data3)
centroids = kmeans_model.cluster_centers_
kmeans_n = pd.DataFrame(data=results)
data["Score"] = kmeans_n
res = kmeans_model.__dict__

res2 = pd.DataFrame.from_dict(res, orient='index')


# %%
data4 = data3
H_cluster = linkage(data4, 'ward')
cluster_2 = fcluster(H_cluster, 5, criterion='maxclust')
cluster_Hierarchical = pd.DataFrame(cluster_2)
data4.to_csv("tree.csv")


# %%
X = data3.drop(["Stars"], axis=1)
data3["Stars"] = pd.to_numeric(data3["Stars"], errors='coerce')
y = data3["Stars"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())
X_test = X_test.fillna(X_test.mean())
y_test = y_test.fillna(y_test.mean())
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# %%
X = data3.drop(["Stars"], axis=1)
data3["Stars"] = pd.to_numeric(data3["Stars"], errors='coerce')
y = data3["Stars"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
abc = AdaBoostClassifier(n_estimators=500, learning_rate=1)
X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())
X_test = X_test.fillna(X_test.mean())
y_test = y_test.fillna(y_test.mean())
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
