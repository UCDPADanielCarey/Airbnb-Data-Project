##Code of Airbnb Data Analytics Project by Daniel Carey

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import statistics as stats
import geopandas as gpd
import numpy as np
import shapely.geometry as shp
import sklearn
import requests 
import lxml.html as lh

from sklearn import preprocessing
#import fiona
sns.set_style(style= "darkgrid")

#Here I am creating the data frames
lisbon_airbnb = pd.read_csv(r"C:\Users\danie\OneDrive\Documents\Airbnb\lisbon_listings.csv")
pd.DataFrame(lisbon_airbnb)
print(lisbon_airbnb)

lisbon_airbnb.dropna()

madrid_airbnb = pd.read_csv(r"C:\Users\danie\OneDrive\Documents\Airbnb\madrid_listings.csv")
pd.DataFrame(madrid_airbnb)
print(madrid_airbnb)
madrid_airbnb.dropna()

#Merging data
Lis_V_Mad = pd.merge(lisbon_airbnb, madrid_airbnb, how='inner', on='price')

Lis_V_Mad.groupby(['price']).count()['price'].sort_index(ascending=False)

Lis_V_Mad.dropna()

lisbon_airbnb.describe() #from this I can see I probably won't need the IDs

lisbon_airbnb = lisbon_airbnb.drop(['id','host_id','reviews_per_month','name','host_name','license','last_review'], axis=1) 
lisbon_airbnb.head()


lisbon_airbnb.isnull().sum() #here I am checking if there are any null values

#lisbonVmadrid = pd.merge (room_type, how='inner')

#Groupby of rental types by area
lisbon_airbnb.groupby(['room_type']).count()['neighbourhood_group'].sort_index(ascending=False)

lisbon_airbnb.describe() #from this I can see I probably won't need the IDs

lisbon_airbnb = lisbon_airbnb.drop(['id','host_id','reviews_per_month','name','host_name','license','last_review'], axis=1) 
lisbon_airbnb.head()


lisbon_airbnb.isnull().sum() #here I am checking if there are any null values

#lisbonVmadrid = pd.merge (room_type, how='inner')

#Groupby of rental types by area
lisbon_airbnb.groupby(['room_type']).count()['neighbourhood_group'].sort_index(ascending=False)

#Now I want to see on average, the price per night in each area of Lisbon.
from statistics import mode,mean,median
jh = lisbon_airbnb.groupby('neighbourhood_group').agg({'price':'mean'}).sort_values('price',ascending=False)
jh.rename(columns={'price':'Average price'},inplace=True)
jh.reset_index(inplace=True)
plt.figure(figsize=(12,8))
order=lisbon_airbnb.neighbourhood.value_counts().iloc[:10].index
sns.barplot(data=jh,y='neighbourhood_group',x='Average price',palette='colorblind') 
plt.title('Average accommodation price in Lisbon by area')
sns.despine()

#The types of accommodation available on Airbnb in Lisbon.
def bar (col):
    plt.figure (figsize = (15,8))
    ax = sns.barplot (x = col.value_counts ().index, y = col.value_counts ())
    if col.dtypes != "object":
        ax.set_xlim (outlier (col))
plt.title ('Airbnb types available in Lisbon')      
bar (lisbon_airbnb ["room_type"])
box (lisbon_airbnb ["price"])
hist (lisbon_airbnb ["price"])
lisbon_airbnb ["room_type"].value_counts (normalize = True)

#I'll now run the below to see where Airbnbs of over 100 per night are located.
lisbon_airbnb['geometry']=lisbon_airbnb[['longitude','latitude']].apply(shp.Point, axis=1)
lisbon_airbnb=gpd.GeoDataFrame(lisbon_airbnb)
lisbon_airbnb.crs={'init': 'epsg:4329'}
lisbon_airbnb[lisbon_airbnb['price']<100].plot.scatter(x='longitude', y='latitude', c='price', figsize=(10,10), cmap='icefire_r', alpha=0.5);


#Conparing Lisbon prices with Madrid
def plot_multiple_hist(lisbon_airbnb,madrid_airbnb, col, thresh = None, bins = 30): 
    if thresh:
        data1 = lisbon_airbnb[col][lisbon_airbnb[col]< thresh]
        data2 = madrid_airbnb[col][madrid_airbnb[col]< thresh]
    else:
        data1 = lisbon_airbnb[col]
        data2 = madrid_airbnb[col]
    data1.hist(bins = bins, alpha = 0.5, label='Lisbon', color = '#2B30B0')
    data2.hist(bins = bins, alpha = 0.5, label='Madrid', color = '#F065D5')
    plt.legend(fontsize=20)
    plt.xlabel(col, fontsize=20)
    plt.ylabel("No. of Airbnb Properties", fontsize=20)
    if thresh :
        plt.title("{} histogram (< {} )".format(col, thresh), fontsize=20)
    else:
        plt.title(col+ " histogram")
   # plt.savefig("figures/price_histogram.png", dpi = 600,  bbox_inches='tight')
plot_multiple_hist(lisbon_airbnb,madrid_airbnb, "price", thresh=500)


#Collaration Matrix
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(lisbon_airbnb.corr(),annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)
plt.show()

#Make sure the target variable (‘price’) is :Normally distributed, kurtosis and skewness are normal.
sns.distplot(lisbon_airbnb['price'], kde=True,);
fig = plt.figure()
res = stats.probplot(lisbon_airbnb['price'], plot=plt)
print("Skewness: %f" % lisbon_airbnb['price'].skew())
print("Kurtosis: %f" % lisbon_airbnb['price'].kurt())

#Modelling
#reviews vs price
sns.lmplot(x='number_of_reviews',y='price', data = lisbon_airbnb)
plt.xlabel('The number of reviews per property')
plt.ylabel('Price in €')
plt.show

# Reviews per Month vs Price
sns.lmplot(x = 'reviews_per_month', y = 'price', lowess = True, 
           scatter=False,
           hue = 'room_type', data = lisbon_airbnb)
plt.title('Reviews per Month vs Price by Room Type');
my_colours = ["skyblue", "red", "orange", "purple"]
sns.set_palette(my_colours)

# Reviews per Month vs Price
sns.lmplot(x = 'reviews_per_month', y = 'price', lowess = True, 
           scatter=False,
           hue = 'room_type', data = lisbon_airbnb)
plt.title('Reviews per Month vs Price by Room Type');
my_colours = ["skyblue", "red", "orange", "purple"]
sns.set_palette(my_colours)


# Availability vs Price
sns.lmplot(x = 'availability_365', y = 'price', lowess = True, 
           scatter=False,
           hue = 'room_type', data = lisbon_airbnb)
plt.title('Availability vs Price by Room Type');










