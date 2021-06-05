#!/usr/bin/env python
# coding: utf-8

# #### 李坤 3220201059
# 1.数据读取


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

sns.set(style="whitegrid")

# 读取数据集
df = pd.read_csv("data/hotel_bookings.csv")

# 查看数据缺失情况
df.isnull().sum()[df.isnull().sum()!=0]

# 缺失值处理
df.drop("company", axis=1, inplace=True)
df["agent"].fillna(0, inplace=True)
df["children"].fillna(0.0, inplace=True)
df["country"].fillna("Unknown", inplace=True)

df.isnull().sum()

# meal属性值同义替换
df["meal"].replace("Undefined", "SC", inplace=True)

df["meal"].value_counts()

# 异常值处理
zero_guests = list(df["adults"]
                   + df["children"]
                   + df["babies"]==0)
df.drop(df.index[zero_guests], inplace=True)

base = df.copy()

# 整体入住情况分析
base["is_canceled"] = base["is_canceled"].astype(object)
base["is_canceled"].replace(0, "入住", inplace=True)
base["is_canceled"].replace(1, "取消", inplace=True)

total_booking = base["is_canceled"].value_counts()

fig = px.pie(total_booking,
             values=total_booking.values,
             names=total_booking.index,
             title="整体入住情况",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

# 预订需求对比
booking = base["hotel"].value_counts()

fig = px.pie(booking,
             values=booking.values,
             names=booking.index,
             title="两种酒店的预订需求比较",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

#入住率对比
rh = base[base['hotel']=='Resort Hotel']
ch = base[base['hotel']=='City Hotel']

rh_checkin = rh["is_canceled"].value_counts()
fig = px.pie(rh_checkin,
             values=rh_checkin.values,
             names=rh_checkin.index,
             title="度假酒店入住情况",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

ch_checkin = ch["is_canceled"].value_counts()
fig = px.pie(ch_checkin,
             values=ch_checkin.values,
             names=ch_checkin.index,
             title="城市酒店入住情况",
             template="seaborn")
fig.update_traces(rotation=-90, textinfo="value+percent+label")
fig.show()

plt.figure(figsize=(8,6))
sns.countplot(x='hotel', hue='is_canceled', data=df)
plt.show()

#入住/取消
check_in = df[df["is_canceled"]==0]
cancel = df[df["is_canceled"]==1]

#度假酒店/城市酒店
res = df[df['hotel']=='Resort Hotel']
cty = df[df['hotel']=='City Hotel']

#度假酒店入住/城市酒店入住
res_checkin = df.loc[(df["hotel"] == "Resort Hotel") & (df["is_canceled"] == 0)]
cty_checkin = df.loc[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 0)]


df['lead_time'].describe()


# 关于提前入住时长，频数排在前30位的绝大多数都在50天以内。
# 
# 很大部分客户的提前预订时长为0，最长的时间为提前737天预订。

#频数分布柱状图
plt.figure(figsize=(16,18))
plt.subplots_adjust(hspace=0.7)

plt.subplot(3,2,1)
plt.title('Days of booking in advance -- Check in')
check_in['lead_time'].value_counts().head(20).plot.bar() 

plt.subplot(3,2,2)
plt.title('Days of booking in advance -- Cancel')
cancel['lead_time'].value_counts().head(20).plot.bar()

plt.subplot(3,2,3)
plt.title('Days of booking in advance -- Resort Hotel')
res['lead_time'].value_counts().head(20).plot.bar()

plt.subplot(3,2,4)
plt.title('Days of booking in advance -- City Hotel')
cty['lead_time'].value_counts().head(20).plot.bar()

plt.subplot(3,2,5)
plt.title('Days of booking in advance -- Check in Resort Hotel')
res_checkin['lead_time'].value_counts().head(20).plot.bar()

plt.subplot(3,2,6)
plt.title('Days of booking in advance -- Check in City Hotel')
cty_checkin['lead_time'].value_counts().head(20).plot.bar()

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
check_in['lead_time'].plot(kind='hist', title='lead_time / Check in', bins=30)

plt.subplot(1,2,2)
cancel['lead_time'].plot(kind='hist', title='lead_time / Cancel', bins=30)

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
res['lead_time'].plot(kind='hist', title='lead_time -- Resort Hotel', bins=30)

plt.subplot(1,2,2)
cty['lead_time'].plot(kind='hist', title='lead_time -- City Hotel', bins=30)

#只考虑入住数据
rh = df.loc[(df["hotel"] == "Resort Hotel") & (df["is_canceled"] == 0)]
ch = df.loc[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 0)]

# 提取出入住时长相关数据
rh["total_nights"] = rh["stays_in_weekend_nights"] + rh["stays_in_week_nights"]
ch["total_nights"] = ch["stays_in_weekend_nights"] + ch["stays_in_week_nights"]

num_nights_res = list(rh["total_nights"].value_counts().index)
num_bookings_res = list(rh["total_nights"].value_counts())
rel_bookings_res = rh["total_nights"].value_counts() / sum(num_bookings_res) * 100 # 转换为百分比

num_nights_cty = list(ch["total_nights"].value_counts().index)
num_bookings_cty = list(ch["total_nights"].value_counts())
rel_bookings_cty = ch["total_nights"].value_counts() / sum(num_bookings_cty) * 100 

res_nights = pd.DataFrame({"hotel": "Resort hotel",
                           "num_nights": num_nights_res,
                           "rel_num_bookings": rel_bookings_res})

cty_nights = pd.DataFrame({"hotel": "City hotel",
                           "num_nights": num_nights_cty,
                           "rel_num_bookings": rel_bookings_cty})

nights_data = pd.concat([res_nights, cty_nights], ignore_index=True)

plt.figure(figsize=(16, 8))
sns.barplot(x = "num_nights", y = "rel_num_bookings", hue="hotel", data=nights_data,
            hue_order = ["City hotel", "Resort hotel"])
plt.title("Length of stay", fontsize=16)
plt.xlabel("Number of nights", fontsize=16)
plt.ylabel("Guests [%]", fontsize=16)
plt.legend(loc="upper right")
plt.xlim(0,22)
plt.show()

#频数分布柱状图
plt.figure(figsize=(16,18))
plt.subplots_adjust(hspace=0.7)

plt.subplot(3,2,1)
plt.title('Days of Booking interval -- Check in')
check_in[check_in['days_in_waiting_list'] > 0]['days_in_waiting_list'].value_counts().head(20).plot.bar() 

plt.subplot(3,2,2)
plt.title('Days of Booking interval -- Cancel')
cancel[cancel['days_in_waiting_list'] > 0]['days_in_waiting_list'].value_counts().head(20).plot.bar()

plt.subplot(3,2,3)
plt.title('Days of Booking interval -- Resort Hotel')
res[res['days_in_waiting_list'] > 0]['days_in_waiting_list'].value_counts().head(20).plot.bar()

plt.subplot(3,2,4)
plt.title('Days of Booking interval -- City Hotel')
cty[cty['days_in_waiting_list'] > 0]['days_in_waiting_list'].value_counts().head(20).plot.bar()

plt.subplot(3,2,5)
plt.title('Days of Booking interval -- Check in Resort Hotel')
res_checkin[res_checkin['days_in_waiting_list'] > 0]['days_in_waiting_list'].value_counts().head(20).plot.bar()

plt.subplot(3,2,6)
plt.title('Days of Booking interval -- Check in City Hotel')
cty_checkin[cty_checkin['days_in_waiting_list'] > 0]['days_in_waiting_list'].value_counts().head(20).plot.bar()

#考虑全部预订数据
rh = df.loc[(df["hotel"] == "Resort Hotel")]
ch = df.loc[(df["hotel"] == "City Hotel")]

# 提取出餐食预订相关数据
meal_res = list(rh["meal"].value_counts().index)
num_meal_res = list(rh["meal"].value_counts())
rel_meal_res = rh["meal"].value_counts() / sum(num_meal_res) * 100 # 转换为百分比

meal_cty = list(ch["meal"].value_counts().index)
num_meal_cty = list(ch["meal"].value_counts())
rel_meal_cty = ch["meal"].value_counts() / sum(num_meal_cty) * 100 

res_meals = pd.DataFrame({"hotel": "Resort hotel",
                           "meal_booking": meal_res,
                           "rel_num_bookings": rel_meal_res})

cty_meals = pd.DataFrame({"hotel": "City hotel",
                           "meal_booking": meal_cty,
                           "rel_num_bookings": rel_meal_cty})

meal_data = pd.concat([res_meals, cty_meals], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.barplot(x = "meal_booking", y = "rel_num_bookings", hue="hotel", data=meal_data,
            hue_order = ["City hotel", "Resort hotel"])
plt.title("Meal booking options", fontsize=16)
plt.xlabel("Kind of meal", fontsize=16)
plt.ylabel("Guests [%]", fontsize=16)
plt.legend(loc="upper right")
plt.xlim(0,4)
plt.show()

plt.figure(figsize = (10,6))
sns.countplot(x='meal',hue = 'hotel',data=base)

# 整体的月度人流量

plt.figure(figsize = (10,5))
base.groupby(['arrival_date_month'])['arrival_date_month'].count().plot.bar()

df["adr_pp"] = df["adr"] / (df["adults"] + df["children"])
full_data_guests = df.loc[df["is_canceled"] == 0]
rh = df.loc[(df["hotel"] == "Resort Hotel") & (df["is_canceled"] == 0)]
ch = df.loc[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 0)]

# 获取数据:
room_prices_mothly = full_data_guests[["hotel", "arrival_date_month", "adr_pp"]].sort_values("arrival_date_month")

# 根据月份排序:
ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
room_prices_mothly["arrival_date_month"] = pd.Categorical(room_prices_mothly["arrival_date_month"], categories=ordered_months, ordered=True)

plt.figure(figsize=(12, 8))
sns.lineplot(x = "arrival_date_month", y="adr_pp", hue="hotel", data=room_prices_mothly, 
            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))
plt.title("Room price per night and person over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Price [EUR]", fontsize=16)
plt.show()

# 月度人流量
resort_guests_monthly = rh.groupby("arrival_date_month")["hotel"].count()
city_guests_monthly = ch.groupby("arrival_date_month")["hotel"].count()

resort_guest_data = pd.DataFrame({"month": list(resort_guests_monthly.index),
                    "hotel": "Resort hotel", 
                    "guests": list(resort_guests_monthly.values)})

city_guest_data = pd.DataFrame({"month": list(city_guests_monthly.index),
                    "hotel": "City hotel", 
                    "guests": list(city_guests_monthly.values)})
full_guest_data = pd.concat([resort_guest_data,city_guest_data], ignore_index=True)

# 根据月份排序:
ordered_months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
full_guest_data["month"] = pd.Categorical(full_guest_data["month"], categories=ordered_months, ordered=True)

# 数据集包含3年的7月和8月，2年的其他月份:
full_guest_data.loc[(full_guest_data["month"] == "July") | (full_guest_data["month"] == "August"),
                    "guests"] /= 3
full_guest_data.loc[~((full_guest_data["month"] == "July") | (full_guest_data["month"] == "August")),
                    "guests"] /= 2

plt.figure(figsize=(12, 8))
sns.lineplot(x = "month", y="guests", hue="hotel", data=full_guest_data, 
             hue_order = ["City hotel", "Resort hotel"], size="hotel", sizes=(2.5, 2.5))
plt.title("Average number of hotel guests per month", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of guests", fontsize=16)
plt.show()

plt.figure(figsize = (12,8))
sns.countplot(x='arrival_date_month',hue = 'hotel',data=base)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data/hotel_bookings.csv")

df.drop(['agent','company','country'],inplace=True, axis=1)
df['children'].fillna(0,inplace=True)

cat=df.select_dtypes(include='object').columns
cat

#使用Label Encoder对非数字项进行编码
encode = LabelEncoder()
df['arrival_date_month'] = encode.fit_transform(df['arrival_date_month'])
df['meal'] = encode.fit_transform(df['meal'])
df['market_segment'] = encode.fit_transform(df['market_segment'])
df['distribution_channel'] = encode.fit_transform(df['distribution_channel'])
df['reserved_room_type'] = encode.fit_transform(df['reserved_room_type'])
df['assigned_room_type'] = encode.fit_transform(df['assigned_room_type'])
df['deposit_type'] = encode.fit_transform(df['deposit_type'])
df['customer_type'] = encode.fit_transform(df['customer_type'])
df['reservation_status'] = encode.fit_transform(df['reservation_status'])

#使用map函数将year转换为编码值
df['arrival_date_year'] = df['arrival_date_year'].map({2015:1, 2016:2, 2017:3})

#缩小lead_time和adr的映射范围
scaler = MinMaxScaler()
df['lead_time'] = scaler.fit_transform(df['lead_time'].values.reshape(-1,1))
df['adr'] = scaler.fit_transform(df['adr'].values.reshape(-1,1))


#计算相关系数
df.corr()

# 选取与is_canceled相关性较强的属性
data = df[['reservation_status','total_of_special_requests','required_car_parking_spaces',
           'deposit_type','booking_changes','assigned_room_type','previous_cancellations',
           'distribution_channel','lead_time','is_canceled']]

# 划分数据集
X = data.drop(['is_canceled'],axis= 1)
y = data['is_canceled']


# Logistics回归
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# 准确率
accuracy = logreg.score(X_test,y_test)
print(accuracy)

# 混淆矩阵
matrix = confusion_matrix(y_test, y_pred.round())
matrix

# 预测结果
y_pred

df.loc[y_test.index]

result=pd.DataFrame()
result['Hotel Name']=df.loc[y_test.index].hotel
result['Booking_Possibility']=y_pred

# 将结果存储为csv文件
result.to_csv('pred.csv')