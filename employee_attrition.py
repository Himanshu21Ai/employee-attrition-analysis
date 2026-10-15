import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;



df = pd.read_csv("dataset/IBM.csv")

#understanding the data
head=df.head()
shape=df.shape
cloumn=df.columns
info=df.info()

#cleaning the data 
nullvalue=df.isnull().sum() #aka missing values
duplicate=df.duplicated().sum()
datatypes=df.dtypes
summary=df.describe()
specific=df["MonthlyIncome"].describe()

#analyzing the data for attirtion
total=df["Attrition"].value_counts() #ATTRITION COUNT or total employee that are left
department=df.groupby("Department")["Attrition"].value_counts() #each department wise attirtion
salary=df.groupby("Attrition")["MonthlyIncome"].mean() #itmeans does salary was the reason of attirtion
worklife=df.groupby("WorkLifeBalance")["Attrition"].value_counts() #does work life affect
jobsatifaction=df.groupby("JobSatisfaction")["Attrition"].value_counts() #does low job satisfaction has more attirtion

#eda
salaryavg=(df.groupby("Attrition")["MonthlyIncome"].mean())
ageavg=(df.groupby("Attrition")["Age"].mean())
locationavg=(df.groupby("Attrition")["DistanceFromHome"].mean())
yearsavg=(df.groupby("Attrition")["YearsAtCompany"].mean())

#vizulization of data for attirtion

df["Attrition"].value_counts().plot(kind="bar")
plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.tight_layout()


plt.figure()
df.groupby("Department")["Attrition"].value_counts().unstack().plot(kind="bar")
plt.title("Department Wise Attrition")
plt.xlabel("Department")
plt.ylabel("Employee Count")
plt.tight_layout()

df.groupby("JobSatisfaction")["Attrition"].value_counts().unstack().plot(kind="bar")
plt.title("Attrition vs Job Satisfaction")

df.groupby("WorkLifeBalance")["Attrition"].value_counts().unstack().plot(kind="bar")
plt.title("attirtion VS worklifebalance")

plt.figure(figsize=(8,5))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")

plt.show()


