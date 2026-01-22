import pandas as pd
df = pd.read_csv(r"22ndJan26\dataset.csv")
df_new = pd.read_csv("22ndJan26\depart.csv")

#print(df[
   # (df["Salary"] > 5000) &
   # (df["Department"] == "HR")
#])
#print(df["Salary"].max())
#print(df[df["Salary"] == df["Salary"].max()])
#print(df["Salary"].sort_values())
#df["Bonus"] = 5000
#df=df.drop(columns = "Bonus")

#filtered_df = df[
   # (df["Department"] == "HR") &
   # (df["Salary"] == df["Salary"] * 1.10)
#]
#print(filtered_df)
#df["Department"]= df["Department"].replace("HR",1)
#df.to_csv("dataset.csv",index = False)
#print(df)
df = df.merge(df_new,on='Department',how='left')
print(df)

import matplotlib as plt
