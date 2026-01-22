import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"22ndJan26\dataset.csv")
df1 = pd.read_csv(r"22ndJan26\d.csv")

plt.bar(df["Years_Experience"],df["Salary"])
plt.xlabel("Experience")
plt.ylabel("Salary")


plt.bar(df1["Years_Experience"],df1["Department"])
plt.xlabel("DEPARTMENT_ID")
plt.ylabel("DEPARTMENT")




dept_counts = df["Department"].value_counts()

plt.bar(dept_counts.index, dept_counts.values)
plt.xlabel("Department")
plt.ylabel("Employee Count")
plt.title("Employees per Department")
plt.show()

#Employee_ID,Name,Department,Position,Salary,Age,Years_Experience
 
