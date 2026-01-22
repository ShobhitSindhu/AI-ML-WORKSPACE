import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"22ndJan26\dataset.csv")

sns.barplot(data=df, x="Department", y="Salary")
plt.title("Average Salary by Department")
plt.show()

sns.scatterplot(
    data=df,
    x="Department",
    y="Salary",
    hue="Years_Experience"
)
plt.title("Salary vs Department by Experience")
plt.show()

