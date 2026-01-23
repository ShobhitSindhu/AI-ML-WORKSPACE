#from sklearn.linear_model import LogisticRegression

#x = [[1000],[2000],[3000],[4000]]
#y = [0,1,0,1]

#model = LogisticRegression()
#model.fit(x, y)

#print(model.predict([[5000]]))

import numpy as np 
import pandas as pd
df = pd.read_csv(r"23rdJan26\Loan dataset_classification.csv")

print(df.isnull())

