import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
df=pd.read_csv('/content/dataset.csv')
df.head()
df.info()
df=df.dropna()
df.info()
anomaly_inputs=['Apple','Amazon']
model_IF=IsolationForest(contamination=0.1,random_state=20)
model_IF.fit(df[anomaly_inputs])
df['scores']=model_IF.decision_function(df[anomaly_inputs])
df['anomaly_inputs']=model_IF.predict(df[anomaly_inputs])
df.loc[:,['Apple','Amazon','scores','anomaly_inputs']]
anomaly_inputs = ['Apple', 'Amazon']
y_pred = model_IF.predict(df[anomaly_inputs])
anomaly_indices = y_pred == -1
df[anomaly_indices]
anomaly_indices = df[df['anomaly_inputs'] == -1]
plt.figure(figsize=(10,6))
plt.scatter(anomaly_indices['Apple'],anomaly_indices['Amazon'],color='red',label='anomaly')
plt.plot(df['Apple'],df['Amazon'],color='blue',label='normal')
plt.title('Anomaly Detection in Time series')
plt.xlabel('Apple')
plt.ylabel('Amazon')
plt.legend()
plt.show()
print("Number of Anomalies:")
print(anomaly_indices)
