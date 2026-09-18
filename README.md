# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES


## DATASET: E-COMMERCE

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```
## DATALOAD
```
data=pd.read_csv('/content/ecommerce_sales_analytics_5000.csv')
```
```
N=1000
plt.rcParams['figure.figsize'] = [12, 6]
X=data['revenue'] # Using 'revenue' column for time series analysis
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```
```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
# Fitting the ARMA(1,1) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="990" height="522" alt="image" src="https://github.com/user-attachments/assets/deeeac7f-146b-47bc-872e-1700e4bf6cba" />



Partial Autocorrelation
<img width="997" height="532" alt="image" src="https://github.com/user-attachments/assets/016a5214-883f-4643-87ef-01b131583e33" />

Autocorrelation

<img width="987" height="532" alt="image" src="https://github.com/user-attachments/assets/60c7febf-3724-45fe-9efb-eb341bcb3448" />


SIMULATED ARMA(2,2) PROCESS:

<img width="987" height="540" alt="image" src="https://github.com/user-attachments/assets/052e284c-b9ef-4727-b3f4-d3be35a9794a" />

Partial Autocorrelation

<img width="990" height="522" alt="image" src="https://github.com/user-attachments/assets/208d8231-b02b-4511-9fcb-bfa7edbb287b" />

Autocorrelation

<img width="997" height="527" alt="image" src="https://github.com/user-attachments/assets/9fa5e104-fddd-4e70-b289-6ca4719c6ebf" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.

