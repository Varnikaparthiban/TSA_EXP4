# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

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


data = pd.read_csv('/content/weather_2013_to_2024.csv')
print(data.head())

X = data['temp']

### VISUALIZE ORIGINAL DATA


plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)

plt.title('Temperature Data')

plt.xlabel('Index')

plt.ylabel('Temperature')

plt.show()


### PLOT ACF AND PACF


plt.subplot(2, 1, 1)

plot_acf(X, lags=30, ax=plt.gca())

plt.title('Temperature Data ACF')

plt.subplot(2, 1, 2)

plot_pacf(X, lags=30, ax=plt.gca())

plt.title('Temperature Data PACF')

plt.tight_layout()

plt.show()

### FIT ARMA(1,1) MODEL

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print(arma11_model.summary())


### EXTRACT PARAMETERS

phi1 = arma11_model.params['ar.L1']

theta1 = arma11_model.params['ma.L1']

### SIMULATE ARMA(1,1)

N = 1000

ar1 = np.array([1, -phi1])

ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

### Plot simulated process

plt.plot(ARMA_1)

plt.title('Simulated ARMA(1,1) Process')

plt.xlim([0, 500])

plt.show()

### ACF AND PACF OF ARMA(1,1)

plot_acf(ARMA_1)

plt.title('ACF of ARMA(1,1)')

plt.show()

plot_pacf(ARMA_1)

plt.title('PACF of ARMA(1,1)')

plt.show()

### FIT ARMA(2,2) MODEL

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print(arma22_model.summary())

### EXTRACT PARAMETERS

phi1_22 = arma22_model.params['ar.L1']

phi2_22 = arma22_model.params['ar.L2']

theta1_22 = arma22_model.params['ma.L1']

theta2_22 = arma22_model.params['ma.L2']

### SIMULATE ARMA(2,2)


ar2 = np.array([1, -phi1_22, -phi2_22])

ma2 = np.array([1, theta1_22, theta2_22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

### Plot simulated process

plt.plot(ARMA_2)

plt.title('Simulated ARMA(2,2) Process')

plt.xlim([0, 500])

plt.show()

### ACF AND PACF OF ARMA(2,2)

plot_acf(ARMA_2)

plt.title('ACF of ARMA(2,2)')

plt.show()

plot_pacf(ARMA_2)

plt.title('PACF of ARMA(2,2)')

plt.show()

```
## OUTPUT:
### Original Data:
<img width="966" height="520" alt="image" src="https://github.com/user-attachments/assets/1f509b6d-1713-4260-89a2-d55454db04a2" />

### Autocorrelation:
<img width="1148" height="290" alt="image" src="https://github.com/user-attachments/assets/4214f070-b41f-4340-8b52-56f2cc85d1fc" />

### Partial Autocorrelation:
<img width="1173" height="279" alt="image" src="https://github.com/user-attachments/assets/2a89eff0-4046-447f-b148-bee2f28ba248" />

### SIMULATED ARMA(1,1) PROCESS:
<img width="947" height="484" alt="image" src="https://github.com/user-attachments/assets/950320a3-de96-4693-bb08-a7dda4cbfa54" />

### Autocorrelation:
<img width="982" height="506" alt="image" src="https://github.com/user-attachments/assets/b77e35ce-f155-47b9-b12e-083c69dc2673" />

### Partial Autocorrelation:
<img width="995" height="512" alt="image" src="https://github.com/user-attachments/assets/fdd8c9fb-9284-4143-ac11-abfdd5119e12" />

### SIMULATED ARMA(2,2) PROCESS: 
<img width="1018" height="495" alt="image" src="https://github.com/user-attachments/assets/dd35157c-0829-42da-b067-40d4349eafbb" />

### Autocorrelation:
<img width="985" height="497" alt="image" src="https://github.com/user-attachments/assets/dfbf2d2d-2fe5-438c-b456-6dea505ecd43" />

### Partial Autocorrelation:
<img width="1007" height="504" alt="image" src="https://github.com/user-attachments/assets/e69accd1-7b72-407a-ac99-0d7e46b34c53" />


## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
