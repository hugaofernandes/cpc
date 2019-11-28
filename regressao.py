

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#pd.set_option('display.max_rows', None)

data = pd.read_csv('data.csv', sep=',')
data.columns = ['time', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13']
data = data[data.Q12.isna() != True]

data['Whatsapp'] = pd.DataFrame(data.Q5.str.contains('Whatsapp'))
#data.Whatsapp[data.Whatsapp == True] = 1
#data.Whatsapp[data.Whatsapp != 1] = 0
data['Facebook'] = pd.DataFrame(data.Q5.str.contains('Facebook'))
#data.Facebook[data.Facebook == True] = 1
#data.Facebook[data.Facebook != 1] = 0
data['Instagram'] = pd.DataFrame(data.Q5.str.contains('Instagram'))
#data.Instagram[data.Instagram == True] = 1
#data.Instagram[data.Instagram != 1] = 0
data['Telegram'] = pd.DataFrame(data.Q5.str.contains('Telegram'))
#data.Telegram[data.Telegram == True] = 1
#data.Telegram[data.Telegram != 1] = 0
data['Twitter'] = pd.DataFrame(data.Q5.str.contains('Twitter'))
#data.Twitter[data.Twitter == True] = 1
#data.Twitter[data.Twitter != 1] = 0
data['YouTube'] = pd.DataFrame(data.Q5.str.contains('YouTube'))
#data.YouTube[data.YouTube == True] = 1
#data.YouTube[data.YouTube != 1] = 0

data.Q1[data.Q1 == 'Masculino'] = 1
data.Q1[data.Q1 == 'Feminino'] = 0

data.Q2[data.Q2 == 'Ciências Exatas/Engenharia/Tecnologia'] = 0
data.Q2[data.Q2 == 'Educação'] = 1
data.Q2[data.Q2 == 'Ciências Biológicas/Saúde'] = 2
data.Q2[data.Q2 == 'Ciências Sociais'] = 3
data.Q2[data.Q2 == 'Negócios'] = 4
data.Q2[data.Q2 == 'Ciências da Terra/Agrícolas'] = 5

data.Q6[data.Q6 == 'Nunca fiquei sem redes sociais'] = 0
data.Q6[data.Q6 == 'Entre 1 e 3 dias'] = 1
data.Q6[data.Q6 == 'Entre 4 e 10 dias'] = 2
data.Q6[data.Q6 == 'Mais de 10 dias'] = 3
data.Q6[data.Q6 == 'Não tenho redes sociais'] = 4


#print(data.head())
X = data[['Q1', 'Q2', 'Q3', 'Q4', 'Q6', 'Q12', 'Whatsapp', 'Facebook', 'Instagram', 'Telegram', 'Twitter', 'YouTube']]
y = data[['Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
reg = LinearRegression().fit(X_train, y_train)
coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(reg.coef_))], axis = 1)
print(coefficients)



X_train, X_test, y_train, y_test = train_test_split(X, data.Q7, test_size=None)
reg = LinearRegression().fit(X_train, y_train)

fig, ax = plt.subplots()
ax.plot(y_test, y_test, 'o', label="Q7_Original")
predict = reg.predict(X_test)
ax.plot(predict, predict, '.', label="Q7_Predição")
ax.set(xlabel='Amostras', ylabel='Escala Likert', title='Q7 \nAcuracia: ' + str(accuracy_score(y_test, np.rint(predict))*100) + '%')
ax.legend(loc='upper left')
ax.grid()
fig.savefig("regressionQ7.png")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, data.Q8, test_size=None)
reg = LinearRegression().fit(X_train, y_train)

fig, ax = plt.subplots()
ax.plot(y_test, y_test, 'o', label="Q8_Original")
predict = reg.predict(X_test)
ax.plot(predict, predict, '.', label="Q8_Predição")
ax.set(xlabel='Amostras', ylabel='Escala Likert', title='Q8 \nAcuracia: ' + str(accuracy_score(y_test, np.rint(predict))*100) + '%')
ax.legend(loc='upper left')
ax.grid()
fig.savefig("regressionQ8.png")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, data.Q9, test_size=None)
reg = LinearRegression().fit(X_train, y_train)

fig, ax = plt.subplots()
ax.plot(y_test, y_test, 'o', label="Q9_Original")
predict = reg.predict(X_test)
ax.plot(predict, predict, '.', label="Q9_Predição")
ax.set(xlabel='Amostras', ylabel='Escala Likert', title='Q9 \nAcuracia: ' + str(accuracy_score(y_test, np.rint(predict))*100) + '%')
ax.legend(loc='upper left')
ax.grid()
fig.savefig("regressionQ9.png")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, data.Q10, test_size=None)
reg = LinearRegression().fit(X_train, y_train)

fig, ax = plt.subplots()
ax.plot(y_test, y_test, 'o', label="Q10_Original")
predict = reg.predict(X_test)
ax.plot(predict, predict, '.', label="Q10_Predição")
ax.set(xlabel='Amostras', ylabel='Escala Likert', title='Q10 \nAcuracia: ' + str(accuracy_score(y_test, np.rint(predict))*100) + '%')
ax.legend(loc='upper left')
ax.grid()
fig.savefig("regressionQ10.png")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, data.Q11, test_size=None)
reg = LinearRegression().fit(X_train, y_train)

fig, ax = plt.subplots()
ax.plot(y_test, y_test, 'o', label="Q11_Original")
predict = reg.predict(X_test)
ax.plot(predict, predict, '.', label="Q11_Predição")
ax.set(xlabel='Amostras', ylabel='Escala Likert', title='Q11 \nAcuracia: ' + str(accuracy_score(y_test, np.rint(predict))*100) + '%')
ax.legend(loc='upper left')
ax.grid()
fig.savefig("regressionQ11.png")
plt.show()

#print(X_test)
#print(reg.score(X_test, y_test))
#print(r2_score(reg.predict(X_test), y_test))
#print(reg.get_params(deep=True))
#print(reg.predict(X_test))
#print(y_test.Q7)
#print(reg.predict(X_test)[0])
#print
