
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', sep=',')
data.columns = ['time', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13']

data.Q6[data.Q6 == 'Nunca fiquei sem redes sociais'] = 5
data.Q6[data.Q6 == 'Entre 1 e 3 dias'] = 4
data.Q6[data.Q6 == 'Entre 4 e 10 dias'] = 3
data.Q6[data.Q6 == 'Mais de 10 dias'] = 2
data.Q6[data.Q6 == 'Não tenho redes sociais'] = 1



entrevistados = data[data.Q13 == '0']
entrevistados = entrevistados[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
anominos = data[data.Q13 != '0']
anominos = anominos[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]

fig, ax = plt.subplots()
ax.plot(entrevistados.mean(), '-o', label="Entrevistados")
ax.plot(anominos.mean(), '-o', label='Anônimos')
ax.set(xlabel='Questões', ylabel='Escala', title='Entrevistados x Anônimos')
ax.legend(loc='upper right')
ax.grid()
fig.savefig("analiseByGroups.png")
plt.show()

fig, ax = plt.subplots()
labels = 'entrevistados', 'anominos'
sizes = [len(entrevistados), len(anominos)]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set(title='Distribuição Entrevistados x Anônimos')
fig.savefig("ByGroups.png")
plt.show()



homens = data[data.Q1 == 'Masculino']
homens = homens[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
mulheres = data[data.Q1 == 'Feminino']
mulheres = mulheres[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]

fig, ax = plt.subplots()
ax.plot(homens.mean(), '-o', label="Homens")
ax.plot(mulheres.mean(), '-o', label='Mulheres')
ax.set(xlabel='Questões', ylabel='Escala', title='Masculino x Feminino')
ax.legend(loc='upper right')
ax.grid()
fig.savefig("analiseBySex.png")
plt.show()

fig, ax = plt.subplots()
labels = 'masculino', 'feminino'
sizes = [len(homens), len(mulheres)]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set(title='Distribuição por Sexo')
fig.savefig("BySex.png")
plt.show()



exatas = data[data.Q2 == 'Ciências Exatas/Engenharia/Tecnologia']
exatas = exatas[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
educacao = data[data.Q2 == 'Educação']
educacao = educacao[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
saude = data[data.Q2 == 'Ciências Biológicas/Saúde']
saude = saude[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
sociais = data[data.Q2 == 'Ciências Sociais']
sociais = sociais[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
negocios = data[data.Q2 == 'Negócios']
negocios = negocios[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
terra = data[data.Q2 == 'Ciências da Terra/Agrícolas']
terra = terra[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]

fig, ax = plt.subplots()
ax.plot(exatas.mean(), '--o', label="Exatas")
ax.plot(educacao.mean(), '--o', label="Educação")
ax.plot(negocios.mean(), '--o', label="Negócios")
ax.plot(terra.mean(), '--o', label="Ciências da Terra")
ax.plot(saude.mean(), '--o', label="Saúde")
ax.plot(sociais.mean(), '--o', label='Ciências Sociais')
ax.set(xlabel='Questões', ylabel='Escala', title='ByCurso')
ax.legend(loc='upper right')
ax.grid()
fig.savefig("analiseByCurso.png")
plt.show()

fig, ax = plt.subplots()
labels = 'Exatas', 'Educação', 'Biológicas/Saúde', 'Ciências Sociais', 'Negócios', 'Terra/Agrícolas'
sizes = [len(exatas), len(educacao), len(saude), len(sociais), len(negocios), len(terra)]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set(title='Distribuição por Formação')
fig.savefig("ByCurso.png")
plt.show()



data = data[data.Q12.isna() != True]
data.Q12 = data.Q12.astype(int)

fig, ax = plt.subplots()
groupby = pd.DataFrame(data.Q12).groupby('Q12').size().reset_index(name='quant')
plt.bar(groupby.Q12, groupby.quant)
ax.set(xlabel='Idades', ylabel='Quantidade', title='Distribuição por Idade')
fig.savefig("ByIdade.png")
plt.show()



data = data[['Q3', 'Q4', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
erroUp = data + (1.96 * (data.std() / np.sqrt(len(data))))
erroDown = data - (1.96 * (data.std() / np.sqrt(len(data))))

fig, ax = plt.subplots()
ax.plot(erroUp.mean(), '--or', label="Int. de Conf. Superior")
ax.plot(data.mean(), '-o', label="Média Geral")
ax.plot(erroDown.mean(), '--og', label="Int. de Conf. Inferior")
ax.set(xlabel='Questões', ylabel='Escala', title='Intervalo de Confiança (95%)')
ax.legend(loc='upper right')
ax.grid()
fig.savefig("erro.png")
plt.show()





