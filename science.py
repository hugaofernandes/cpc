# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from apyori import apriori
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


data = pd.read_csv('./Downloads/dataset.csv', sep=',') #abrindo o arquivo dataset.csv e separando por virgula
data.columns = ['time', 'sexo', 'area', 'fe', 'nota', 'redes', 'offline', 'Aquecimento Global', 'Teoria da Evolucao', 'Terraplana', 'Pouso na Lua', 'Vacinas', 'idade', 'opiniao']

#print (data.idade.head(30))
data = data.drop('time', 1) #excluir coluna time
data = data[data.idade > 0] #excluindo algumas respostas com valores de idade faltando

mlb = MultiLabelBinarizer() #algoritmo de categorização e tratamento
data.redes = data.redes.str.replace(', ', ',') #padronizando o formato das virgulas
data.redes = data.redes.str.split(',')
expandedLabelData = mlb.fit_transform(data.redes)
labelClasses = mlb.classes_
expandedLabels = pd.DataFrame(expandedLabelData, columns=labelClasses) #categorizando coluna redes
data = data.join(expandedLabels) #unindo as categorias de redes a tabela principal
data = data.drop('redes', 1) #excluindo coluna redes

entrevistados = data[data.opiniao == '0']
#entrevistados = entrevistados[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
anonimos = data[data.opiniao != '0']
#anominos = anominos[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]

homens = data[data.sexo == 'Masculino']
#homens = homens[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
mulheres = data[data.sexo == 'Feminino']
#mulheres = mulheres[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]

exatas = data[data.area == 'Ciências Exatas/Engenharia/Tecnologia']
#exatas = exatas[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
educacao = data[data.area == 'Educação']
#educacao = educacao[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
saude = data[data.area == 'Ciências Biológicas/Saúde']
#saude = saude[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
sociais = data[data.area == 'Ciências Sociais']
#sociais = sociais[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
negocios = data[data.area == 'Negócios']
#negocios = negocios[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
terra = data[data.area == 'Ciências da Terra/Agrícolas']
#terra = terra[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]

zero = data[data.offline == 'Nunca fiquei sem redes sociais']
umatres = data[data.offline == 'Entre 1 e 3 dias']
quatroadez = data[data.offline == 'Entre 4 e 10 dias']
dezoumais = data[data.offline == 'Mais de 10 dias']

facebook = np.sum(data.Facebook == 1)
instagram = np.sum(data.Instagram == 1)
youtube = np.sum(data.YouTube == 1)
whatsapp = np.sum(data.Whatsapp == 1)
twitter = np.sum(data.Twitter == 1)
telegram = np.sum(data.Telegram == 1)



'''
ANÁLISES QUE PRECISAM SER FEITAS:
relação de redes e teorias
relação de idade e teorias
relação de tempo_offline e nota como estudante
relação de nota e teorias
cluster para tentar encontrar grupos


#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) # grafico 4x4

# proporção por sexo
fig, ax1 = plt.subplots()
labels = 'Homens', 'Mulheres'
sizes = len(homens), len(mulheres)
explode = (0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax1.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend(labels, loc='upper right')
ax1.set_title('Distribuição por Gênero')
plt.show()

# proporção de entrevistados e anonimos
fig, ax2 = plt.subplots()
labels = 'Entrevistados', 'Anonimos'
sizes = len(entrevistados), len(anonimos)
explode = (0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax2.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.legend(labels, loc='center right')
ax2.set_title('Entrevistados x Anonimos')
plt.show()

# proporção por área
fig, ax3 = plt.subplots()
labels = 'Exatas', 'Educação', 'Biológicas', 'Sociais', 'Negócios', 'Agrícolas'
sizes = len(exatas), len(educacao), len(saude), len(sociais), len(negocios), len(terra)
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
pie = ax3.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax3.legend(pie[0],labels, bbox_to_anchor=(0,0), loc="lower left", bbox_transform=plt.gcf().transFigure)
ax3.set_title('Distribuição por Áreas')
plt.show()

# proporção de dias offline
fig, ax4 = plt.subplots()
labels = 'Zero dias', 'De 1 a 3 dias', 'De 4 a 10 dias', 'Mais de 10 dias'
sizes = len(zero), len(umatres), len(quatroadez), len(dezoumais)
explode = (0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
pie = ax4.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax4.legend(pie[0],labels, bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure)
ax4.set_title('Tempo Offline')
plt.show()

# proporção por redes sociais
fig, ax = plt.subplots()
labels = 'Facebook', 'Instagram', 'Twitter', 'Whatsapp', 'YouTube', 'Telegram'
sizes = facebook, instagram, twitter, whatsapp, youtube, telegram
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Uso de Redes Sociais')
plt.show()

# histograma das idades absoluta e relativa
fig, axs = plt.subplots(1, 2, tight_layout=True)
N, bins, patches = axs[0].hist(data.idade, bins=len(data.idade))
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
axs[1].hist(data.idade, bins=len(data.idade), density=True)
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
axs[0].set_title('Idades Absolutas')
axs[1].set_title('Idades Relativas')
plt.show()

# proporção de nota como estudante
fig, ax = plt.subplots()
#labels = data.nota.unique().tolist()
labels = [8, 1, 9, 10, 5, 7, 4, 6, 3] # ficou feio tive que inventar isso...
sizes = []
for i in labels:
    sizes.append(len(data[data.nota == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Nota como Estudante')
plt.show()

# proporção de nível de espiritualidade
fig, ax = plt.subplots()
labels = data.fe.unique().tolist()
sizes = []
for i in labels:
    sizes.append(len(data[data.fe == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Nível de Espiritualidade')
plt.show()

#teorias com margem de erro
#data = data[['Q3', 'Q4', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']]
teorias = data.iloc[:, 5:10]
erroUp = teorias + (1.96 * (teorias.std() / np.sqrt(len(teorias))))
erroDown = teorias - (1.96 * (teorias.std() / np.sqrt(len(teorias))))
#FONTES: https://pt.surveymonkey.com/mp/margin-of-error-calculator/
fig, ax = plt.subplots()
ax.plot(erroUp.mean(), '--or', label='Margens de erro de 5%')
ax.plot(teorias.mean(), '-o', label='Valor médio')
ax.plot(erroDown.mean(), '--or')
ax.set(ylabel='Escala Likert', title='Grença em Teorias')
plt.legend(loc='upper center')
plt.xticks(rotation=45, wrap=True)
ax.grid()
#fig.savefig("erro.png")
plt.show()


#diferença de entrevistados e anonimos
fig, ax = plt.subplots()
ax.plot(entrevistados.iloc[:, 5:10].mean(), '-o', label="Entrevistados")
ax.plot(anonimos.iloc[:, 5:10].mean(), '-o', label='Anônimos')
ax.set(ylabel='Escala Likert', title='Grença em Teorias entre Entrevistados e Anônimos')
ax.legend(loc='upper center')
plt.xticks(rotation=45, wrap=True)
ax.grid()
plt.show()

#diferença de homens e mulheres
fig, ax = plt.subplots()
ax.plot(homens.iloc[:, 5:10].mean(), '-o', label="Homens")
ax.plot(mulheres.iloc[:, 5:10].mean(), '-o', label='Mulheres')
ax.set(ylabel='Escala Likert', title='Grença em Teorias por Gênero')
ax.legend(loc='upper center')
plt.xticks(rotation=45, wrap=True)
ax.grid()
plt.show()


#diferença por área
fig, ax = plt.subplots()
ax.plot(exatas.iloc[:, 5:10].mean(), '-o', label="Exatas")
ax.plot(educacao.iloc[:, 5:10].mean(), '-o', label='Educação')
ax.plot(negocios.iloc[:, 5:10].mean(), '-o', label='Negócios')
ax.plot(terra.iloc[:, 5:10].mean(), '-o', label='Agrícolas')
ax.plot(saude.iloc[:, 5:10].mean(), '-o', label='Biológicas')
ax.plot(sociais.iloc[:, 5:10].mean(), '-o', label='Sociais')
ax.set(ylabel='Escala Likert', title='Grença em Teorias por Áreas')
ax.legend(loc='upper center')
plt.xticks(rotation=45, wrap=True)
ax.grid()
plt.show()


#diferença por idades
fig, ax = plt.subplots()
ax.plot(data[data.idade <= 20].iloc[:, 5:10].mean(), '-o', label="Até 20 anos")
ax.plot(data[(data.idade > 20) & (data.idade <= 23)].iloc[:, 5:10].mean(), '-o', label="De 21 a 23 anos")
ax.plot(data[(data.idade > 23) & (data.idade <= 26)].iloc[:, 5:10].mean(), '-o', label="De 24 a 26 anos")
ax.plot(data[(data.idade > 26) & (data.idade <= 30)].iloc[:, 5:10].mean(), '-o', label="De 27 a 30 anos")
ax.plot(data[(data.idade > 31) & (data.idade <= 35)].iloc[:, 5:10].mean(), '-o', label="De 31 a 35 anos")
ax.plot(data[(data.idade > 35) & (data.idade <= 40)].iloc[:, 5:10].mean(), '-o', label="De 36 a 40 anos")
ax.plot(data[data.idade > 40].iloc[:, 5:10].mean(), '-o', label="Mais de 40 anos")
ax.set(ylabel='Escala Likert', title='Grença em Teorias por Idade')
ax.legend(loc='upper center')
plt.xticks(rotation=45, wrap=True)
ax.grid()
plt.show()

# proporção de crença em aquecimento global
fig, ax = plt.subplots()
labels = data['Aquecimento Global'].unique().tolist()
sizes = []
for i in labels:
    sizes.append(len(data[data['Aquecimento Global'] == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Crença em Aquecimento Global')
plt.show()


# proporção de crença em Teoria da Evolução
fig, ax = plt.subplots()
labels = data['Teoria da Evolucao'].unique().tolist()
sizes = []
for i in labels:
    sizes.append(len(data[data['Teoria da Evolucao'] == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Crença em Teoria da Evolução')
plt.show()


# proporção de crença em Vacinas
fig, ax = plt.subplots()
labels = data['Vacinas'].unique().tolist()
sizes = []
for i in labels:
    sizes.append(len(data[data['Vacinas'] == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Crença em Vacinas')
plt.show()

# proporção de crença em Terraplana
fig, ax = plt.subplots()
labels = data['Terraplana'].unique().tolist()
sizes = []
for i in labels:
    sizes.append(len(data[data['Terraplana'] == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Crença em Terraplana')
plt.show()


# proporção de crença em pouso na lua
fig, ax = plt.subplots()
labels = data['Pouso na Lua'].unique().tolist()
sizes = []
for i in labels:
    sizes.append(len(data[data['Pouso na Lua'] == i]))
explode = [0.1]*len(labels)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.legend(labels, loc='lower right')
ax.set_title('Crença Pouso na Lua')
plt.show()
'''


#data = data.astype(str)
#association_rules = apriori(data.values.tolist(), min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
#association_results = list(association_rules)
#FONTE: https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

'''
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder() #initializing an object of class LabelEncoder
data.sexo = labelencoder.fit_transform(data.sexo)
data.offline = labelencoder.fit_transform(data.offline) #fitting and transforming the desired categorical column.
data.area = labelencoder.fit_transform(data.area)
data.idade = data.idade.astype(int)
data = data.drop('opiniao', 1) #excluir coluna opiniao
data = data.drop('Steam ', 1) #excluir coluna Steam *sim com espaço
data = data.drop('Email', 1) #excluir coluna Email
data = data[data.Facebook >= 0] #por alguma razão as 3 ultimas linhas nas redes estavam NaN
data.iloc[:, 11:] = data.iloc[:, 11:].astype(int)

#print (data.idade.head(10))
#print (data.iloc[:, 10:].head(1000))

from sklearn.cluster import KMeans
X = data.values

kmeans = KMeans(n_clusters=8)
kmeans.fit(X)
labels = kmeans.labels_
print (labels)
plt.subplot()
plt.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor='k')
plt.show()
'''
