**IMPORT DATA**


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastai.tabular.all import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.dates as mdates
import glob

from google.colab import drive
drive.mount('/content/drive')
#defino ruta
ruta_store = '/content/drive/My Drive/rossmann-store-sales/store.csv'
ruta_train = '/content/drive/My Drive/rossmann-store-sales/train.csv'
ruta_test = '/content/drive/My Drive/rossmann-store-sales/test.csv'


#importo archivos
store = pd.read_csv(ruta_store)
train = pd.read_csv(ruta_train, parse_dates=[2])
test = pd.read_csv(ruta_test, parse_dates=[3])

# Configurar pandas para mostrar todas las columnas sin truncar y evitar que se envuelvan las filas.
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#importo Data Externa
ruta_weather = '/content/drive/My Drive/rossmann-store-sales/Weather/*.csv'
ruta_state = '/content/drive/My Drive/rossmann-store-sales/store_states.csv'
ruta_google_trend = '/content/drive/My Drive/rossmann-store-sales/Google Trend/*.csv'

#Clima
tiempo_de_los_estados_alemanes = glob.glob(ruta_weather)

#Vista previa de archivo meteorológico

pd.read_csv(tiempo_de_los_estados_alemanes[0],delimiter=";").head()

#Estado
store_state = pd.read_csv(ruta_state)
store_state.head()

"""**ANALISIS EXPLORATORIO**

Analisis Exploratorio - Store
"""

# busco entender los datos, qué info hay oculta en la data
# analizo cada dataset
print(store)
#observo 1115 rows con datos adicionales de las stores
print(store.shape)
store.info() # Con data.info() puedo ver las variables categóricas
store.isnull().sum() # hay 354 missings en CompetitionOpenSinceMonth, CompetitionOpenSinceYear, 3 en competition distance,
#y 544 en Promo2SinceWeek, Promo2SinceYear, PromoInterval

# Filtrar y mostrar las filas con NaN en la variable CompetitionDistance
rows_with_na = store[store['CompetitionDistance'].isna()]
print(rows_with_na) #tienen NAN en CompetitionDistance,CompetitionOpenSinceMonth,CompetitionOpenSinceYear,Promo2SinceWeek,Promo2SinceYear . La eliminaremos en featuring engeneering

# analizo las cuatro variables categoricas
columnas_cat = ['StoreType', 'Assortment', 'Promo2', 'PromoInterval']
for col in columnas_cat:
  print(f'{col} contiene: {store[col].nunique()} categorias diferentes')
#StoreType contiene: 4 categorias diferentes (a,b,c,d)
#Assortment contiene: 3 categorias diferentes
#Promo2 contiene: 2 categorias diferentes (es binaria 0,1)
#PromoInterval contiene: 3 categorias diferentes

# StoreType: analizo la cantidad de stores dentro de cada tipo
store_counts = store['StoreType'].value_counts()
#a    602
#d    348
#c    148
#b     17
print(store_counts)
graficar(store_counts.index, store_counts.values, 'bar', 'Sales by stores type')

#Assortment: analizo la cantidad dentro de cada categoria
store.Assortment.value_counts()
#a    593
#c    513
#b      9

# PromoInterval: cuento unique values
store.PromoInterval.value_counts()
#Jan,Apr,Jul,Oct     335
#Feb,May,Aug,Nov     130
#Mar,Jun,Sept,Dec    106
store.CompetitionOpenSinceYear.value_counts()
conteos = store.CompetitionOpenSinceYear.value_counts()
plt.bar(conteos.index, conteos.values) #se ve que hay un store que abrio en el 1900
graficar(conteos.index, conteos.values, 'bar', 'Año de apertura de la competencia')

store.describe() # se observa una gran dispersion en los datos de CompetitionDistance (std alta), los min y max se observan coherentes

"""Analisis Exploratorio - Train"""

print(train.shape) # (1017209, 9) filas, cols
print(train)
train.info() # hay dos objects (string o cat), state holiday y day of week; el resto ints
train.isnull().sum() #no hay datos faltantes!

# analizo las unicas variables que no son de tipo int64
columnas_cat = ['Date', 'StateHoliday']
for col in columnas_cat:
  print(f'Columna {col}: {train[col].nunique()} categorias diferentes') # 942 fechas diferentes, 5 categorias diferentes para StateHoliday,cuando deberian ser 4. Detectamos que es debido a un espacio en blanco
print("Valores:\n", train['StateHoliday'].value_counts())
#Columna Date: 942 categorias diferentes
#Columna StateHoliday: 5 categorias diferentes

print(train['StateHoliday'].unique()) #hay una mezcla de diferentes representaciones del valor cero, incluyendo '0' (como string) y 0 (como un entero).

# selecciono solo las columnas de tipo 'object'
object_cols = train.select_dtypes(include=['object']).columns
# aplico strip() a todas las columnas de tipo 'object'
train[object_cols] = train[object_cols].applymap(lambda x: x.strip() if isinstance(x, str) else x)
print("Valores:\n", train['StateHoliday'].value_counts()) #Chequeamos que se soluciono el error en StateHoliday

# El gráfico de recuento para la variable StateHoliday
sns.countplot(x='StateHoliday', data=train)
plt.xlabel('StateHoliday')
plt.ylabel('Recuento')
plt.title('Recuento de eventos por StateHoliday')
plt.show()

# El gráfico de recuento para la variable Date
sns.countplot(x='Date', data=train)
plt.xticks(rotation=50, fontsize=7)
# Formatear las fechas en el eje x
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xlabel('Fecha')
plt.ylabel('Recuento')
plt.title('Recuento de eventos por fecha')
plt.show()
# Vemos que hay muchos StateHoliday = 0. pocos a,b,c con mas a (public holiday).
# Muchisimas fechas con un drop off en una seccion, puedo combinar por mes o algo para ver obs  mejor (a futuro creo que esta bueno reemplazarlo por cero por lo leido en la documentacion de quien subio el dataset en la competencia)

train.describe()

# Correlacion entre Sales vs Customers
plt.figure(figsize=(10,6))
sns.scatterplot(x='Customers', y='Sales', data=train)
lin_fit = np.polyfit(train['Customers'], train['Sales'], 1)
lin_func = np.poly1d(lin_fit)(train['Customers'])
plt.plot(train['Customers'], lin_func, "r--", lw=1)
plt.title(f"Correlación entre Clientes y Ventas: {round(train['Customers'].corr(train['Sales'])*100, 2)}%")
# Cambiar etiquetas de los ejes
plt.xlabel('Clientes')
plt.ylabel('Ventas')
plt.show()
# observo la relacion linear entre customers y sales

train.groupby('Open')['Sales'].sum() #confirma que no hay ventas cuando las tiendas estan cerradas
train.groupby(['DayOfWeek', 'Promo']).mean()

# Generamos un histograma para entender la distribucion de sales
nbins = 75
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(x='Sales', data=train, ax=ax, bins=nbins, kde=True)
ax.set_xlabel('Ventas')
ax.set_ylabel('Conteo')
ax.set_title('Distribución de ventas')
plt.show()
# Sales 175k con 0, despues dist con mayoria de obs abajo de 10k gasto

# observo como varian los valores de sales en las distintas stores
graficar('Store', 'Sales', 'scatter', 'Sales variation in different stores')

#observo columnas numericas
cols_num = ['DayOfWeek','SchoolHoliday','Open']
for col in cols_num:
    graficar(col, 'Sales', 'bar', f'Ventas y {col}')

  # Day of week, los domingos no se vende en la mayoria de los stores
  # School Holiday, hay ventas igual
  # Open: efectivamente no hay error, no hay ventas cuando cerrado

#observo tiendas abiertas en funcion de los dias de la semana
sns.countplot(x = 'DayOfWeek', hue = 'Open', data = train)
plt.title('Conteo de tiendas abiertas por día')

#Graficamos boxplots para ver outliers
cols_num = ['Sales', 'Customers', 'Open', 'SchoolHoliday']

for col in cols_num:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=col, data=train)
    ax.set_title(col)
    plt.show()

# Comentarios:
    # Hay un customer mas alla de los 7000
    # Sales con muchos outliers pero obs con +40000 muy outlier
    # Open y SchoolHoliday binarias

"""Analisis Exploratorio - Test"""

print(test.shape) #(41088, 8)
print(test)
test.info() #la variable open binaria esta en float y convierto a integer
test.isnull().sum() #estos nulos fueron error de quien cargo el dataset, se puede reemplazar por cero o eliminar el registro (son 11 valores, no es significativo eliminarlos con la base de datos que poseemos, pero decidimos reemplazar por 0)

test['Open'] = test['Open'].astype(int)
columnas_cat = ['Date', 'StateHoliday']
for col in columnas_cat:
  print(f'{col}: {test[col].nunique()} categorias diferentes')
#Date: 48 categorias diferentes
#StateHoliday: 2 categorias diferentes
test.describe() #es racional lo descripto en lo devuelto en esta linea de codigo

"""**FEATURING ENGENIEERING**

Store + Train

"""

#droppeo costumers ya que la correlacion con sales es alta y
train = train.drop(columns=['Customers'])

#corrijo este formato que habia descubierto en el EDA en Train y Test
train['StateHoliday'] = train['StateHoliday'].replace(0, '0') # convierto the '0' string values a integers
train.StateHoliday.value_counts() #se observa corregido
test['Open'] = test['Open'].fillna(0) #error de carga de datos, el host de la competencia recomendaba reemplazar por cero
print(test.isnull().sum().sum()) #se verifica y quedo bien el reemplazo

# transformo Store para evitar que tenga datos faltantes y variables categoricas

# Promo2SinceWeek, Promo2SinceYear, PromoInterval: Inputamos "0" a las variables numéricas y "No promo" a PromoInterval
store['Promo2SinceWeek'].fillna(0, inplace=True)
store['Promo2SinceYear'].fillna(0, inplace=True)
store['PromoInterval'].fillna('No Promo', inplace=True)

#Calulamos la media para CompetitionOpenSinceMonth y CompetitionOpenSinceYear ya que no queremos eliminarlas, y no tiene sentido reemplazar con 0
imputer = SimpleImputer(strategy='mean')
cols_to_impute = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
imputer.fit(store[cols_to_impute])
store.loc[:, cols_to_impute] = imputer.transform(store[cols_to_impute])
print(store.isnull().sum().sum()) # verifico que no haya nulos

#store[cols_to_impute] = imputer.transform(store[cols_to_impute])

#transformo valores float a integer en train
store['CompetitionDistance'] = store['CompetitionDistance'].astype(int)
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].astype(int)
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].astype(int)
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].astype(int)
store['Promo2SinceYear'] = store['Promo2SinceYear'].astype(int)
test['Open'] = test['Open'].astype(int)

# aplico One-Hot encoding a las variables categorigas.
# Encoding Assorted - Al ser variables categoricas ordinales a = basic, b = extra, c = extended, las trasnformamos en 1,2,3.
store['Assortment']=store['Assortment'].map({'a':1,'b':2,'c':3})
print(store['Assortment'].value_counts())

#Encoding Promo Interval
store['PromoInterval']=store['PromoInterval'].map({'No Promo':0,'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3})
print(store['PromoInterval'].value_counts())

#Encoding StoreType
store['StoreType']=store['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
print(store['StoreType'].value_counts())

# Hacer state holiday booleana en train y test
train['StateHoliday'] = train['StateHoliday'].map({'0': 0, 'a': 1, 'b': 1, 'c': 1})
print(train['StateHoliday'].value_counts())
test['StateHoliday'] = test['StateHoliday'].map({'0': 0, 'a': 1, 'b': 1, 'c': 1})
print(test['StateHoliday'].value_counts())

#verifico que quedaron todas las variables corregidas y en correcto formato para poder hacer el merge
store.info()
train.info()
test.info()
store.isnull().sum()
train.isnull().sum()
test.isnull().sum()
print(store)
print(train)
print(test)

"""Data Externa"""

# Hago trasnformaciones a Weather para poder usarla

lista_eventos = ['', 'Fog-Rain', 'Fog-Snow', 'Fog-Thunderstorm',
              'Rain-Snow-Hail-Thunderstorm', 'Rain-Snow', 'Rain-Snow-Hail',
              'Fog-Rain-Hail', 'Fog', 'Fog-Rain-Hail-Thunderstorm', 'Fog-Snow-Hail',
              'Rain-Hail', 'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow', 'Rain-Thunderstorm',
              'Fog-Rain-Snow-Hail', 'Rain', 'Thunderstorm', 'Snow-Hail',
              'Rain-Snow-Thunderstorm', 'Snow', 'Fog-Rain-Thunderstorm']
lista_eventos_map = dict(zip(lista_eventos , range(len(lista_eventos))))
#Confirmar el mapping
[(k,v) for k,v in lista_eventos_map.items()][:3]

def nombres_estados_a_abreviaciones(nombre_estado):
    d = {}
    d['BadenWürttemberg'] = 'BW'
    d['Bayern'] = 'BY'
    d['Berlin'] = 'BE'
    d['Brandenburg'] = 'BB'  # no existe en store_state
    d['Bremen'] = 'HB'  # uso Niedersachsen en lugar de Bremen
    d['Hamburg'] = 'HH'
    d['Hessen'] = 'HE'
    d['MecklenburgVorpommern'] = 'MV'  # no existe en store_state
    d['Niedersachsen'] = 'HB,NI'  # uso Niedersachsen en lugar de Bremen
    d['NordrheinWestfalen'] = 'NW'
    d['RheinlandPfalz'] = 'RP'
    d['Saarland'] = 'SL'
    d['Sachsen'] = 'SN'
    d['SachsenAnhalt'] = 'ST'
    d['SchleswigHolstein'] = 'SH'
    d['Thüringen'] = 'TH'

    return d[nombre_estado]

lista_clima = []
for estado_aleman in tiempo_de_los_estados_alemanes:
    nombre_estado = os.path.splitext(os.path.basename(estado_aleman))[0]
    codigo_estado = nombres_estados_a_abreviaciones(nombre_estado)
    clima = pd.read_csv(estado_aleman, delimiter=";", parse_dates=['Date'])
    clima['State'] = codigo_estado

    for temp in ['Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC']:
        clima[temp] = (clima[temp] - 10) / 30

    for humi in ['Max_Humidity','Mean_Humidity', 'Min_Humidity']:
        clima[humi] = (clima[humi] - 50) / 50

    clima['Max_Wind_SpeedKm_h'] = clima['Max_Wind_SpeedKm_h'] / 50
    clima['Mean_Wind_SpeedKm_h'] = clima['Mean_Wind_SpeedKm_h'] / 30
    clima['CloudCover'].fillna(0,inplace=True)
    clima['Events'] = clima['Events'].map(lista_eventos_map)
    #El evento climático en blanco es el índice 0
    clima['Events'].fillna(0,inplace=True)
    clima = clima[['Date','State','Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC','Max_Humidity','Mean_Humidity', 'Min_Humidity',\
                      'Max_Wind_SpeedKm_h','Mean_Wind_SpeedKm_h','CloudCover','Events']]
    lista_clima.append(clima)

clima_total = pd.concat(lista_clima, ignore_index=True)

print(clima_total.isnull().sum().sum())
clima_total.head()

"""Data Set Final: train_merged, test_merged

"""

### combino Train y Store transformado
print(store['Store'].duplicated().sum())
store = store.drop_duplicates(subset='Store')
train['Store'] = train['Store'].astype(int)
test['Store'] = test['Store'].astype(int)

train_merged = train.merge(store, how='left', on='Store', validate='many_to_one')
print(train_merged.shape)
train_merged.to_csv('train_merged.csv', index=False)  # Guardar en un archivo CSV

# combino test y store
test_merged = test.merge(store, how='left', on='Store')
print(test_merged.shape)
test_merged.to_csv('test_merged.csv', index=False)  # Guardar en un archivo CSV

# Leer los datos combinados desde los archivos CSV
train_merged = pd.read_csv('train_merged.csv', low_memory=False)
test_merged = pd.read_csv('test_merged.csv', low_memory=False)

#Se chequea que quedo todo correcto post mergeada
train_merged.head()
test_merged.head()
train_merged.info()
test_merged.info()
train_merged.isnull().sum()
test_merged.isnull().sum()

### creo nuevas variables y transformo

#Transformaciones para Date
# convierto Date a datetime type
train_merged['Date'] = pd.to_datetime(train_merged['Date'])
test_merged['Date'] = pd.to_datetime(test_merged['Date'])

# Desagregamos date y extraemos info en nuevas variables
train_merged['Year'] = train_merged['Date'].dt.year
train_merged['Month'] = train_merged['Date'].dt.month
train_merged['Day'] = train_merged['Date'].dt.day
train_merged['Weekday'] = train_merged['Date'].dt.weekday
train_merged['Quarter'] = train_merged['Date'].dt.quarter

test_merged['Year'] = test_merged['Date'].dt.year
test_merged['Month'] = test_merged['Date'].dt.month
test_merged['Day'] = test_merged['Date'].dt.day
test_merged['Weekday'] = test_merged['Date'].dt.weekday
test_merged['Quarter'] = test_merged['Date'].dt.quarter

test_merged.info()
train_merged.info()

# Vemos los dfs modificados
print(train_merged.head())
print(test_merged.head())

# verifico las columnas en los dfs
print(train_merged.columns)  # Display all columns in train_merged df
print(test_merged.columns)

### combino con Data Externa

#State
train_merged = train_merged.merge(store_state,how='left',on='Store')
print(train_merged.shape)
print("train missing value ",train_merged.isnull().sum().sum())
train_merged.head()

test_merged = test_merged.merge(store_state,how='left',on='Store')
print(test_merged.shape)
print("test missing value ",test_merged.isnull().sum().sum())
test_merged.head()

# Weather
train_merged = train_merged.merge(clima_total,how='left',left_on=['State','Date'],right_on=['State','Date'])
print(train_merged.shape)
print("train missing value ",train_merged.isnull().sum().sum())
train_merged.head()

test_merged = test_merged.merge(clima_total,how='left',left_on=['State','Date'],right_on=['State','Date'])
print(test_merged.shape)
print("test missing value ",test_merged.isnull().sum().sum())
test_merged.head()

# Hacemos One - Hot Encoding con State
train_merged = pd.get_dummies(train_merged, columns=['State'])
test_merged = pd.get_dummies(test_merged, columns=['State'])

# Muestra las primeras filas del conjunto de datos codificado
print(train_merged.head())
print(test_merged.head())

"""Matriz de correlacion variables

"""

# Generamos una matriz de correlación usando todas las features
var_correlacion = train_merged.corr().abs()

# Graficamos un heat map
fig, axes = plt.subplots(figsize=(12, 12))
sns.heatmap(var_correlacion, annot = True, fmt='.2f', annot_kws={'size': 10},  vmax=.8, square=True, cmap='Blues');

"""#**MODELOS**

- Cosideramos una serie de modelos, entre ellos Xgboost, Random Forest, LightGBM.
- Para la creacion del conjunto de validation utilizo en primer lugar un reordenamiento cronologico cortando en 80/20 el set de train.
- Luego defino y utilizo TimeSeriesSplit.
- En funcion de estas tecnicas de CV optimizamos los hiperparametros de cada modelo.

Hold-Out Set
"""

# Tomamos un hold out set (80/20) para hcer cross-validation
# Ordenamos el conjunto de datos por año y mes en orden ascendente para considerar la temporalidad de los datos.
sorted_data = train_merged.sort_values(['Year', 'Month'])

# Calcular el índice de corte
cut_index = int(np.floor(len(sorted_data) * 0.8))

# Dividir el conjunto de datos en entrenamiento y validación
train_data = sorted_data.iloc[:cut_index]
valid_data = sorted_data.iloc[cut_index:]

# Obtener los índices correspondientes a los conjuntos de entrenamiento y validación
train_idx = train_data.index.values
valid_idx = valid_data.index.values

splits = (list(train_idx), list(valid_idx))

# verifico la cantidad de filas de los data sets
print(train_merged.shape)
print(train_data.shape)
print(valid_data.shape)

"""TimeSeriesSplit"""

# Probamos TimeSeriesFolds para generar folds de train y validations sets que consideren la temporalidad del data frame.
# analizo que los data sets tscv generados por TimeSeriesSplit sean correctos
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

x = train_merged.drop(columns=['Sales', 'Date']).values
y = np.log(train_merged['Sales'] + 1).values

# Inicializamos listas para guardar los resultados
x_train_list = []
x_val_list = []
y_train_list = []
y_val_list = []

train_index_list = []
val_index_list = []

for train_index, val_index in tscv.split(x):
    train_index_list.append(train_index)
    val_index_list.append(val_index)

train_index_list
val_index_list

"""**XG BOOST**

XG Boost + Hold Out Set
"""

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.utils import parallel_backend

x_train = train_data.drop(columns=['Sales','Date'])
y_train = log(train_data['Sales']+1)
x_test = valid_data.drop(columns=['Sales', 'Date'])
y_test = log(valid_data['Sales']+1)

# uso random search para iterar entre los parámetros de XGBoost
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]
}

# RMSPE
def rmspe(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    percentage_error = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return rmspe

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# entreno el modelo
xgb_reg = xgb.XGBRegressor(seed=42)
random_search = RandomizedSearchCV(
    xgb_reg,
    param_distributions=param_grid,
    n_iter=5,  # Numero de iteraciones
    scoring=rmspe_scorer,
    n_jobs=-1,  # Corre jobs en paralelo
    random_state=42,
    cv=5,
    verbose=3  # Verbosity
)
random_search.fit(x_train, y_train)

# obtengo los parámetros que mejor performan
print("Mejores Parámetros XGBoost: ", random_search.best_params_)

# predigo en test
y_pred = random_search.predict(x_test)

# calculo RMSPE
print("RMSPE = ", rmspe(y_test, y_pred))

"""XG Boost + TimeSeriesSplit"""

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
import lightgbm as lgb

train_merged = train_merged.sort_values(['Year', 'Month'])
x = train_merged.drop(columns=['Sales', 'Date']).values
y = np.log(train_merged['Sales'] + 1).values

def rmspe(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    percentage_error = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return rmspe

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# uso TimeSeriesSplit para crear 5 folds
tscv = TimeSeriesSplit(n_splits=5)

# elijo parámetros posibles
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15, 20, 25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9],
    'num_leaves': [31, 63, 127, 255],

}

rmspe_values = []
best_params = {}
best_rmspe = 1.0

for train_index, val_index in tscv.split(x):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    lgb_reg = lgb.LGBMRegressor(seed=42)
    random_search = RandomizedSearchCV(
        lgb_reg,
        param_distributions=param_grid,
        n_iter=5,
        scoring=rmspe_scorer,
        n_jobs=-1,
        random_state=42,
        cv=3, #para no ovverfittear
        verbose=3
    )

    random_search.fit(x_train, y_train)

    y_pred = random_search.predict(x_val)

    rmspe_val = rmspe(y_val, y_pred)
    rmspe_values.append(rmspe_val)

    if rmspe_val < best_rmspe:
        best_params = random_search.best_params_
        best_rmspe = rmspe_val

avg_rmspe = np.mean(rmspe_values)
print("Average RMSPE =", avg_rmspe)
print("Best parameters:", best_params)

# Feature importance cuando corro con time series
lgb_reg = lgb.LGBMRegressor(**best_params)
lgb_reg.fit(x, y)

# obtengo las variables + importantes
importances = lgb_reg.feature_importances_
importances_df = pd.DataFrame({'feature': train_merged.drop(columns=['Sales', 'Date']).columns, 'importance': importances})
importances_df = importances_df.sort_values('importance', ascending=False)
print("Important variables:", importances_df)

# Graficamos
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.barh(importances_df['feature'], importances_df['importance'], align='center')
plt.gca().invert_yaxis()
plt.show()

"""**RANDOM FOREST**

Random Forest + TimeSeriesSplit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer

train_merged = train_merged.sort_values(['Year', 'Month'])
x = train_merged.drop(columns=['Sales', 'Date']).values
y = np.log(train_merged['Sales'] + 1).values

def rmspe(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    percentage_error = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return rmspe

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# Use TimeSeriesSplit to create 5 folds
tscv = TimeSeriesSplit(n_splits=5)

# Define the parameter grid
param_grid = {
    'n_estimators': [250, 300, 350],
    'max_depth': [5, 10, 15],
    'min_samples_split': [20, 30, 40],
    'min_samples_leaf': [5, 10, 15],
    'max_features': [5, 10, 15]
}

rf_reg = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
        rf_reg,
        param_distributions=param_grid,
        n_iter=5,
        scoring=rmspe_scorer,
        n_jobs=-1,
        cv=tscv,
        verbose=3,
        random_state=42
)

random_search.fit(x,y)

# Best Params
print("Best Parameters: ", random_search.best_params_)

# RMSPE score
print("Best Score (RMSPE): ", -random_search.best_score_)

# Feature importance
best_params = random_search.best_params_
rf_reg = RandomForestRegressor(**best_params, random_state=42)
rf_reg.fit(x,y)

importances = rf_reg.feature_importances_
importances_df = pd.DataFrame({'feature': train_merged.drop(columns=['Sales', 'Date']).columns, 'importance': importances})
importances_df = importances_df.sort_values('importance', ascending=False)
print("Important variables:", importances_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.barh(importances_df['feature'], importances_df['importance'], align='center')
plt.gca().invert_yaxis()
plt.show()

"""Random Forest + TimeSeriesSplit + Predicciones OOB"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

train_merged = train_merged[train_merged['Open'] != 0]
test_merged = test_merged[test_merged['Open'] != 0]

# Reordering train set
sorted_data = train_merged.sort_values(['Year', 'Month'])

x = sorted_data.drop(columns=['Sales', 'Date'])
y = np.log(sorted_data['Sales'] + 1)

# RMSPE scorer
def rmspe(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    percentage_error = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return rmspe

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# Parameter grid
param_grid = {
    'n_estimators': [250, 300, 350],
    'max_depth': [5, 10, 15],
    'min_samples_split': [25, 30, 50],
    'min_samples_leaf': [5, 10, 15],
    'max_features': [5, 10, 15, 20] # sume un 20 en max features
}

# Random Forest model
rf_reg = RandomForestRegressor(oob_score=True, random_state=42)

# TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Randomized search
random_search = RandomizedSearchCV(rf_reg,
                                   param_distributions=param_grid,
                                   n_iter=5,
                                   scoring=rmspe_scorer,
                                   n_jobs=-1,
                                   cv=tscv,
                                   verbose=3,
                                   random_state=42)

random_search.fit(x, y)

# Mejores Parametros
print("Best RF parameters: ", random_search.best_params_)

# OOB Score
print(f"Out-of-bag score: {random_search.best_score_:.3f}")

"""Random Forest + Feature Importance + Modelo Final"""

#entreno un modelo con las variables más importantes y un número de iteraciones grandes.
#uso como data set = train + validation

from sklearn.ensemble import RandomForestRegressor
x = train_merged.drop(columns=['Sales'])
y = np.log(train_merged['Sales'] + 1)

#Nos quedmaos únicamente con las features mas impoprtantes
features_to_keep = ["Open", "DayOfWeek", "Weekday", "StateHoliday", "Promo",
            "CompetitionDistance", "Store" , "Day", "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "Month", "StoreType", "Promo2SinceYear",
           "Assortment", "Promo2SinceWeek", "State_NW", "PromoInterval" ]

x_important = x[features_to_keep]


rf_reg_important = RandomForestRegressor(n_estimators=300,
                                      min_samples_split=5,
                                      min_samples_leaf=4,
                                      max_features=24,
                                      max_depth=25,
                                      random_state=42,
                                      verbose=3)
rf_reg_important.fit(x_important, y)

# predigo ocn el train set
x_test = test_merged[features_to_keep]
y_pred = rf_reg_important.predict(x_test)

# Aplicar función exponencial inversa a las predicciones
y_pred_real = np.exp(y_pred) - 1

#generamos el sample submission
sample_submission = pd.DataFrame({'Id': test['Id'], 'Sales': y_pred_real})

# Export the DataFrame to a .csv file
sample_submission.to_csv('sample_submission.csv', index=False)

"""**LIGHTG BM**

Light BM + TimeSeriesSplit
"""

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
import lightgbm as lgb

train_marged = train_merged.sort_values(['Year', 'Month'])

def rmspe(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    percentage_error = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return rmspe

# Create rmspe scorer
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# Utilize the TimeSeriesSplit module from sklearn
tscv = TimeSeriesSplit(n_splits=5)

# Prepare the data
x = train_merged.drop(columns=['Sales', 'Date']).values
y = np.log(train_merged['Sales'] + 1).values

# Perform cross-validation and parameter tuning
param_grid = {
    'n_estimators': [50, 100, 150, 300],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [10, 20, 25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9],
    'num_leaves': [31, 63, 127, 255],

}

rmspe_values = []
best_params = {}
best_rmspe = 1.0

for train_index, val_index in tscv.split(x):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    lgb_reg = lgb.LGBMRegressor(seed=42)
    random_search = RandomizedSearchCV(
        lgb_reg,
        param_distributions=param_grid,
        n_iter=5,
        scoring=rmspe_scorer,
        n_jobs=-1,
        random_state=42,
        cv=3, #para no overfittear
        verbose=3
    )

    random_search.fit(x_train, y_train)

    y_pred = random_search.predict(x_val)

    rmspe_val = rmspe(y_val, y_pred)
    rmspe_values.append(rmspe_val)

    if rmspe_val < best_rmspe:
        best_params = random_search.best_params_
        best_rmspe = rmspe_val

avg_rmspe = np.mean(rmspe_values)
print("Average RMSPE =", avg_rmspe)
print("Best parameters:", best_params)

# Feature importance cuando corro con time series
lgb_reg = lgb.LGBMRegressor(**best_params)
lgb_reg.fit(x, y)

# Obtener las variables mas importantes
importances = lgb_reg.feature_importances_
importances_df = pd.DataFrame({'feature': train_merged.drop(columns=['Sales', 'Date']).columns, 'importance': importances})
importances_df = importances_df.sort_values('importance', ascending=False)
print("Important variables:", importances_df)

# Graficar
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.barh(importances_df['feature'], importances_df['importance'], align='center')
plt.gca().invert_yaxis()
plt.show()

"""LightBM + Hold Out Set"""

#Lightgbm
from lightgbm import LGBMRegressor

#defino rmspe
def rmspe(y_true, y_pred):
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    percentage_error = np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)
    rmspe = np.sqrt(np.mean(np.square(percentage_error)))
    return rmspe

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# defino las limitaciones de los parametros
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [15,20,25],
    'num_leaves': [20, 30, 40],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

#entreno al modelo
lgbm_reg = LGBMRegressor(seed=42)
random_search = RandomizedSearchCV(lgbm_reg,
                                   param_distributions=param_grid,
                                   n_iter=5,
                                   scoring=rmspe_scorer,
                                   n_jobs=-1,
                                   cv=5,
                                   verbose=3,
                                   random_state=42)
random_search.fit(x_train, y_train)

# obtengo los mejores parámetros
print("Mejores Parametros Lgbm: ",random_search.best_params_)

# predigo en test
y_pred = random_search.predict(x_test)

# calculo RMSPE
print("RMSPE = ", rmspe(y_test, y_pred))

"""Light BM + Feature Importance + Modelo Final


"""

#entreno un modelo con las variables más importantes y un número de iteraciones grandes.
#uso como data set = train + validation

x = train_merged[["Store" , "DayOfWeek","Open", "StoreType" , "Day", "Promo", "Month", "Assortment", "State_NW",
           "PromoInterval", "State_SH", "Max_TemperatureC", "Weekday"]]
y = np.log(train_merged['Sales'] + 1)

lgb_reg_important = lgb.LGBMRegressor(n_estimators=5000,
                                      reg_lambda=1.1,
                                      reg_alpha=1.3,
                                      num_leaves=32,
                                      max_depth=10,
                                      colsample_bytree=0.7,
                                      subsample=0.7,
                                      random_state=42,
                                      verbose=3)
lgb_reg_important.fit(x, y)
# predigo con el train set

x_test = test_merged[["Store" , "DayOfWeek","Open", "StoreType" , "Day", "Promo", "Month", "Assortment", "State_NW",
           "PromoInterval", "State_SH", "Max_TemperatureC", "Weekday"]]
y_pred = lgb_reg_important.predict(x_test)

# Aplicar función exponencial inversa a las predicciones
y_pred_real = np.exp(y_pred) - 1

# Generar el DataFrame para la muestra de envío
sample_submission = pd.DataFrame({'Id': test['Id'], 'Sales': y_pred_real})

# Exportar el DataFrame a un archivo CSV
sample_submission.to_csv('sample_submission3.csv', index=False)

"""**INTERPRETACION MODELOS**"""

### Clustering Jerarquico - Dendograma

from scipy.cluster import hierarchy
from scipy.spatial import distance

# Calculo Spearman correlation (rank correlation)
corr_matrix = pd.DataFrame(x_important).corr(method='spearman')
#fig, axes = plt.subplots(figsize=(20, 20))
#sns.heatmap(corr_matrix, annot = True, fmt='.2f', annot_kws={'size': 10},  vmax=.8, square=True, cmap='Blues')

# Convierto correlation values a distancias
dist_matrix = 1 - np.abs(corr_matrix)

# Creo hierarchical clustering plot
linkage = distance.squareform(dist_matrix)
dendrogram = hierarchy.dendrogram(hierarchy.linkage(linkage, method='average'))

### ICE Plot
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=[10,8])

common_params = {"grid_resolution": 20, "random_state": 42}
features_info = {"features": ["Year"], "kind": "both", "centered": True}

display = PartialDependenceDisplay.from_estimator(
    rf_reg,
    x_train,
    **features_info,
    ax=ax,
    **common_params,
)

### Contributions Chart

#Random Forest- Contributions Chart

from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt
import waterfall_chart

# Escoger una instancia para analizar
instance = x_important.iloc[0]

# Calcular las contribuciones de las características
prediction, bias, contributions = ti.predict(rf_reg_important, instance.values.reshape(1, -1))

# Preparar los datos para el gráfico de cascada
features = x_important.columns
contributions = contributions[0]
data = [(features[i], contribution) for i, contribution in enumerate(contributions)]

# Crear el gráfico de cascada
waterfall_chart.plot([x[0] for x in data], [x[1] for x in data], rotation_value=45, formatting='{:,.2f}')

"""-----------------------------------------------------
-----------------------------------------------------
-----------------------------------------------------

**Segundo problema de Machine Learning**

Un problema sencillo que se puede abordar con los datos proporcionados es la de definir si una tienda esta participando de una promocion en un dia dado o no. Esto se puede plantear como un problema de clasificación binaria, donde utilizaremos un conjunto de variables para entrenar un modelo de Random Forest Classifier.

Variables a utilizar:
Las detalladas en el documento Store, Train y Test. Decidimos no incluir las de la informacion externa (como por ejemplo Statestore) en nuestra version final, ya que habiendolo hecho en un principio las variables no resultaban relevantes para el modelo (lo puedo verificar en el analisis de Feature Importance).

La precisión del modelo se evalúa utilizando la métrica de accuracy_score y se presenta la matriz de confusión para evaluar el rendimiento del modelo en términos de falsos positivos, falsos negativos, verdaderos positivos y verdaderos negativos. Además, se muestra el informe de clasificación que proporciona métricas como precision, recall, f1-score y support para cada clase.
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dividir los datos en características (x) y etiquetas (y)
x_train = train_merged.drop(columns=['Sales', 'Date', 'Promo'])
y_train = train_merged['Promo']
x_test = test_merged.drop(columns=['Date', 'Promo'])
y_test = test_merged['Promo']  # Definir y_test

# Crear el modelo de clasificación binaria
clf = RandomForestClassifier(random_state=42)
# Entrenar el modelo
clf.fit(x_train, y_train)
# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(x_test)
# Calcular la precisión y la matriz de confusión
accuracy = accuracy_score(y_test, y_pred)
confusion_mtx = confusion_matrix(y_test, y_pred)

# Mostrar los resultados
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_mtx)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Grafico
feature_importance = clf.feature_importances_
feature_names = x_train.columns
# Ordenar características de menor a mayor importancia
sorted_indices = np.argsort(feature_importance)
sorted_feature_names = feature_names[sorted_indices]
sorted_feature_importance = feature_importance[sorted_indices]
# Crear el gráfico de barras horizontales
plt.figure(figsize=(8, 6))
plt.barh(sorted_feature_names, sorted_feature_importance)
plt.xlabel("Importancia")
plt.ylabel("Característica")
plt.title("Importancia de las Características")
plt.xticks(fontsize=8)  # Ajustar tamaño de letra en el eje x
plt.yticks(fontsize=8)  # Ajustar tamaño de letra en el eje y
plt.show()