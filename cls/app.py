#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st


# # Создание дэшборда

# ## Общие функции

# pandas==1.2.4
# numpy==1.19.5

# In[13]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


# Объявим функцию, загружающую описание к датасету. Хэшируем ее, поскольку нам не нужно загружать каждый раз это же описание.

# In[23]:


@st.cache
def load_descr(filename='data/description.md'):
    with open(filename, 'r', encoding='utf-8') as fout:
        return fout.read()


# Объявим функцию, загружающую датасет. Хэшируем ее, поскольку нам не нужно загружать каждый раз новый датасет.

# In[21]:


@st.cache
def load_data(filename='data/df_train.csv'):
    df = pd.read_csv(filename, sep=',')
    return df


# Метод `print_metrics` выводит метрики для предсказанных значений
# 
# - `classification_report` печатает отчет о метриках, в том числе ошибки 1 и 2 рода;
# - `confusion_matrix` показывает, сколько из объектов были отнесены к правильному классу, а сколько - нет.

# In[30]:


def print_metrics(y_true=None, y_pred=None):
    st.write('## Print metrics')
    if y_true is None or y_pred is None:
        st.write('No data to print metrix')
        return None, None
    st.write('### Classification report.')
    cl_report = classification_report(y_true, y_pred, zero_division=0)
    st.write(cl_report)
    st.write('### Confusion matrix.')
    conf_matrix = confusion_matrix(y_true, y_pred)
    st.write(conf_matrix)
    return cl_report, conf_matrix


# Метод `fit_predict` обучает модель на размеченных данных и предсказывает метки для тестовой выборки. Особенности:
# 1. Создает базовую модель `model` с параметрами `model_params`.
# 1. Выполняет подбор гиперпараметров из дополнительных аргументов, переданных в функцию.
# 1. Если не передан `X_test`, на выход вместо предсказанных значений подается `None`.

# In[16]:


def fit_predict(X_train=[], y_train=[], X_test=None, model=DecisionTreeClassifier, model_params={},
                verbose=False, **params):
    if verbose:
        print('Обучаем {} модель.'.format(model))
    base_est = model(**model_params)
    est = GridSearchCV(base_est, params)
    est.fit(X_train, y_train)
    if verbose:
        print('Получили лучшие параметры модели: {}.'.format(est.best_params_))
    if X_test is None:
        return est.best_model_, est.best_params_, None
    y_pred = est.predict(X_test)
    return est.best_estimator_, est.best_params_, y_pred


# Проведем сериализацию четырех лучших моделей c помощью модуля `pickle`. (выполнено в модуле 3).
# 
# Напишем метод, сериализующий модель в файл.

# In[29]:


def serialize_to_file(model, filename='models/BC.dat'):
    with open(filename, 'wb') as fin:
        pickle.dump(model, fin)
    return filename


# Напишем метод, десериализующий модель из файла. Получение новой модели из файла - не сильно затратная операция, моделей не так много, поэтому можем захэшировать

# In[26]:


def deserialize_from_file(filename='models/BC.dat'):
    if filename is None:
        filename='models/BC.dat'
    with open(filename, 'rb') as fout:
        return pickle.load(fout)


# Напишем метод, загружающий датасет и выводищий его на печать

# In[38]:


def write_dataset(filename='data/df_train.csv', description='data/description.md', batch_size=100):
    st.subheader('Отобразим датасет с информацией о нем')
    st.write(load_descr(filename=description))
    st.write('Посмотрим на датасет {}'.format(filename))
    filename_new = 'data/data_prepared.csv'
    df = load_data(filename=filename)
    if df is None:
        st.write('Loading dataset {}'.format(filename_new))
    else:
        n_batch = st.slider('Select number of batch for dataset', min_value=0, max_value=(df.shape[0] - 1) // batch_size, value=0, key='n_batch')
        # load by batchs
        st.write(df.loc[n_batch*100:(n_batch+1)*100])
    return df


# Функция `ask_data` принимает ввод пользователя

# In[40]:


def ask_data():
    age = st.slider('Select your age', min_value=21, max_value=79, value=21, key='age')
    experience = st.slider('Select your experience', min_value=0, max_value=20, value=0, key='experience')
    income = st.slider('Select your income', min_value=1e4, max_value=1e7, value=1e5, step=1e4, key='income')
    current_job_years = st.slider('Select your current job years', min_value=0, max_value=14, value=0, key='current_job_years')
    current_house_years = st.slider('Select your current house years', min_value=10, max_value=14, value=10, key='current_house_years')
    
    married = st.radio('Are you married?', ['Yes', 'No'], key='married')
    car_ownership = st.radio('Do you have car?', ['Yes', 'No'], key='car_ownership')
    state = st.selectbox('Select your state', ['Kerala', 'Sikkim', 'Other'], key='state')
    house_ownership = st.selectbox('Select your ownership', ['norent_noown', 'owned', 'rent', 'other'], key='house_ownership')
    
    return np.array([[age, experience, married == 'Yes', car_ownership == 'Yes', current_job_years, 
                      current_house_years, state == 'Kerala', state == 'Sikkim', 
                      house_ownership == 'norent_noown', house_ownership == 'owned', income]])


# Функция `write_model` будет записывать модель в дэшборд

# In[41]:


def write_model(type_model='BaggingClassifier', build_in_data='data/df_train.csv', target='risk_flag', 
                is_classification=True, sep=0.5, batch_size=100):
    files = {
        'BaggingClassifier': 'models/BC.dat',
        'RandomForestClassifier': 'models/RFC.dat',
        'KNeighborsClassifier': 'models/KNN.dat',
        'DecisionTreeClassifier': 'models/DTC.dat',
    }
    model = deserialize_from_file(filename=files[type_model])
    type_data = st.selectbox('Choose type of data', ['Build-in', 'Interactive'], key='type_data')
    st.write('## Input params to model')
    if type_data == 'Build-in':
        df = load_data(filename=build_in_data)
        X_test = df.drop([target], axis=1)
        n_batch = st.slider('Select number of batch for dataset', min_value=0, max_value=(df.shape[0] - 1) // batch_size, value=0, key='n_batch_1')
        # load by batchs
        st.write(df.loc[n_batch*100:(n_batch+1)*100])
    else:
        X_test = ask_data()
        st.write('Input data')
        st.write(X_test)
    y_pred = model.predict(X_test)
    st.write('## Model predict')
    if is_classification:
        y_pred = np.where(y_pred > sep, 1, 0)
    st.write('Model {} predict values:'.format(type_model))
    if type_data == 'Build-in':
        st.write(y_pred[n_batch*100:(n_batch+1)*100])
        y_true = df[target]
        print_metrics(y_true=y_true, y_pred=y_pred)
    else:
        st.write('Risk flag: {}'.format(y_pred[0]))
    return y_pred


# ## Влияние признаков на целевую переменную

# Как уже было показано, некоторые признаки сильно влияют на целевую переменную risk_flag, а некоторые слабее. Выделим колонки, которые влияют сильнее всего. Их и оставим в качестве признаков. Весь код возьмем из модуля 2.

# In[49]:


def write_goal_data(filename = 'data/data_prepared.csv', target = 'risk_flag'):
    st.subheader('Влияние признаков на целевую переменную')
    df = load_data(filename='data/data_prepared.csv')
    X = df.drop(target, axis=1)
    y = df[target]
    # 0.001 - лучшее значение alpha по предыдущему модулю
    model = Lasso(alpha=0.001)
    model.fit(X, y)
    eps = 1e-6
    num_meaningful_params = np.sum(np.where(np.abs(model.coef_) - eps > 0, 1, 0))
    best_columns=df.columns[np.where(np.abs(model.coef_) - eps > 0)]
    
    st.write('Выберем значащие параметры с помощью Lasso: признаки')
    st.write(np.array(best_columns))
    hist_values = np.abs(model.coef_)[np.where(np.abs(model.coef_) - eps > 0)]
    st.write('имеют веса относительно целевой переменной:')
    st.bar_chart(hist_values)
    st.write('Также должен влиять параметр income, включим его.')


# ## Обоснование выбора разделения на обучающую и тестовую выборки

# Выведем обоснование разделения на обучающую и тестовую выборки

# In[ ]:


def write_explanation_split():
    st.subheader('Обоснование выбора разделения на обучающую и тестовую выборки')
    st.write('Разбиение на обучение и тест выполняем случайным образом. Мы не имеем каких-либо данных по дате обращения в компанию, поэтому имеет смысл подготовиться ко всем случаям и сделать разбиение случайно.')
    st.write('В этом есть еще одно преимущество: во 2 модуле мы заметили дисбаланс классов, от которого мы можем избавиться, поставив stratify=y в разбиении на обучение и тест.')
    st.write('Подумаем, сколько значений оставить на тест. Так как датасет достаточно большой (252000 значений), имеет смысл сделать достаточно большую тестовую выборку, скажем, 25%.')


# ## Графики

# Построим графики

# In[50]:


def draw_hist(df, columns=[]):
    for i, col in enumerate(columns):
        min_val=int(np.min(df[col]))
        max_val=int(np.max(df[col]))
        st.write('Распределение данных для колонки {}'.format(col))
        bins=max_val - min_val + 1
        hist_data = np.histogram(df[col], bins=bins, range=(min_val, max_val))[0]
        st.bar_chart(hist_data)


# In[51]:


def write_graphics(df):
    st.subheader('Построим графики данных')
    draw_hist(df, columns=['age', 'experience', 'current_job_years', 'current_house_years'])


# ## Основная функция

# Основная функция отвечает за обработку запуска приложения

# In[29]:


def main():
    page = st.sidebar.selectbox('Choose page', ['Description', 'Model info'])
    if page == 'Description':
        st.title('Описание датасета')
        df = write_dataset(filename='data/df_train.csv', description='data/description.md')
        write_goal_data()
        write_explanation_split()
        write_graphics(df)
    elif page == 'Model info':
        type_model = st.selectbox('Choose model:', 
                                  ['BaggingClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier'], key='type_model')
        write_model(type_model=type_model)
    pass


# In[ ]:


if __name__=='__main__':
    main()

