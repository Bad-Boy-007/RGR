import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle
import seaborn as sns
import numpy as np
from pycaret.regression import *

#------------------------------------------------------------------------------------------------------------------

data = pd.read_csv("kc_house_data.csv")
data['bathrooms']=data['bathrooms'].astype(int)
data['price']=data['price'].astype(int)
data.pop('date')
data.pop('id')
data.pop('lat')
data.pop('long')
data.pop('zipcode')
data['floors']=data['floors'].astype(int)
data=data.drop_duplicates()
data.head()

#------------------------------------------------------------------------------------------------------------------

neoroReg=keras.models.load_model('models/model.keras')
lightgbm=load_model('models/lightgbm')
with open("models/bagging_model.pkl","rb") as f:
    baggingReg=pickle.load(f)
with open("models/gbr_model.pkl","rb") as f:
    gradientBoostingReg=pickle.load(f)
with open("models/stk_model.pkl","rb") as f:
    stackingReg=pickle.load(f)

#------------------------------------------------------------------------------------------------------------------

def main():
    select_page = st.sidebar.selectbox("Page list", ("Заголовок","Описание датасета", "Графики","Прогнозы"), key = "Select")
    if (select_page == "Заголовок"):
        title_page()

    elif (select_page == "Описание датасета"):
        description_page()

    elif (select_page == "Графики"):
        visualization()

    elif (select_page == "Прогнозы"):
        prediction_page()

#------------------------------------------------------------------------------------------------------------------

def make_prediction_single(df):
    st.header("Результат прогноза:")
    lightgbm_result=lightgbm.predict(df)
    st.write(f"Результат LightGBM: {round((float(lightgbm_result)), 3)}$")
    baggingReg_result=baggingReg.predict(df)
    st.write(f"Результат BaggingRegressor: {round(float(baggingReg_result), 3)}$")
    gradientBoostingReg_result=gradientBoostingReg.predict(df)
    st.write(f"Результат GradientBoostingRegressor: {round(float(gradientBoostingReg_result), 3)}$")
    stackingReg_result=stackingReg.predict(df)
    st.write(f"Результат StackingRegressor: {round(float(stackingReg_result), 3)}$")
    neoroReg_result=neoroReg.predict(df)
    st.write(f"Результат нейронной сети: {round(float(neoroReg_result), 3)}$")

#------------------------------------------------------------------------------------------------------------------

def title_page():
    st.title('РГР по машинному обучению, студент: Миронов Д.Г. , учебная группа: ФИТ-222')
    
#------------------------------------------------------------------------------------------------------------------

def description_page():
    st.title('Описание датасета')
    st.write("Датасет о данных цен недвижимости в King County, Washington State, USA")
    st.write(data)
    st.header("Описание столбцов")
    st.write("- price: стоимость дома ")
    st.write("- bedrooms: число спален")
    st.write("- bathrooms: число ванных комнат")
    st.write("- sqft_living: число квадратных футов жилого пространства")
    st.write("- sqft_lot15: число квадратных футов пространства")
    st.write("- floors: число этажей")
    st.write("- waterfront: находождение у побережья")
    st.write("- view: оценка вида дома (если не оценивался, то 0)")
    st.write("- condition: состoяние дома 1 до 5")
    st.write("- grade: оценка дома")
    st.write("- sqft_above: общая площадь в футах")
    st.write("- sqft_basement: площадь подвала в футах")
    st.write("- yr_built: год постройки")
    st.write("- yr_renovated: год реновации дома (если не было, то 0)")
    st.write("- sqft_living15: количество квадратных футов жилого пространства (в пересчёте)")
    st.write("- sqft_lot15: количество квадратных футов пространства (в пересчёте)")

#------------------------------------------------------------------------------------------------------------------

def prediction_page():
    st.header("Интерактивный ввод данных")

    sqft_living= st.slider("Choose sqft_living", 200, 20000)
    sqft_lot= st.slider("Choose sqft_lot", 200, 2000000)
    sqft_basement= st.slider("Choose sqft_basement", 0, 8000)
    sqft_above= st.slider("Choose sqft_above", 200, 20000)
    sqft_living15= st.slider("Choose sqft_living15",200, 20000)
    sqft_lot15 = st.slider('Choose sqft_lot15',200, 2000000)


    bedrooms = st.number_input('Choose the number of bedrooms', min_value= 0,
        max_value=100, value =1, step=1)
    bathrooms = st.number_input('Choose the number of bathrooms', min_value= 0,
        max_value=100, value =1, step=1)
    floors = st.number_input('Choose the number of floors', min_value= 1,
        max_value=100, value =1, step=1)
    
    condition = st.slider('Choose condition',1, 5)
    grade = st.slider('Choose grade',1, 13)
    view = st.slider('Choose view',0,4)
    yr_built = st.slider('Choose year_built',1800, 2015)

    Question_of_renovation = st.selectbox("Renovated?", ("Yes", "No"), key = "answer")
    if (Question_of_renovation == "Yes"):
        yr_renovated = st.slider('Choose renovation year',1800, 2015)

    elif (Question_of_renovation == "No"):
        yr_renovated=0

    checkbox_one = st.checkbox("On the coast?")

    if (checkbox_one):
        waterfront = 1
    else:
        waterfront = 0


    arrayToPredict=[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,
                             sqft_above,sqft_basement,yr_built,yr_renovated,sqft_living15,sqft_lot15]
    arrayToPredictNames=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade',
                             'sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']
    st.header("Введённые вами данные:")

    new_df=pd.DataFrame(data=np.array([arrayToPredict]),columns=arrayToPredictNames )
    st.write(new_df)
    getPredButton=st.button("Получить предсказание")
    if getPredButton:
        make_prediction_single(pd.DataFrame(data=np.array([arrayToPredict]),columns=arrayToPredictNames ))

    st.header("Загрузить свой датасет для обработки")
    uploaded_file = st.file_uploader("Выберите файл в формате .csv", type='csv')
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        if 'price' in dataframe.columns:
            dataframe.pop('price')
        getPredButton1=st.button("Получить предсказание при помощи LightGBM")
        getPredButton2=st.button("Получить предсказание при помощи BaggingRegressor")
        getPredButton3=st.button("Получить предсказание при помощи GradientBoostingRegressor")
        getPredButton4=st.button("Получить предсказание при помощи StackingRegressor")
        getPredButton5=st.button("Получить предсказание при помощи нейронной сети")
        if getPredButton1:
            linReg_result=linReg.predict(dataframe)
            st.write('Результат LightGBM:', pd.DataFrame(linReg_result, columns=["predicted_price"]))
        if getPredButton2:
            baggingReg_result=baggingReg.predict(dataframe)
            st.write("Результат BaggingRegressor:", pd.DataFrame(baggingReg_result,columns=['predicted_price']))
        if getPredButton3:
            gradientBoostingReg_result=gradientBoostingReg.predict(dataframe)
            st.write("Результат GradientBoostingRegressor:", pd.DataFrame(gradientBoostingReg_result,columns=['predicted_price']))
        if getPredButton4:
            stackingReg_result=stackingReg.predict(dataframe)
            st.write("Результат StackingRegressor:", pd.DataFrame(stackingReg_result,columns=['predicted_price']))
        if getPredButton5:
            neoroReg_result=neoroReg.predict(dataframe)
            st.write("Результат нейронной сети:", pd.DataFrame(neoroReg_result,columns=['predicted_price']))

#------------------------------------------------------------------------------------------------------------------

def visualization():
    st.header("Соотношение домов по близости с побережьем")
    fig=plt.figure()
    size=data.groupby("waterfront").size()
    plt.pie(size.values,labels=size.index,autopct='%1.0f%%')
    st.pyplot(plt)

    st.header("Корреляция стоимости с количеством спален, жилого пространства и состояния дома")
    fig=plt.figure()
    fig.add_subplot(sns.heatmap(data[["price",'bedrooms','sqft_living','condition']].corr(),annot=True))
    st.pyplot(fig)

    st.header("Boxplot о распределении площадей жилищ")
    fig=plt.figure()
    plt.boxplot(data[['sqft_living','sqft_living15','sqft_basement']],labels=['sqft_living','sqft_living15','sqft_basement'])
    st.pyplot(plt)

    st.header("Histogram, показывающая, оценки домов")
    fig=plt.figure()
    plt.hist(data['grade'])
    st.pyplot(plt)

#------------------------------------------------------------------------------------------------------------------

main()
