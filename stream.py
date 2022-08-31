import streamlit as st
import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
warnings.filterwarnings('ignore')

TITLE = "Mauna Loa Gözlemevi, Hawaii, ABD'deki Sürekli Hava Örneklerinden Atmosferdeki CO2 Miktarı "
if 'ses' not in st.session_state:
    st.session_state['ses'] = False
if 'des' not in st.session_state:
    st.session_state['des'] = False
if 'tes' not in st.session_state:
    st.session_state['tes'] = False


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache
def get_data():
    data = sm.datasets.co2.load_pandas()
    y = data.data
    y = y['co2'].resample('MS').mean()
    y = y.fillna(y.bfill())
    return y


def ts_decompose(y, model="additive"):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    return fig


def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    fig = go.Figure()
    fig.add_scatter(x=train.index, y=train.values, name='Train')
    fig.add_scatter(x=test.index, y=test.values, name='Test', line_color='red')
    fig.add_scatter(x=y_pred.index, y=y_pred.values, name='Prediction', line_color='green')
    fig.update_layout(title=f"{title}, MAE: {round(mae,2)}", xaxis_title='Date', yaxis_title='CO2')
    return fig


def introduction():
    st.title('Streamlit ile Smoothing Methods')
    st.header('Streamlit nedir?')
    st.write('Streamlit Python kodlarınızı web sayfalarına çevirmenize yarar. Herhangi bir front-end bilgisine '
            'ihtiyaç duymadan interaktif bir web sayfası oluşturabilirsiniz.')
    st.header('Streamlit ile neler yapılabilir?')
    st.write("Streamlit veri bilimi projenizi sunmanızın yanında kullanıcıya 'Keşifçi Veri Analizi' ve 'Modelleme' "
             "kısımlarında etkileşim imkanı sunar.")
    st.header('Veri Seti')
    y = get_data()
    st.plotly_chart(px.line(y, title=TITLE))
    fig = ts_decompose(y, model="additive")
    st.pyplot(fig)


def change_state(session, state):
    session['des'] = session['ses'] = session['tes'] = False
    session[state] = True


def ses(container):
    y = get_data()
    container.header('Single Exponential Smoothing')
    train = y[:'1997-12-01']
    test = y['1998-01-01':]
    # Modeling
    selected_smoothing_level = st.slider('Smoothing Level', min_value=0.5, max_value=1., step=0.01, value=0.5)
    ses_model = SimpleExpSmoothing(train).fit(smoothing_level=selected_smoothing_level)
    y_pred = ses_model.forecast(48)
    st.plotly_chart(plot_co2(train, test, y_pred, "Single Exponential Smoothing"))


def des(container):
    y = get_data()
    container.header('Double Exponential Smoothing')
    train = y[:'1997-12-01']
    test = y['1998-01-01':]
    # Modeling
    selected_smoothing_level = st.slider('Smoothing Level', min_value=0.01, max_value=1., step=0.01, value=0.5)
    selected_smoothing_slope = st.slider('Smoothing Slope', min_value=0.01, max_value=1., step=0.01, value=0.5)
    des_model = ExponentialSmoothing(train,
                                     trend='add').fit(smoothing_level=selected_smoothing_level,
                                                      smoothing_slope=selected_smoothing_slope)
    y_pred = des_model.forecast(48)
    st.plotly_chart(plot_co2(train, test, y_pred, "Double Exponential Smoothing"))


def tes(container):
    y = get_data()
    container.header('Triple Exponential Smoothing')
    train = y[:'1997-12-01']
    test = y['1998-01-01':]
    # Modeling
    selected_smoothing_level = st.slider('Smoothing Level', min_value=0.01, max_value=1., step=0.01, value=0.5)
    selected_smoothing_slope = st.slider('Smoothing Slope', min_value=0.01, max_value=1., step=0.01, value=0.5)
    selected_smoothing_seasonal = st.slider('Smoothing Seasonal', min_value=0.01, max_value=1., step=0.01, value=0.5)
    tes_model = ExponentialSmoothing(train,
                                     trend='add',
                                     seasonal='add',
                                     seasonal_periods=12).fit(smoothing_level=selected_smoothing_level,
                                                              smoothing_slope=selected_smoothing_slope,
                                                              smoothing_seasonal=selected_smoothing_seasonal)
    y_pred = tes_model.forecast(48)
    st.plotly_chart(plot_co2(train, test, y_pred, "Triple Exponential Smoothing"))


def main():
    introduction()
    local_css('style.css')
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.button('SES', on_click=lambda: change_state(st.session_state, 'ses'))
    col2.button('DES', on_click=lambda: change_state(st.session_state, 'des'))
    col3.button('TES', on_click=lambda: change_state(st.session_state, 'tes'))
    container = st.container()
    if st.session_state['ses']:
        ses(container)
    elif st.session_state['des']:
        des(container)
    elif st.session_state['tes']:
        tes(container)
    else:
        container.write('Please select a method')


if __name__ == '__main__':
    main()
