import os
# import threading
# import time
import warnings

# from datetime import datetime
# import pandas as pd
# import src.model as model
# import src.preprocess as preprocess
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.functions import download_dataset_from_kaggle
from src.streamlit_functions import data_metadata, data_preview, data_profiling

# from st_pages import show_pages_from_config

warnings.filterwarnings("ignore")


def home():
    st.title('Web Mining Project Title')
    st.header('Web Mining Project Header')
    st.subheader('Bu uygulama Web Mining (BLM-5121) Projesi kapsamında ML Algoritmaları için geliştirilmiştir.')
    st.markdown('**1. Project Proposal:** Proje önerisi ve proje hakkında bilgi alabilirsiniz.')
    st.markdown('**2. Data Info:** Veri seti önizlemesi yapabilirsiniz. Veri seti hakkında bilgi alabilirsiniz.')
    st.markdown('**3. Multi Class Classification:** Çoklu sınıflandırma uygulamasıdır. Veri seti üzerinde çoklu sınıflandırma yapabilirsiniz.')
    st.markdown('**4. Regression:** Regresyon uygulamasıdır. Veri seti üzerinde regresyon analizi yapabilirsiniz.')
    st.markdown('**5. Clustering:** Kümeleme uygulamasıdır. Veri seti üzerinde kümeleme analizi yapabilirsiniz.')
    st.markdown('**6. App Info. & Credits:** Bu projede kullanılan Framework ve Libraryleri içermektedir.')


def data():
    st.title('Dataset Information Title')
    st.header('Dataset Information Header')
    st.subheader('Veri seti önizlemesi yapabilirsiniz. Veri seti hakkında bilgi alabilirsiniz.')
    tab1, tab2, tab3, tab4 = st.tabs(["Meta Data", "Preview", "Profile(Raw Data)", "Profile(Preprocess Data)"])
    with tab1:
        st.image(image="https://storage.googleapis.com/kaggle-datasets-images/1732554/2832282/1be2ae7e0f1bc3983e65c76bfe3a436e/dataset-cover.jpg?t=2021-11-20-09-31-54",
                 caption="Body Performance Dataset from Kaggle",
                 width=200,
                 use_column_width="auto"
                 )
        st.title('Meta Data')
        st.header("Meta Data")
        data_metadata(file_path=DATA_FILE)
        # st.page_link(page="http://www.google.com", label="Dataset Url: Kaggle", icon="🌎")
    with tab2:
        st.title('Data Preview')
        st.header("Data Preview")
        data_preview(file_path=DATA_FILE)
    with tab3:
        st.title('Raw Data Profiling')
        st.header("Raw Data Profiling")
        with open(file="data/profiling/ProfilingReport.html", encoding="utf8") as p:
            components.html(p.read(), height=4096, width=2160, scrolling=True)
    with tab4:
        st.title('Preprocess Data Profiling')
        st.header("Preprocess Data Profiling")
        with open(file="data/profiling/ProfilingReport.html", encoding="utf8") as p:
            components.html(p.read(), height=4096, width=2160, scrolling=True)


def proposal():
    # st.title('Project Proposal Title')
    # st.header('Proje Önerisi Header')
    # st.subheader('Proje önerisi ve proje hakkında bilgi alabilirsiniz.')
    with open(file="ProjectProposal.md", encoding="utf8") as p:
        st.markdown(p.read())


def classification():
    st.title('Multi Class Classification Title')
    st.header('Multi Class Classification Algorithms Header')
    st.subheader('Çoklu sınıflandırma uygulamasıdır. Veri seti üzerinde çoklu sınıflandırma yapabilirsiniz.')
    st.write('1. Veri seti yükleme')
    
    option = st.selectbox("Multi Class Classification Alogritmaları?",
                          ("Decision Tree", "KNN", "LightGBM"),
                          index=None,
                          placeholder="Model seçiniz...",
                          )
    st.write("Model Seçimi:", option)


def regression():
    st.title('Regression Title')
    st.header('Regression Algorithms Header')
    st.subheader('Regresyon uygulamasıdır. Veri seti üzerinde regresyon analizi yapabilirsiniz.')
    # st.write('1. Veri seti yükleme')
    tab1, tab2, tab3 = st.tabs(["Training", "Chart", "Prediction"])
    # tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

    with tab1:
        st.header("Training")
        st.write('Training işlemi yapılacak.')
        # st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

    with tab2:
        st.header("Chart")
        st.write('Chart işlemi yapılacak.')
        tab2_1, tab2_2, tab2_3 = st.tabs(["Loss Chart", "Accuracy Chart", "Other Chart"])
        with tab2_1:
            st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        with tab2_2:
            st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        with tab2_3:
            st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
        st.header("Prediction")
        st.write('Prediction işlemi yapılacak.')
        # st.image("https://static.streamlit.io/examples/owl.jpg", width=200)    


def clustering():
    st.title('Clustering Title')
    st.header('Clustering Algorithms Header')
    st.subheader('Kümeleme uygulamasıdır. Veri seti üzerinde kümeleme analizi yapabilirsiniz.')
    st.write('1. Veri seti yükleme')


def credits():
    st.title('App Info. & Credits Title')
    st.header('App Info. & Credits Header')
    st.subheader('App Info. & Credits: Bu projede kullanılan Framework ve Libraryleri içermektedir.') 

    st.markdown('**Programming Language:** Python 3.12')
    st.markdown('**Libraries:** Pandas, Scikit-learn, Numpy, Matplotlib, Seaborn, Plotly')
    st.markdown('**UI:** Streamlit')
    st.markdown('**Dev. Tools:** Docker & Git')
    st.markdown('**Dash Url:** [StreamLit App](https://web-mining-project.streamlit.app/)')
    st.markdown('**Developed by:** Metin Uslu & Anıl Özcan')
    st.page_link(page="http://www.google.com", label="Google", icon="🌎")
    # st.page_link("your_app.py", label="Home", icon="🏠")
    # st.page_link("pages/page_1.py", label="Page 1", icon="1️⃣")
    # st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)


def menu(user_name=None, user_password=None):
    """Streamlit UI Menu
    Params:
        user_name: str 
        user_password: str
    """

    st.sidebar.title('Web Mining Project')
    menu = {
        'Giriş': home,
        'Project Proposal': proposal,
        'Dataset Info': data,
        'Multi Class Classification Algorithms': classification,
        'Regression Algorithms': regression,
        'Clustering Algorithms': clustering,
        'App. Info. & Credits': credits
    }

    # dev_user = "webmining"
    # dev_pass = "webmining"
    if st.session_state.get('login_success'):
        choice = st.sidebar.radio('Applications', list(menu.keys()))
        menu[choice]()
    else:
        with st.sidebar:
            with st.form(key='login_form'):
                st.title('Loging Page')
                username = st.text_input('User Name')
                password = st.text_input('Password', type='password')
                if st.form_submit_button('Login'):
                    if username == user_name and password == user_password:
                        st.session_state['login_success'] = True
                        st.success('Giriş başarılı, yönlendiriliyorsunuz...')
                        st.experimental_rerun()
                    else:
                        st.error('Kullanıcı adı veya şifre yanlış.')
                        st.session_state['login_success'] = False
    # show_pages_from_config()


if __name__ == "__main__":

    # Set Constants
    ROOT_PATH = os.getcwd()
    CFG_PATH = os.path.join(ROOT_PATH, 'cfg')
    ENV = os.path.join(CFG_PATH, '.env')
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed')
    PROFILLING_PATH = os.path.join(DATA_PATH, 'profiling')

    DATA_FILE = os.path.join(RAW_DATA_PATH, 'bodyPerformance.csv')
    
    # Load Environment Variables
    load_dotenv(dotenv_path=ENV, encoding='utf-8', verbose=False)
    
    # Get Constants
    USER_NAME = os.environ.get("USER_NAME")
    USER_PASSWORD = os.environ.get("USER_PASSWORD")

    # st.set_page_config(page_title=None,
    #                    page_icon=None,
    #                    layout="centered",
    #                    initial_sidebar_state="auto",
    #                    menu_items=None)
        
    st.set_page_config(
        page_title="Web Mining Project UI ",
        page_icon=":gem:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
                    'Report a bug': "https://www.extremelycoolapp.com/bug",
                    'About': "# This is a header. This is an *extremely* cool app!"
                    }
                    )
    
    download_dataset_from_kaggle(user_name="kukuroo3", dataset_name="body-performance-data", path=RAW_DATA_PATH)
    data_profiling(file_path=DATA_FILE, report_path=PROFILLING_PATH)
    menu(user_name=USER_NAME, user_password=USER_PASSWORD)