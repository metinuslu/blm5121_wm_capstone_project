import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset_from_kaggle_old(user_name: str, dataset_name: str, path: str):
    if not os.path.isfile(os.path.join(path, "bodyPerformance.csv")):
        os.makedirs(path, exist_ok=True)
        # Kaggle API objesini oluşturun
        api = KaggleApi()

        # API anahtarını yükleyin
        # api.set_config_value('username', 'asdasdasd')
        # api.set_config_value('key', 'asdasdasd')

        # api.set_config_value('username', st.secrets["kaggle"]["username"])
        # api.set_config_value('key', st.secrets["kaggle"]["key"])

        # API anahtarını yükleyin
        api.authenticate()

        # Veri setini indirin
        api.dataset_download_files(user_name + '/' + dataset_name, path=path, unzip=True)
    else:
        print("Dataset already exists!")


def download_dataset_from_kaggle(kaggle_user_name: str, kaggle_key: str, dataset_user_name: str, dataset_name: str, path: str):
    if not os.path.isfile(os.path.join(path, "bodyPerformance.csv")):
        os.makedirs(path, exist_ok=True)
        # Kaggle API objesini oluşturun
        api = KaggleApi()

        # API anahtarını yükleyin
        # api.set_config_value('username', st.secrets["kaggle"]["username"])
        # api.set_config_value('key', st.secrets["kaggle"]["key"])
        api.set_config_value('username', kaggle_user_name)
        api.set_config_value('key', kaggle_key)
        api.authenticate()

        # Veri setini indirin
        api.dataset_download_files(dataset_user_name + '/' + dataset_name, path=path, unzip=True)
    else:
        print("Dataset already exists!")