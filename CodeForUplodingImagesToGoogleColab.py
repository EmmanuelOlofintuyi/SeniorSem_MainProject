Python 3.10.4 (v3.10.4:9d38120e33, Mar 23 2022, 17:29:05) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
!gdown https://drive.google.com/uc?id=1xgk7svdjBiEyzyUVoZrCz4PP6dSjVL8S
  
  import sklearn.model_selection as model_selection
  
  ##upload the PASCAL files
def upload_files():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())
##shuffling the images randomly then selecting half for training set and data set
from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(DATA_IMAGES,
                                                    DATA_LABELS,
                                                    test_size = 0.15,
                                                    random_state = 41)

