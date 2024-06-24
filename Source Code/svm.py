import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#file reading for test and train data

c_params = 0
g_params = 0
def load_dt(params):
    global c_params,g_params
    pre_process = ['dct', 'dwt', 'both']
    
    if params == 0: ##dct params
        c_params = 10
        g_params = 0.01
    elif params == 1:  ##dwt params
        c_params = 100
        g_params = 0.01
    else: 
        c_params = 100          #if both DCT + DWT
        g_params = 0.001
       
        
    process = pre_process[params]
    train = f'train-{process}.csv'
    test = f'test-{process}.csv'
    
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    
    return train, test

def make_predictions(params):
    global c_params,g_params
    #split data 40% testing, 60% training
    train_data, test_data = load_dt(params)
    pred = pd.read_csv('img.csv')


    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pred_scaled = scaler.transform(pred)


    svm_classifier = SVC(kernel='rbf', gamma = g_params, C = c_params)
    svm_classifier.fit(X_train_scaled, y_train)


    y_pred = svm_classifier.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Print evaluation scores
    print("\nAccuracy:", accuracy * 100)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(cm)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()

    inp = svm_classifier.predict(pred_scaled)
    
    print("Prediction:", inp )

    return inp
