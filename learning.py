import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns   
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

# Inisialisasi agar setiap kali program dijalankan, nilai
# random akan selalu sama
np.random.seed(1)


class predictClasses():
    def __init__(self):
        # membaca file titanic
        self.df = pd.read_csv('titanic/train.csv')
    
    def main(self):

        # Drop Fitur dari Cabin dan Passenger Id karena pada kasus ini
        # tidak terlalu diperlukan
        self.df.drop(['Cabin', 'PassengerId'], axis=1, inplace=True)
        print(self.df)
        # Karena data dari Sex/Gender dan Embarked merupakan bertipe
        # kategori, jadi data tersebut akan di dibagi
        sex = get_dummies(self.df['Sex'], drop_first=True)
        embark = get_dummies(self.df['Embarked'])

        # Data yang telah diolah digabung dan membuang data yang asal
        # agar tidak terdapat data yang ganda
        dfHasil = pd.concat([sex, embark], axis=1)
        self.df = pd.concat([self.df, dfHasil], axis=1)
        self.df.drop(['Name', 'Ticket', 'Sex', 'Embarked'],
                     axis=1, inplace=True)

        # Karena age berkorelasi dengan pclass, maka dapat dimanfaatkan
        # untuk mengambil rata-ratanya, lalu di set dan dimasukan ke
        # data age
        self.df['Age'] = self.df[['Pclass', 'Age']].apply(self.pengisian, axis=1)

        y = self.df['Survived'].values
        X = self.df.drop('Survived', axis=1).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Mencari nilai n terbaik untuk model KNN
        error = []
        minVar = 100
        simpanI = 0
        for i in range(1, 100):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(X_train, y_train)
            knnPredict = model.predict(X_test)
            error.append(np.mean(y_test != knnPredict))
            if minVar > np.mean(y_test != knnPredict):
                minVar = np.mean(y_test != knnPredict)
                simpanI = i

        knn = KNeighborsClassifier(n_neighbors=simpanI)
        knn.fit(X_train, y_train)

        return knn

    def pengisian(self,col):
        PClass = col[0]
        Age = col[1]

        if pd.isnull(Age):
            if PClass == 1:
                return self.df[self.df['Pclass'] == 1]['Age'].mean()//1
            elif PClass == 2:
                return self.df[self.df['Pclass'] == 2]['Age'].mean()//1
            elif PClass == 3:
                return self.df[self.df['Pclass'] == 3]['Age'].mean()//1
        else:
            return Age

    def prediction(self,PClass,Age,SibSp,Parch,Fare,Sex,C,Q,S):
        knn = self.main()
        prediction_value = [[PClass, Age, SibSp, Parch, Fare, Sex, C, Q, S]]
        prediction_value = np.array(prediction_value)
        predicted = knn.predict(prediction_value)

        return predicted

# For testing purpose
if __name__ == "__main__":
    jalan = predictClasses()
    Survived = "Survived" if jalan.prediction(2, 68,1, 0, 88.9, 1, 0, 1, 0) == [1] else "Not Survived"
    print(Survived)
