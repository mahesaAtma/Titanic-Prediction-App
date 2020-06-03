from flask import Flask,render_template,url_for,redirect,flash
from markupsafe import escape
import pandas as pd
import numpy as np
from forms import needed_form
from learning import predictClasses

df = pd.read_csv("titanic/train.csv")
dropped_column = df.drop(['Cabin','PassengerId',"Ticket","Name"],axis=1)
sex = pd.get_dummies(dropped_column['Sex'], drop_first=True)
embark = pd.get_dummies(dropped_column['Embarked'])
dfHasil = pd.concat([sex, embark], axis=1)
dfAkhir = pd.concat([dropped_column, dfHasil], axis=1)

dfDict = df.isnull().sum().to_dict()
listData = [i for i in dfDict]

app = Flask(__name__)
app.config["SECRET_KEY"] = '5791628bb0b13ce0c676dfde280ba245'

# Route for displaying the way 
@app.route('/display',methods=("POST","GET"))
def display():
    return render_template('display.html',tables=[df.to_html(classes='data',header="true")], 
                           dfDict2=dfDict, dropped=[dropped_column.to_html(classes='data', header="true")],
                           dfAkhir=[dfAkhir.to_html(classes='data', header="true")])

# Display the application and process the prediction
@app.route('/',methods=("POST","GET"))
def index():
    prediction_knn = predictClasses()
    form = needed_form()
    Survived="null"
    category = "success"
    if form.validate_on_submit():
        # PClass data
        PClass = form.PClass.data

        # Age Data
        Age = form.Age.data

        # SibSp Data
        if form.SibSp.data == True:
            SibSp = 1
        else:
            SibSp = 0

        # Parch Data
        if form.Parch.data == True:
            Parch = 1
        else:
            Parch = 0

        # Fare Data
        Fare = form.Fare.data

        # Sex Data
        if form.Male.data == "M":      
            Sex = 1
        else:
            Sex = 0
        
        # Embarked Data
        if form.Embarked.data == "C":
            C = 1
            Q = 0
            S = 0
        elif form.Embarked.data == "Q":
            Q = 1
            C = 0 
            S = 0
        elif form.Embarked.data == "S":
            C = 0
            Q = 0
            S = 1
        Survived = "Survived" if prediction_knn.prediction(PClass,Age,
                                SibSp,Parch,Fare,Sex,C,Q,S) == [1] else "Not Survived"
        category = "success" if Survived == "Survived" else "danger"
        flash("Passenger {}".format(Survived),"{}".format(category))
        redirect(url_for("index"))
    else:
        print(form.Age.data)
    return render_template("index.html",forms=form,survived=Survived)

if __name__ == '__main__':
    app.run(debug=True)
