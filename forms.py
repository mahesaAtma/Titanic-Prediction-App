from wtforms import IntegerField,RadioField,BooleanField,SubmitField,SelectField
from wtforms.validators import DataRequired,Length,NumberRange,Optional,Email
from flask_wtf import FlaskForm

class needed_form(FlaskForm):
    PClass = IntegerField("Passenger Class", validators=[
                          DataRequired(), NumberRange(min=1,max=3)])
    Age = IntegerField("Passenger Age", validators=[
                        DataRequired(), NumberRange(min=5, max=150)])
    Fare = IntegerField("Passenger Fare", validators=[
                        DataRequired(), NumberRange(min=1, max=513)])
    Male = SelectField("Passenger Sex", 
                        choices=[("M","Male"),("F","Female")])
    Embarked = SelectField("Embarked", choices=[("C","C"),("Q","Q"),("S","S")])
    SibSp = BooleanField("SibSp", validators=[Optional()])
    Parch = BooleanField("Parch", validators=[Optional()])
    Submit = SubmitField("Submit")
        
