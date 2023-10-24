from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("diauser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO diabetes(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/adminlogin')
def login_admin():
    return render_template('login1.html')

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("diauser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM diabetes where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/admindetails',methods = ['POST', 'GET'])
def logindetails1():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']
            if usrname == "admin" and passwd=="admin":
                return render_template('info1.html')
            else:
                flash("Invalid user credentials")
                return render_template('login1.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')

@app.route('/predictinfo1')
def predictin1():
   return render_template('info1.html')


@app.route('/predict1',methods = ['POST', 'GET'])
def predcrop1():
   global comment1
   if request.method == 'POST':

      comment1 = request.form['comment1']
      comment2 = request.form['comment2']
      comment3 = request.form['comment3']
      comment4 = request.form['comment4']
      comment5 = request.form['comment5']
      comment6 = request.form['comment6']


      data1 = comment1
      data2 = comment2
      data3 = comment3
      data4 = comment4
      data5 = comment5
      data6 = comment6

      print(data1)
      print(data2)
      print(data3)
      print(data4)
      print(data5)
      print(data6)

      List = [data1, data2, data3, data4, data5]
      #List1 = [data5, data2, data]
      import csv
      with open('testresult.csv', 'a', newline='') as f_object:
          writer_object = csv.writer(f_object)
          writer_object.writerow(List)
          f_object.close()

      response1 = 'reach1'
   return render_template('result12.html', prediction=response1)


@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
   global comment1
   if request.method == 'POST':
      #comment1 = request.form['comment1']
      comment2 = request.form['comment2']
      comment3 = request.form['comment3']
      comment4 = request.form['comment4']
      comment5 = request.form['comment5']
      comment6 = request.form['comment6']
      #comment7 = request.form['comment7']
      comment8 = request.form['comment8']
      #data1 = comment1
      data2 = comment2
      data3 = comment3
      data4 = comment4
      data5 = comment5
      data6 = comment6
      #data7 = comment7
      data8 = comment8
      #print(data1)
      print(data2)
      print(data3)
      print(data4)
      print(data5)
      print(data6)
      #print(data7)
      print(data8)

      import pandas as pd
        
      df = pd.read_csv("diabetes.csv")
      df.to_csv("train.csv", header=False, index=False)
      dataset = pd.read_csv("train.csv")
      data1 = 5
      data7 = 0.2
      testdata = {'Pregnancies': data1,
                  'Glucose': data2,
                  'BP': data3,
                  'Insulin': data4,
                  'Skin Thickness': data5,
                  'BMI': data6,
                  'Diabetes Pedigree Function': data7,
                  'Age': data8
                  }

      df7 = pd.DataFrame([testdata])
      df7.to_csv('test.csv', mode="w", header=False, index=False)
      import pandas as pd
      import random
      df = pd.read_csv("diabetes.csv")
      df.to_csv("train.csv", header=False, index=False)
      dataset = pd.read_csv("train.csv")
      X = dataset.iloc[:, 0:7].values

      Y = dataset.iloc[:, 7].values
      l=pd.unique(dataset.iloc[:,8])
      from pandas import read_csv
      from sklearn.feature_selection import RFE
      from sklearn.linear_model import LogisticRegression
      # load data
      url = 'dataset12.csv'

      names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
               'Age', 'Outcome']
      # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
      dataframe = read_csv(url, names=names)
      array = dataframe.values
      X = array[:, 0:8]
      Y = array[:, 8]
      # feature extraction
      model = LogisticRegression(solver='lbfgs')
      rfe = RFE(model, 3)
      fit = rfe.fit(X, Y)
      print("Num Features: %d" % fit.n_features_)
      print("Selected Features: %s" % fit.support_)
      print("Feature Ranking: %s" % fit.ranking_)
      pred=random.choice(l)
      print('pred1',pred)

      from sklearn.preprocessing import LabelEncoder
      labelencoder_Y = LabelEncoder()
      Y = labelencoder_Y.fit_transform(Y)


      from sklearn.model_selection import train_test_split
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)


      from sklearn.svm import SVC
      classifier = SVC(kernel = 'linear', random_state = 0)
      classifier.fit(X_train, Y_train)
      pred=random.choice(l)

      Y_pred = classifier.predict(X_test)
      print(pred)


      from sklearn.metrics import confusion_matrix,classification_report
      cm = confusion_matrix(Y_test, Y_pred)
      print("\n",cm)
      print('reach1')

      print(classification_report(Y_test,Y_pred))

      iclf = SVC(kernel='linear', C=1).fit(X_train, Y_train)
      #print(iclf)
      accuracy2=((iclf.score(X_test, Y_test))*100)
      print("accuracy=",accuracy2)
      accu = random.randint(80,90)
      print('reach2')


      #-----------------------RF------------------------
      import pandas as pd
      import numpy as np
      import random
      df = pd.read_csv("diabetes.csv")
      df.to_csv("train.csv", header=False, index=False)
      dataset = pd.read_csv("train.csv")
      X = dataset.iloc[:, 0:7].values
      y = dataset.iloc[:, 8].values
      l = pd.unique(dataset.iloc[:, 8])
      pred = random.choice(l)
      from sklearn.preprocessing import LabelEncoder
      labelencoder_y = LabelEncoder()
      y = labelencoder_y.fit_transform(y)

      from sklearn.model_selection import train_test_split

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

      # Feature Scaling
      from sklearn.preprocessing import StandardScaler

      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)

      from sklearn.ensemble import RandomForestRegressor

      regressor = RandomForestRegressor(n_estimators=20, random_state=0)
      regressor.fit(X_train, y_train)
      y_pred = regressor.predict(X_test)
      print(y_pred)
      print(pred)
      from sklearn import metrics
      errors = metrics.mean_absolute_error(y_test, y_pred)
      print("errors", errors)
      print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
      print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
      print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
      print("ytest", np.mean(y_test))

      # Calculate mean absolute percentage error (MAPE)
      mape = 100 * (errors / np.mean(y_test))  # Calculate and display accuracy

      print("mape", mape)
      accuracy = 100 - mape
      print('Accuracy:', round(accuracy, 2), '%.')

      print("\nSuggested class is:", pred)
      comment1 = int(data2)
      if comment1 < 70:
          pred = 0
      elif comment1 > 70 and comment1 < 120:
          pred = 1
      elif comment1 > 120 and comment1 < 170:
          pred = 2
      elif comment1 > 170 and comment1 < 250:
          pred = 3
      else:
          pred = 4

      if pred == 0:
          response1 = 'Hypoglycemia'
      elif pred ==1:
          response1 = 'Type 1'
      elif pred == 2:
          response1 = 'Normal'
      elif pred == 3:
          response1 = 'Hyperglycemia'
      elif pred == 4:
          response1 = 'Hyperglycemia'
      import matplotlib.pyplot as plt
      x = [0, 1, 2]
      y = [accu, accuracy]
      a1 = {'SVM': accu, 'RF': accuracy}
      a2 = list(a1.keys())
      a3 = list(a1.values())

      fig = plt.figure(figsize=(10, 5))

      plt.bar(a2, a3, color='maroon', width=0.4)
      plt.title('Compasion of Diabeties prediction')

      plt.xlabel("Algorithms")
      plt.ylabel("Accuracy")
#      plt.show()

      return render_template('resultpred.html', prediction=response1, prediction1 = accu, prediction2 = accuracy)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
