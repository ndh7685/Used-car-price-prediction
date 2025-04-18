#Import thư viện
import pandas as pd
import numpy as np
import math
import os
from os import listdir
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm

#Hàm đọc dữ liệu
def read_data():
  dirs = os.listdir('data')
  basic = 'data/'
  df = pd.DataFrame(data = None)
  for file in dirs:
    url =  basic + file
    new_df = pd.read_csv(url) #đọc từng file trong folder
    new_df['automaker'] = file.split(".")[0] #tạo cột hãng xe
    if file == 'hyundi.csv': 
      new_df.rename(columns = {'tax(£)': 'tax'}, inplace =True) #đổi tên cột thuế
    df = pd.concat([df,new_df]) #gộp file
  categorical_cols = ['model','transmission','fuelType','automaker']
  for categorical_col in categorical_cols:
    df[categorical_col] = df[categorical_col].astype('category')
  df['ln_mileage'] = df['mileage'].apply(lambda x: math.log(x))
  df['ln_mpg'] = df['mpg'].apply(lambda x: math.log(x))
  avg = df[df['ln_mpg'].notnull()]['ln_mpg'].mean()
  df['ln_mpg'] = df['ln_mpg'].fillna(avg)
  df['tax'] = df['tax'].fillna(df[df['tax'].notnull()]['tax'].mean())
  df['Class'] = df['automaker'].apply(lambda x: fill_class(x))
  df['year'] = pd.cut(df['year'],
                    bins = [0, 2015.9, 2016.9, 2018.9, 2061],
                    labels =['hơn 5 năm','4-5 năm','2-4 năm','dưới 2 năm'])
  df['engineSize'] = pd.cut(df['engineSize'],
                    bins = [-1, df['engineSize'].quantile(0.25)-0.01, df['engineSize'].quantile(0.5)-0.01, df['engineSize'].quantile(0.75)-0.01, df['engineSize'].max()+0.01],
                    labels =['Small','Medium','Large','Very Large'])
  cols = ['ln_mileage','tax','ln_mpg','price']
  for col in cols:
      # Tính IQR của biến
      Q1 = df[col].quantile(0.25)
      Q3 = df[col].quantile(0.75)
      IQR = Q3 - Q1

      # Xác định ngưỡng trên và ngưỡng dưới
      upper_threshold = Q3 + (1.5 * IQR)
      lower_threshold = Q1 - (1.5 * IQR)

      # Thay thế các giá trị outlier
      df[col] = df[col].apply(lambda x: upper_threshold if x >= upper_threshold else (lower_threshold if x <= lower_threshold else x))
      cat_features = ['year','transmission','fuelType','engineSize','Class']
      num_features = ['ln_mileage','tax','ln_mpg']
      label = ['price']
      features = cat_features + num_features
      df = df[features+label].reset_index(drop=True)
      X = df[features].reset_index(drop=True)
      y = df['price'].reset_index(drop=True)
  return df, X, y

#Hàm fill Class
def fill_class(x):
  if x in ('merc','audi','bmw'):
    return "Luxury"
  if x in ('vw','skoda','focus'):
    return "Mid-range"
  else:
    return "Affordable"

#Hàm Fit chuyển đổi dữ liệu
def preprocessing(X):
  #Đường ống 
    cat_features = ['year','transmission','fuelType','engineSize','Class']
    num_features = ['ln_mileage','tax','ln_mpg']
    num_transformer = Pipeline(steps = [
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('encoder',OneHotEncoder())
    ])
    #Khai báo quá trình xử lý
    preprocessor = ColumnTransformer([
        ('num',num_transformer,num_features),
        ('cat',cat_transformer,cat_features),
    ])
    #Fit vào bộ dữ liệu
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, preprocessor

#Xây dựng mô hình Machine Learning
def build_model_by_sklearn(X_transformed, y):
     from sklearn import linear_model
     linear_regression = linear_model.LinearRegression()
     linear_regression.fit(X_transformed, y)
     intercept = linear_regression.intercept_
     coef = linear_regression.coef_
     return intercept, coef

# Từ X đã chuyển đổi sang X trong mô hình
def from_X_to_Xols(df , X_new, ols_scaler):
  cat_cols = ['year', 'transmission','fuelType','engineSize','Class']
  num_cols = ['ln_mileage','tax','ln_mpg']
  for cat_col in cat_cols:
    min_value = df.groupby(cat_col)['price'].sum().sort_values().index[0]
    values = df[cat_col].unique()
    for value in values:
      X_new[value] = X_new[cat_col].apply(lambda x: 0 if x == min_value else (1 if x == value else 0))
  remove_cols = cat_cols
  X_new = X_new.drop(columns = remove_cols,axis =1)
  X_new[num_cols] = pd.DataFrame(data = ols_scaler.transform(X_new[num_cols]), columns= num_cols)
  return X_new

def build_model_by_sm(df,X,y):
  X = sm.add_constant(X)
  # cols = ['ln_mileage','tax','ln_mpg','price']
  # for col in cols:
  #   # Tính IQR của biến
  #   Q1 = df[col].quantile(0.25)
  #   Q3 = df[col].quantile(0.75)
  #   IQR = Q3 - Q1

  #   # Xác định ngưỡng trên và ngưỡng dưới
  #   upper_threshold = Q3 + (1.5 * IQR)
  #   lower_threshold = Q1 - (1.5 * IQR)

  #   # Thay thế các giá trị outlier
  #   df[col] = df[col].apply(lambda x: upper_threshold if x >= upper_threshold else (lower_threshold if x <= lower_threshold else x))
  #Mã hóa các cột category
  # Thực hiện encode cho các cột categorical
  cat_cols = ['year', 'transmission','fuelType','engineSize','Class']
  for cat_col in cat_cols:
    min_value = df.groupby(cat_col)['price'].sum().sort_values().index[0]
    values = df[cat_col].unique()
    for value in values:
      df[value] = df[cat_col].apply(lambda x: 0 if x == min_value else (1 if x == value else 0))
  remove_cols = cat_cols+ ['price']
  y = df['price']
  X = df.drop(columns = remove_cols,axis =1)
  # Thực hiện scaler cho các cột numerical
  from sklearn.preprocessing import StandardScaler
  num_cols = ['ln_mileage','tax','ln_mpg']
  scaler = StandardScaler()
  ols_scaler = scaler.fit(X[num_cols])
  X[num_cols] = pd.DataFrame(data = ols_scaler.transform(X[num_cols]), columns= num_cols)
  model = sm.OLS(y, X)
  results = model.fit()
  return results, ols_scaler

#Hàm đổi dữ liệu năm
def fill_year(year):
  year_today = date.today().year
  if year <= year_today-2:
    return 'dưới 2 năm'
  elif year <= year_today - 4:
    return '2-4 năm'
  elif year <= year_today - 5:
    return '4-5 năm'
  else:
    return 'hơn 5 năm'
  
#Hàm đổi dữ liệu dung tích động cơ
def fill_engineSize(x):
  if x < 1.2:
    return 'Small'
  elif x < 1.5:
    return 'Medium'
  elif x < 2:
    return 'Large'
  else:
    return 'Very Large'

#Hàm đổi dữ liệu hãng xe
def fill_class(x):
  if x in ('merc','audi','bmw'):
    return "Luxury"
  if x in ('vw','skoda','focus'):
    return "Mid-range"
  else:
    return "Affordable"

#Hàm chuyển dữ liệu từ từ form nhập thành dữ liệu được chuẩn hóa
def from_df_to_X(year, transmission, mileage, fuelType, tax, mpg, engineSize,automaker):
  Class = fill_class(automaker)
  year = fill_year(year)
  ln_mpg = math.log(float(mpg))
  ln_mileage = math.log(float(mileage))
  engineSize = fill_engineSize(engineSize)
  data = {'year': [year],
          'transmission': [transmission],
          'fuelType': [fuelType],
          'engineSize': [engineSize],
          'Class': [Class],
          'ln_mileage': [ln_mileage],
          'tax': [tax],
          'ln_mpg': [ln_mpg]
          }
  X = pd.DataFrame(data=data)
  return X

#Hàm chuyển tự dữ liệu chuẩn hóa thành dữ liệu dùng trong mô hình
def from_X_to_X_transformed(X_new, preprocessor):
  cat_features = ['year','transmission','fuelType','engineSize','Class']
  for cat_feature in cat_features:
    X_new[cat_feature] = X_new[cat_feature].astype('category')
  #Fit và transform vào bộ dữ liệu
  X_new_transformed = preprocessor.transform(X_new)
  return X_new_transformed


# Hàm chạy app
def run_app(intercept, coef, preprocessor,ols_scaler, results, df):
    from flask import Flask, render_template, request

    app = Flask(__name__, static_url_path='/static')

    @app.route('/')
    def home():
        return render_template('index.html')


    @app.route('/predict', methods=['POST'])
    def predict():
###
        # Lấy giá trị đầu vào từ form
        year = float(request.form['year'])
        transmission = request.form['transmission']
        mileage = float(request.form['mileage'])
        fuelType = request.form['fuelType']
        tax = float(request.form['tax'])
        mpg = float(request.form['mpg'])
        automaker = request.form['automaker']
        engineSize = float(request.form['engineSize'])
        isols = request.form['isols']
        
        #Gán các dữ liệu trong form vào data
        X_new = from_df_to_X(year, transmission, mileage, fuelType, tax, mpg, engineSize,automaker)

        # Tiến hành dự báo với mô hình
        if isols == 'ols':
          X_new_transformed  = from_X_to_Xols(df, X_new, ols_scaler)
          print(X_new_transformed)
          prediction = results.predict(X_new_transformed)
          prediction = '{:,.0f}'.format(prediction[0]) + " $"
          image_source = "/static/images/residuals-ols.png"

        else:
          X_new_transformed = from_X_to_X_transformed(X_new, preprocessor)
          prediction = np.dot(coef,X_new_transformed.reshape(23,)) + intercept
          prediction = '{:,.0f}'.format(prediction) + " $"
          image_source = "/static/images/residuals-sklearn.png"
###       
        # prediction = 123

        # Trả về kết quả dự báo
        return render_template('result.html', prediction=prediction, image_source = image_source)

    if __name__ == '__main__':
        app.run()

def main():
  df, X, y = read_data()
  X_transformed, preprocessor = preprocessing(X)
  results, ols_scaler = build_model_by_sm(df,X,y)
  intercept, coef = build_model_by_sklearn(X_transformed,y)
  run_app(intercept, coef, preprocessor,ols_scaler, results, df)


main()