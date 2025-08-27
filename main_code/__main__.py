import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time



# Q1:

#a
arr = np.array([1, 2, 3, 6, 4, 5])
print("reversed:", arr[::-1])

#b
array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
print("using flatten():", array1.flatten())
print("using ravel():", np.ravel(array1))

#c
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
print("are equal:", np.array_equal(arr1, arr2))

#d
x = np.array([1,2,3,4,5,1,2,1,1,1])
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
most_freq_x = np.bincount(x).argmax()
most_freq_y = np.bincount(y).argmax()
print("Most frequent in x:", most_freq_x, ", Indices:", np.where(x == most_freq_x)[0])
print("Most frequent in y:", most_freq_y, ", Indices:", np.where(y == most_freq_y)[0])

#e
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print("sum:", gfg.sum())
print("row sum:", gfg.sum(axis=1))
print("column sum:", gfg.sum(axis=0))

#f
n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])
print("diagonal sum:", np.trace(n_array))
print("eigenvalues:", np.linalg.eigvals(n_array))
vals, vecs = np.linalg.eig(n_array)
print("eigenvectors:\n", vecs)
print("inverse:\n", np.linalg.inv(n_array))
print("determinant:", np.linalg.det(n_array))

#g
p1 = np.array([[1, 2], [2, 3]])
q1 = np.array([[4, 5], [6, 7]])
print("matrix prod 1:\n", np.dot(p1, q1))
print("covar of prod 1:", np.cov(np.dot(p1, q1)))

p2 = np.array([[1, 2], [2, 3], [4, 5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])
print("matrix prod 2:\n", np.dot(p2, q2))
print("covar of prod 2:", np.cov(np.dot(p2, q2)))

#h
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])
print("inner prod:", np.inner(x[0], y[0]))
print("outer prod:\n", np.outer(x, y))
cartesian = np.array(np.meshgrid(x.flatten(), y.flatten())).T.reshape(-1, 2)
print("cart prod:\n", cartesian)








# Q2:
array = np.array([[1, -2, 3], [-4, 5, -6]])
print("abs:", np.abs(array))
flat = array.flatten()
print("percentiles (flat):", np.percentile(flat, [25, 50, 75]))
print("oercentiles (cols):", np.percentile(array, [25, 50, 75], axis=0))
print("percentiles (rows):", np.percentile(array, [25, 50, 75], axis=1))
print("mean:", flat.mean(), "Median:", np.median(flat), "std dev:", flat.std())
print("mean (cols):", array.mean(axis=0), "median (cols):", np.median(array, axis=0))
print("mean (rows):", array.mean(axis=1), "median (rows):", np.median(array, axis=1))

#b
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
print("floor:", np.floor(a))
print("ceiling:", np.ceil(a))
print("truncated:", np.trunc(a))
print("rounded:", np.round(a))






# Q3:
array = np.array([10, 52, 62, 16, 16, 54, 453])
print("sorted:", np.sort(array))
print("sorted indices:", np.argsort(array))
print("4 smallest:", np.sort(array)[:4])
print("5 largest:", np.sort(array)[-5:])

array2 = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
print("integer elements:", array2[array2 == array2.astype(int)])
print("float elements:", array2[array2 != array2.astype(int)])






# Q4: Image Processing

def img_to_array(path):
    img = Image.open(path)
    arr = np.array(img)
    if len(arr.shape) == 2:
        np.savetxt("gray_image.txt", arr, fmt='%d')
    else:
        np.savetxt("rgb_image.txt", arr.reshape(-1, arr.shape[2]), fmt='%d')
    return arr


## ---------- Assignment 2 ------    ##

# Part 1
df = pd.read_csv("AWCustomers.csv")

selected_features = ['CustomerID', 'Title', 'FirstName', 'MiddleName', 'LastName',
                     'Suffix', 'AddressLine1', 'AddressLine2', 'City', 'StateProvinceName',
                     'CountryRegionName', 'PostalCode', 'PhoneNumber', 'BirthDate', 'Education',
                     'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NumberCarsOwned',
                     'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'LastUpdated']

df_selected = df[selected_features]

# Part 2

df_selected['YearlyIncome'] = df_selected['YearlyIncome'].fillna(df_selected['YearlyIncome'].mean())
df_selected['Education'] = df_selected['Education'].fillna(df_selected['Education'].mode()[0])
df_selected['NumberChildrenAtHome'] = df_selected['NumberChildrenAtHome'].fillna(df_selected['NumberChildrenAtHome'].median())
df_selected['NumberCarsOwned'] = df_selected['NumberCarsOwned'].fillna(df_selected['NumberCarsOwned'].median())
df_selected['TotalChildren'] = df_selected['TotalChildren'].fillna(df_selected['TotalChildren'].median())

df_selected['BirthDate'] = pd.to_datetime(df_selected['BirthDate'])
df_selected['Age'] = (pd.Timestamp.today() - df_selected['BirthDate']).dt.days // 365

numeric_features = ['Age', 'YearlyIncome', 'NumberChildrenAtHome', 'NumberCarsOwned', 'TotalChildren']
scaler = MinMaxScaler()
df_selected[numeric_features] = scaler.fit_transform(df_selected[numeric_features])

df_selected['Age_binned'] = pd.cut(df_selected['Age'], bins=5, labels=False)
df_selected['Income_binned'] = pd.qcut(df_selected['YearlyIncome'], 5, labels=False)

df_final = pd.get_dummies(df_selected, columns=['Gender', 'MaritalStatus', 'StateProvinceName'], drop_first=True)

print(df_final.shape)


# Part 3

obj1 = df_final.iloc[0]
obj2 = df_final.iloc[1]

numeric_cols = df_final.select_dtypes(include=['float64','int64','uint8']).columns
obj1_num = obj1[numeric_cols].values
obj2_num = obj2[numeric_cols].values

binary_features = df_final.select_dtypes(include=['uint8']).columns
obj1_bin = obj1[binary_features].values
obj2_bin = obj2[binary_features].values

if len(obj1_bin) > 0:
    simple_match = np.sum(obj1_bin == obj2_bin) / len(obj1_bin)
else:
    simple_match = np.nan
if np.sum(obj1_bin + obj2_bin) == 0:
    jaccard = 0.0
else:
    jaccard = jaccard_score(obj1_bin, obj2_bin, zero_division=0)

cos_sim = cosine_similarity([obj1_num], [obj2_num])[0][0]

print("Simple Matching Similarity:", simple_match)
print("Jaccard Similarity:", jaccard)
print("Cosine Similarity:", cos_sim)


if 'CommuteDistance' in df_final.columns:
    distance_mapping = {
        'Less than 1 Mile': 1,
        '1-2 Miles': 2,
        '2-5 Miles': 3,
        '5-10 Miles': 4,
        '10+ Miles': 5
    }
    df_final['CommuteDistance_num'] = df_final['CommuteDistance'].map(distance_mapping)
    correlation = df_final['CommuteDistance_num'].corr(df_final['YearlyIncome'])
    print("Correlation between CommuteDistance and YearlyIncome:", correlation)
else:
    print("Column 'CommuteDistance' not found in dataset")




## --------------  Assignment 3 --------------------- ###


def load_house_data():
    df = pd.read_csv('USA_Housing.csv')
    print("House Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def k_fold_cross_validation(X, y, k=5):
    n = len(X)
    fold_size = n // k
    best_beta = None
    best_r2 = -np.inf
    results = []

    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else n

        test_indices = list(range(start_idx, end_idx))
        train_indices = list(range(0, start_idx)) + list(range(end_idx, n))

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        X_train_with_intercept = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])

        beta = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept) @ X_train_with_intercept.T @ y_train
        y_pred = X_test_with_intercept @ beta
        r2 = r2_score(y_test, y_pred)

        results.append({
            'fold': i + 1,
            'beta': beta,
            'r2_score': r2
        })

        print(f"Fold {i+1}: R² = {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_beta = beta

    return results, best_beta, best_r2

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    m, n = X_with_intercept.shape
    beta = np.zeros(n)

    for i in range(iterations):
        y_pred = X_with_intercept @ beta
        cost = np.mean((y_pred - y) ** 2)
        gradient = (2 / m) * X_with_intercept.T @ (y_pred - y)
        beta -= learning_rate * gradient

    return beta

def question_1():
    print("=== QUESTION 1: K-FOLD CROSS VALIDATION ===")
    df = load_house_data()

    X = df.drop('Price', axis=1)
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    results, best_beta, best_r2 = k_fold_cross_validation(X_scaled, y)
    print(f"\nBest R² Score: {best_r2:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    X_train_with_intercept = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])

    y_pred_test = X_test_with_intercept @ best_beta
    final_r2 = r2_score(y_test, y_pred_test)

    print(f"Final Test R² Score: {final_r2:.4f}")
    return results, best_beta

def question_2():
    print("\n=== QUESTION 2: GRADIENT DESCENT WITH VALIDATION SET ===")
    df = load_house_data()

    X = df.drop('Price', axis=1)
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    learning_rates = [0.001, 0.01, 0.1, 1]
    best_lr = None
    best_val_r2 = -np.inf
    best_coefficients = None

    for lr in learning_rates:
        coefficients = gradient_descent(X_train, y_train, lr, 1000)

        X_val_with_intercept = np.column_stack([np.ones(len(X_val)), X_val])
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])

        y_val_pred = X_val_with_intercept @ coefficients
        y_test_pred = X_test_with_intercept @ coefficients

        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Learning Rate: {lr}, Validation R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_lr = lr
            best_coefficients = coefficients

    print(f"\nBest Learning Rate: {best_lr}")
    print(f"Best Validation R²: {best_val_r2:.4f}")

    return best_coefficients, best_lr

def load_car_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    columns = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
               "num_doors", "body_style", "drive_wheels", "engine_location",
               "wheel_base", "length", "width", "height", "curb_weight",
               "engine_type", "num_cylinders", "engine_size", "fuel_system",
               "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
               "city_mpg", "highway_mpg", "price"]

    df = pd.read_csv(url, names=columns, na_values='?')
    print("Car Dataset Shape:", df.shape)
    return df

def preprocess_car_data(df):
    df = df.dropna(subset=['price'])

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in numeric_columns:
        if col != 'price':
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    door_mapping = {'two': 2, 'four': 4}
    cylinder_mapping = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}

    df['num_doors'] = df['num_doors'].map(door_mapping)
    df['num_cylinders'] = df['num_cylinders'].map(cylinder_mapping)

    df = pd.get_dummies(df, columns=['body_style', 'drive_wheels'], prefix=['body_style', 'drive_wheels'])

    le = LabelEncoder()
    for col in ['make', 'aspiration', 'engine_location', 'fuel_type']:
        df[col] = le.fit_transform(df[col])

    df['fuel_system'] = df['fuel_system'].apply(lambda x: 1 if 'pfi' in str(x).lower() else 0)
    df['engine_type'] = df['engine_type'].apply(lambda x: 1 if 'ohc' in str(x).lower() else 0)

    return df

def question_3():
    print("\n=== QUESTION 3: CAR PRICE PREDICTION WITH PREPROCESSING ===")
    df = load_car_data()
    df_processed = preprocess_car_data(df)

    print(f"Processed Dataset Shape: {df_processed.shape}")

    X = df_processed.drop('price', axis=1)
    y = df_processed['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_original = r2_score(y_test, y_pred)

    print(f"Original Model R² Score: {r2_original:.4f}")

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA Components: {pca.n_components_} (from {X.shape[1]} original features)")

    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train)
    y_pred_pca = model_pca.predict(X_test_pca)
    r2_pca = r2_score(y_test, y_pred_pca)

    print(f"PCA Model R² Score: {r2_pca:.4f}")

    if r2_pca > r2_original:
        print("PCA improved performance!")
    elif r2_pca < r2_original:
        print("PCA reduced performance.")
    else:
        print("PCA had no significant impact on performance.")

    return r2_original, r2_pca, pca.n_components_

def main():
    try:
        results_q1, best_beta = question_1()
        best_coeffs, best_lr = question_2()
        r2_orig, r2_pca, n_components = question_3()

        print("\n=== SUMMARY ===")
        print(f"Q1 - Best K-Fold R²: {max([r['r2_score'] for r in results_q1]):.4f}")
        print(f"Q2 - Best Learning Rate: {best_lr}")
        print(f"Q3 - Original R²: {r2_orig:.4f}, PCA R²: {r2_pca:.4f}")
        print(f"Q3 - Dimensionality Reduction: {n_components} components")

    except FileNotFoundError as e:
        print(f"Error: Could not find dataset file. {e}")
        print("Make sure 'USA_Housing.csv' is in your project directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

main()




## -------------------------- Assignment 4 ---------------------------------- ####


q1

base_url = "https://books.toscrape.com/catalogue/page-{}.html"
books = []

for page in range(1, 51):
    url = base_url.format(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for book in soup.find_all('article', class_='product_pod'):
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').text.strip()
        availability = book.find('p', class_='instock availability').text.strip()
        star_rating = book.p['class'][1]

        books.append([title, price, availability, star_rating])

df_books = pd.DataFrame(books, columns=['Title', 'Price', 'Availability', 'Star Rating'])
df_books.to_csv("books.csv", index=False)
print("books.csv saved")


# q2

url = "https://www.imdb.com/chart/top/"
driver = webdriver.Chrome()
driver.get(url)
time.sleep(3)

movies = []
rows = driver.find_elements(By.XPATH, '//tbody[@class="lister-list"]/tr')

for row in rows:
    rank = row.find_element(By.XPATH, './/td[@class="titleColumn"]').text.split('.')[0]
    title = row.find_element(By.XPATH, './/td[@class="titleColumn"]/a').text
    year = row.find_element(By.XPATH, './/td[@class="titleColumn"]/span').text.strip("()")
    rating = row.find_element(By.XPATH, './/td[@class="ratingColumn imdbRating"]/strong').text
    movies.append([rank, title, year, rating])

driver.quit()

df_imdb = pd.DataFrame(movies, columns=['Rank', 'Title', 'Year', 'Rating'])
df_imdb.to_csv("imdb_top250.csv", index=False)
print("imdb_top250.csv saved")



# q3

url = "https://www.timeanddate.com/weather/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

cities_data = []
table = soup.find("table", class_="zebra")

if table:
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 3:
            city = cols[0].text.strip()
            temp = cols[1].text.strip()
            condition = cols[2].text.strip()
            cities_data.append([city, temp, condition])

    df_weather = pd.DataFrame(cities_data, columns=["City", "Temperature", "Condition"])
    df_weather.to_csv("weather.csv", index=False)
    print("weather.csv saved with", len(df_weather), "rows")
else:
    print("Weather table not found! Verify the table structure and class name.")
