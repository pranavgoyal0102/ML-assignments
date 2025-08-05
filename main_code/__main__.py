import numpy as np
from PIL import Image

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
