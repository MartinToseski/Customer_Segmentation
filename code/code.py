import numpy as np
import matplotlib.pyplot as plt
import csv
from google.colab import drive

# Database Link -> https://www.kaggle.com/datasets/vetrirah/customer
# Data Path (Google Drive)
# !!! FIX DATA PATH TO WORK FROM A FILE IN THE SAME FOLDER INSTEAD OF THROUGH GOOGLE DRIVE !!!
data_path = "/content/drive/My Drive/Customer_Segmentation/data/"

rNum = 0 #number of rows
cNum = 0 #number of columns
num_components = 3 #reducing the dataset dimensions to this many dimensions (PCA)
trainingData = [] #storing the data
tolerance = 1e-6

# Read data from dataset
# The dataset is in the same folder as the .py file
def ReadData():
    global rNum
    global cNum
    global trainingData

    with open(f"{data_path}Train.csv", "r") as Xfile:
        reader = csv.reader(Xfile)

        for row in reader:
            trainingData.append(row)
            rNum += 1

        cNum = len(trainingData[0])



# Get list of values written in the i-th column (without duplicates)
def GetValues(trainingData, i):
    ls = []
    index = i
    for i in range(1, len(trainingData)):
        find_list = False
        for j in range(len(ls)):
            if trainingData[i][index] == ls[j]:
                find_list = True
        if not find_list: #don't store duplicates
            ls.append(trainingData[i][index])
    return ls



# Replace all values with numbers
# the numbers are centered around 0
def ReplaceWordsWithNumbers(trainingData, index):
    values = GetValues(trainingData, index) #get all unique values written in the data
    if (len(values) == 2):
        cnt = 0
    else:
        cnt = -int(len(values)/2)

    for i in range(len(values)):
        for j in range(1, len(trainingData)):
            if trainingData[j][index] == values[i]:
                trainingData[j][index] = cnt
        cnt+=1

    return trainingData



# Replace non-numeric values with numbers (to use in PCA)
def ReplaceNonNumeric(trainingData):
    for i in range(len(trainingData[0])):
        try:
            #try converting to int
            int(trainingData[1][i])
            continue
        except ValueError:
            try:
                #try converting to float
                float(trainingData[1][i])
                continue
            except ValueError:
                trainingData = ReplaceWordsWithNumbers(trainingData, i) #if the value is not a number replace it with a number
    return trainingData



# Remove rows with missing values except for the profession column - empty fields are regarded as not employed
def RemoveInvalidRows(trainingData):
    profession_index = 0
    for i in range(len(trainingData[0])):
        if (trainingData[0][i] == "Profession"):
            profession_index = i #find the profession column
    trainingData = [row for row in trainingData if all(value != '' for idx, value in enumerate(row) if idx != profession_index)]
    return trainingData



# Remove unnecessary columns - ID, Var_1, Segmentation)
def RemoveColumns(trainingData):
    return [row[1:-2] for row in trainingData]



# Convert numbers in string format to float/int
def GetStringNumbers(trainingData):
    for i in range(1, len(trainingData)):
        for j in range(len(trainingData[0])):
            if '.' in str(trainingData[i][j]): #all values are (replaced with)
                                               #numbers so if the string contains "." it's a float value
                trainingData[i][j] = float(trainingData[i][j])
            else: #else it's an integer
                trainingData[i][j] = int(trainingData[i][j])
    return trainingData



# Scale each value in the column with index "index" to the interval [0, 1]
# for universal comparison of the principal components and avoiding false data in the eigenvectors
def ScaleNumbers(trainingData, index):
    values = GetValues(trainingData, index)
    if (len(values) == 2):
        cnt = 0
    else:
        cnt = -int(len(values)/2)

    min = cnt
    max = min+len(values)-1

    for i in range(len(values)):
        for j in range(1, len(trainingData)):
            if trainingData[j][index] == values[i]:
                #scale values to [0, 1]
                #x_scaled = (x-minx)/(xmax-xmin)
                trainingData[j][index] = (cnt-min)/(max-min)
        cnt+=1

    return trainingData



# Scale the values for each of the columns
def ScaleColumnValues(trainingData):
    for i in range(len(trainingData[0])):
        trainingData = ScaleNumbers(trainingData, i)
    return trainingData



# Clean the Dataset - remove not needed columns and invalid rows (rows with empty fields),
# replace words with number values, convert numbers in a string format to numbers,
# scale all data to a single interval - all to ensure a correct analysis using PCA
def CleanData(trainingData):
    trainingData = RemoveColumns(trainingData)
    trainingData = RemoveInvalidRows(trainingData)
    trainingData = ReplaceNonNumeric(trainingData)
    trainingData = GetStringNumbers(trainingData)
    trainingData = ScaleColumnValues(trainingData)
    return trainingData



# Get a transposed matrix of the given one
def TransposeMatrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]



# Multiply matrix A by matrix B from the right
def MatrixMultiply(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]



# Get the module of a vector (sqrt(x^2 + y^2 + z^2 + ...))
def VectorNorm(v):
    return sum(x**2 for x in v) ** 0.5



# Normalize the vector by diving by its length (unit vector)
def NormalizeVector(v):
    norm = VectorNorm(v)
    return [x / norm for x in v]



# Multiply matrix by a vector
def MatrixVectorMultiply(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(len(matrix[i]))) for i in range(len(matrix))]



# Get the QRDecomposition of a matrix
# where Q is an orthogonal matrix (Qt = Q^(-1)) 
# and A is an upper triangular matrix
def QRDecomposition(matrix):
    n = len(matrix)
    m = len(matrix[0])

    Q = [[0] * n for _ in range(n)]
    R = [[0] * n for _ in range(n)]

    for j in range(n):
        v = [matrix[i][j] for i in range(n)]

        # Orthogonalize against all previous columns of Q
        for i in range(j):
            R[i][j] = sum(Q[k][i] * v[k] for k in range(n))
            v = [v[k] - R[i][j] * Q[k][i] for k in range(n)]

        # Normalize the vector v to get the j-th column of Q
        R[j][j] = VectorNorm(v)
        for i in range(n):
            Q[i][j] = v[i] / R[j][j]

    return Q, R



# Get all eigenvectors and eigenvalues for the (square) matrix
# A_k+1 = R_k*Q_k = Q_k ^ (-1) * Q_k * R_k * Q_k = Q_k ^ (-1) * A_k * Q_k = Q_K ^ T * A_k * Q_k
# all the Ak are similar and hence they have the same eigenvalues
# The algorithm is numerically stable because it proceeds by orthogonal similarity transforms
# Under certain conditions,[4] the matrices Ak converge to a triangular matrix, the Schur form of A
# The eigenvalues of a triangular matrix are listed on the diagonal, and the eigenvalue problem is solved
# In testing for convergence it is impractical to require exact zeros
def QRAlgorithm(matrix, tolerance):
    n = len(matrix)
    A = matrix

    Q_total = [[1 if i == j else 0 for j in range(n)] for i in range(n)] #identity matrix for eigenvectors

    prev_A = A
    while True:
        Q, R = QRDecomposition(A)

        # Update A for the next iteration
        A_next = MatrixMultiply(R, Q)

        # Accumulate Q to get the eigenvectors
        Q_total = MatrixMultiply(Q_total, Q)

        # Check if A has become sufficiently diagonal
        breakCondition = True
        for i in range(n):
            for j in range(n):
                if i != j and abs(A_next[i][j]) > tolerance:
                    breakCondition = False
                    break
        if breakCondition:
            break

        A = A_next

    # The eigenvalues are the diagonal elements of A (converged matrix)
    eigenvalues = [A[i][i] for i in range(n)]

    # The eigenvectors are the columns of Q_total
    eigenvectors = [list(col) for col in zip(*Q_total)]  # Transpose Q_total to get eigenvectors

    return eigenvalues, eigenvectors



#Reduce the dimensions to the given number of components
# 1. Get the mean values 
# 2. Calculate the covariance matrix
# 3. Get the eigenvalues and eigenvectors of the covariance matrix
# 4. Sort them by their eigenvalues in descending order and take the first K
def PCA(trainingData, num_components, tolerance):
    #get mean values for all parameters across x axis
    mean_values = [0 for i in range(len(trainingData[0]))]

    print("-------------------------------- PCA --------------------------------")

    for row in trainingData:
        for i in range(len(row)):
            mean_values[i] += row[i]

    for i in range(len(mean_values)):
        mean_values[i] /= len(trainingData)

    print("Mean values")
    print(mean_values)
    print()

    #find the covariance matrix
    #covariance between two points
    #cov(X, Y) = E[XY] - E[X]E[Y]
    covariance_matrix = []
    for i in range(len(trainingData[0])):
        covariance_matrix.append([])
        for j in range(len(trainingData[0])):
            Ex = mean_values[i]
            Ey = mean_values[j]
            Exy = 0

            for k in range(len(trainingData)):
                Exy += trainingData[k][i] * trainingData[k][j]

            Exy /= len(trainingData)
            covariance_matrix[i].append(Exy - Ex * Ey)

    print("Covariance matrix")
    for row in covariance_matrix:
        print(row)
    print()

    #find the eigenvectors and corresponding eigenvalues, sort them and get the first k(=num_components) largest
    #vectors such that xA = vA -> (x-v)A = 0 -> (A-xI)v = 0
    #the sum of the eigenvalues is 1
    print("Eigenvectors and eigenvalues")
    eigenvalues, eigenvectors = QRAlgorithm(covariance_matrix, tolerance)

    eigen_pairs = []
    for i in range(len(eigenvalues)):
        eigen_pairs.append([eigenvalues[i], eigenvectors[i]])

    eigen_pairs.sort(reverse=True)

    sorted_eigenvalues = []
    sorted_eigenvectors = []

    for i in range(len(eigen_pairs)):
        sorted_eigenvalues.append(eigen_pairs[i][0])
        sorted_eigenvectors.append(eigen_pairs[i][1])

    for i in range(len(eigenvectors)):
        print(f"Eigenvector {i+1} ", sorted_eigenvectors[i])
        print(f"Eigenvalue {i+1} ", sorted_eigenvalues[i])
        print()

    varianceSum = sum(sorted_eigenvalues[i] for i in range(num_components))
    print(f"The first {num_components} components account for {varianceSum*100}% of the data")
    print()

    return sorted_eigenvectors[:num_components], sorted_eigenvalues[:num_components]



# Project the initial data points onto the principal components acquired through PCA
def ProjectData(trainingData):
    sorted_eigenvectors, sorted_eigenvalues = PCA(trainingData, num_components, tolerance)
    return MatrixMultiply(trainingData, TransposeMatrix(sorted_eigenvectors))



# Visualize reduced dimensions
def PlotData(projectedData, labels = None):
    projectedData = np.array(projectedData)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    if labels is not None:
        scatter = ax.scatter(projectedData[:, 0], projectedData[:, 1], projectedData[:, 2], c=labels, cmap='viridis', s=50)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Labels', rotation=270, labelpad=15)
    else:
        ax.scatter(projectedData[:, 0], projectedData[:, 1], projectedData[:, 2], color='blue', s=50)

    # Labels and title
    ax.set_title('3D PCA Visualization', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)

    plt.show()



#Calculating distance without the square root for accracy measures and less computational burden
def SquaredDistance(point1, point2):
  return (sum((point1[i]-point2[i])**2) for i in range(len(point1)))



# Divides the cluster into 2 subclusters based on the distance from each data point to points
# C1 - furthest point from the centroid in the cluster (P); C2 - 2P - C1
def LimitedIterationsTwoMeans(cluster, iter):
    pass



# K-Means Clustering
# Instead of the classic K-Means Clustering
# Each iteration the largest cluster is divided into 2 clusters, until the required number of clusters is achieved
# In this case three iterations are used for two-means
def BisectingKMeansClustering(trainingData, K):
    clusters = []
    clusters.append(trainingData)
    clusters_num = 1

    while (clusters_num < K):
        max_size = len(clusters[0])
        max_index = i
        for i in range(clusters_num):
            if (len(clusters_num[i]) > max_size):
                max_size = len(clusters[i])
                max_index = i

        largest_cluster = clusters[max_index]
        c1, c2 = LimitedIterationsTwoMeans(largest_cluster)
        clusters[max_index] = c1
        clusters.append(c2)
        clusters_num += 1

    # 2 more classic iterations with the obtained K clusters
    return



# Testing



# Visualize the results



# Main
ReadData() #number of rows, columns, data titles, training data
print("Number of rows in dataset:")
print(rNum)
print("Number of columns (parameters) in dataset:")
print(cNum)
print("Dataset:")
print(trainingData)
print()

cleanedData = CleanData(trainingData)
cleanedTitles = cleanedData[0]

print("Cleaned data titles:")
print(cleanedTitles)
print()

cleanedData = cleanedData[1:]
print("Cleaned data:")
print(cleanedData)
print()
print("Cleaned data dimensions:")
print(len(cleanedData), "x", len(cleanedData[0]))
print()

projectedData = ProjectData(cleanedData)
print()
PlotData(projectedData)