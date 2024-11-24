import pandas as pd
import numpy as np

# Download sign language dataset from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
test = pd.read_csv( "C:/Users/jorda/.cache/kagglehub/datasets/datamunge/sign-language-mnist/versions/1/sign_mnist_test.csv" )
train = pd.read_csv( "C:/Users/jorda/.cache/kagglehub/datasets/datamunge/sign-language-mnist/versions/1/sign_mnist_train.csv" )

# Uncomment these lines to predict only a, b, c, d and e
# train = train[train['label'] < 5]
# test = test[test['label'] < 5]

# Isolate X and y
X_train = train.values[:,1:]
y_train = train.values[:,0]
X_test = test.values[:,1:]
y_test = test.values[:,0]

# Reshape data
X_train = X_train.reshape(27455, 28, 28)
X_test = X_test.reshape(7172, 28, 28)
# Uncomment these lines to predict only a, b, c, d and e
# X_train = X_train.reshape(5433, 28, 28)
# X_test = X_test.reshape(1816, 28, 28)

# Convert data to cv2-acceptable datatype
X_train = X_train.astype(np.uint8)
X_test = X_test.astype(np.uint8)

# Uncomment these lines to perform Data Augmentation by adding noisy examples
# original_len = len(X_train)
# for i in range( original_len ):
#     image = X_train[i]

#     noise = np.random.normal(2, 4, image.shape).astype(np.uint8)

#     # Add the noise to the image
#     noisy_image = cv2.add(image, noise)

#     X_train = np.append(X_train, np.array([noisy_image]), axis=0)
#     y_train = np.append(y_train, y_train[i])

print(X_train)
print(X_test)
print(y_train)
print(y_test)