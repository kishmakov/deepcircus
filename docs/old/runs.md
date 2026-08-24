# Exact decision tree solvers for small inputs (of length 16)

## 0: Branching on every bit, don't look at target

1'378'079

## 1: Branching on lowest bit

489'231

## 2: Looking for best bit recursively

133'378

# Using simple network


x_train: (8192, 21780), y_train: (8192,)
x_test: (8192, 21780), y_test: (8192,)
train labels: [4096 4096]
test labels: [4096 4096]
Epoch  10/50  loss=0.0462  elapsed=0.40s  device=cuda
Epoch  20/50  loss=0.0269  elapsed=0.39s  device=cuda
Epoch  30/50  loss=0.0010  elapsed=0.39s  device=cuda
Epoch  40/50  loss=0.0009  elapsed=0.39s  device=cuda
Epoch  50/50  loss=0.0006  elapsed=0.39s  device=cuda
detector_test_accuracy: 0.7341
baseline_test_accuracy: 0.5000
              precision    recall  f1-score   support

           0       0.76      0.69      0.72      4096
           1       0.71      0.78      0.75      4096

    accuracy                           0.73      8192
   macro avg       0.74      0.73      0.73      8192
weighted avg       0.74      0.73      0.73      819