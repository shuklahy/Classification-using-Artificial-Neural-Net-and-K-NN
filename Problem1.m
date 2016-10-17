%% Add VidTIMIT to matlab path
path(path, strcat(pwd,'\Human Activity Recognition'));

%% Load Data
X_train_filename = strcat(pwd,'\Human Activity Recognition\X_train.txt');
Y_train_filename = strcat(pwd,'\Human Activity Recognition\y_train.txt');
X_test_filename  = strcat(pwd,'\Human Activity Recognition\X_test.txt');
Y_test_filename  = strcat(pwd,'\Human Activity Recognition\y_test.txt');

X_training_table = readtable(X_train_filename);
X_testing_table = readtable(X_test_filename);
Y_train_labels = importdata(Y_train_filename);
Y_test_labels = importdata(Y_test_filename);

%% perform KNN classification with k = 7
knn_Model = fitcknn(X_training_table,Y_train_labels,'NumNeighbors',7);
disp(knn_Model);
predicted_labels = predict(knn_Model,X_testing_table);
acc = accuracy(Y_test_labels, predicted_labels);

disp('Accuracy of Problem1 using KNN | k = 7 is ');
disp(acc);



