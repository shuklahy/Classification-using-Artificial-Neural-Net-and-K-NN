
%% Add VidTIMIT to matlab path
path(path, strcat(pwd,'\VidTIMIT'));

%% Load Data
X_train_filename = strcat(pwd,'\VidTIMIT\X_train');
Y_train_filename = strcat(pwd,'\VidTIMIT\y_train');
X_test_filename  = strcat(pwd,'\VidTIMIT\X_test');
Y_test_filename  = strcat(pwd,'\VidTIMIT\y_test');

importfile(X_train_filename)
importfile(Y_train_filename)
importfile(X_test_filename)
importfile(Y_test_filename)

%% perform KNN classification with k = 7
knn_Model = fitcknn(X_train,y_train,'NumNeighbors',7);
disp(knn_Model);
predicted_labels = predict(knn_Model,X_test);
acc = accuracy(y_test, predicted_labels);

disp('Accuracy of Problem2a using KNN | k = 7 is ');
disp(acc);
