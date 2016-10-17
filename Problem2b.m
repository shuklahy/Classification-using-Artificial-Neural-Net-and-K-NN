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

%% Train using feedforwardnet function with hidden layer containing 25 neurons
net = feedforwardnet(25);

% X_train -> features X samples size array
% Y_train -> class    X samples size array
%net.trainFcn = 'trainbr';
net.trainParam.max_fail = 15;
y_train = full(ind2vec(y_train));
[net,tr] = train(net,X_train',y_train);

%disp(tr)
view(net);
predicted_y = net(X_test');
%perf = perform(net,predicted_y, y_test);
%disp('Performance =');
%disp(perf);

output = zeros(1,1000);
for i = 1:1000
   [val, maxIndex] = max(predicted_y(:,i));
   output(1,i) = maxIndex;
end

%predicted_labels = round(predicted_y);
acc = accuracy(y_test, output);

disp('Accuracy of Problem2a using ANN with 25 neurons is ');
disp(acc);