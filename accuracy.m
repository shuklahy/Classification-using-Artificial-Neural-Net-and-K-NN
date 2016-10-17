function acc = accuracy(Y_test_labels, predicted_labels)
    %Calculating the percentage accuracy
    totalCount = length(Y_test_labels);
    num_correctly_classified = 0;
    for i = 1:totalCount
        if predicted_labels(i) == Y_test_labels(i)
            num_correctly_classified = num_correctly_classified+1;
        end
    end
    acc = num_correctly_classified/totalCount * 100;
end

