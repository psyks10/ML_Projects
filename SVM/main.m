typeName = ["Regression","Classification"];
for type = 1:1
    [features, labels] = loadDataset(type);
%     
%     linearResults = linearTraining(type, features, labels);
%     labellist = [linearResults.labels];
%     writematrix(labellist, strcat(typeName{type+1}, 'linearSVM.csv'));
%     
%     guassianResults = gaussianTraining(type, features, labels);
%     labellist = [guassianResults.labels];
%     writematrix(labellist, strcat(typeName{type+1},'gaussianSVM.csv'));
   
    polynomialResults = polynomialTraining(type, features, labels);
    labellist = [guassianResults.labels];
    writematrix(labellist, strcat(typeName{type+1},'polynomialSVM.csv'));

end