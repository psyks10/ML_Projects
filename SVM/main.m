typeName = ["Regression","Classification"];
for type = 0
    [features, labels] = loadDataset(type);
    
%     linearResults = linearTraining(type, features, labels);
%     writetable(struct2table(linearResults), strcat(typeName{type+1},'LinearFullResults.csv'));
%     
    guassianResults = gaussianTraining(type, features, labels);
    writetable(struct2table(guassianResults), strcat(typeName{type+1},'GuassianFullResults.csv'));
   
%     polynomialResults = polynomialTraining(type, features, labels);
%     writetable(struct2table(polynomialResults), strcat(typeName{type+1},'PolynomialFullResults.csv'));
end