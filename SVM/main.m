typeName = ["Regression","Classification"];
for type = 1
    [features, labels] = loadDataset(type);
    
    linearResults = linearTraining(type, features, labels);
    writetable(struct2table(linearResults), strcat(typeName{type+1},'LinearFullResults.csv'));

    gaussianResults = gaussianTraining(type, features, labels);
    writetable(struct2table(guassianResults), strcat(typeName{type+1},'GaussianFullResults.csv'));
    writematrix([guassianResults.labels],strcat(typeName{type+1},'Gaussian.csv'))
    
    polynomialResults = polynomialTraining(type, features, labels);
    writetable(struct2table(polynomialResults), strcat(typeName{type+1},'PolynomialFullResults.csv'));
    writematrix([polynomialResults.labels],strcat(typeName{type+1},'Polynomial.csv'))
    
    if type
        writetable(struct2table([linearResults.Metrics]), strcat(typeName{type+1},'LinearMetrics.csv'));
        writetable(struct2table([gaussianResults.metrics]), strcat(typeName{type+1},'GaussianMetrics.csv'));
        writetable(struct2table([polynomialResults.metrics]), strcat(typeName{type+1},'PolynomialMetrics.csv'));
    end
end
