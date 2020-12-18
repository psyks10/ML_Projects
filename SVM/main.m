typeName = ["Regression","Classification"];
for type = 0:1
    [features, labels] = loadDataset(type);
    
    linearResults = linearTraining(type, features, labels);
    writetable(struct2table(linearResults), strcat(typeName{type+1},'LinearFullResults.csv'));
    writematrix([linearResults.labels],strcat(typeName{type+1},'Linear.csv'))

    gaussianResults = gaussianTraining(type, features, labels);
    writetable(struct2table(guassianResults), strcat(typeName{type+1},'GaussianFullResults.csv'));
    writematrix([guassianResults.labels],strcat(typeName{type+1},'Gaussian.csv'))
    
    polynomialResults = polynomialTraining(type, features, labels);
    writetable(struct2table(polynomialResults), strcat(typeName{type+1},'PolynomialFullResults.csv'));
    writematrix([polynomialResults.labels],strcat(typeName{type+1},'Polynomial.csv'))
    
    if type
        writetable(struct2table([gaussianResults.metrics]), strcat(typeName{type+1},'GaussianMetrics.csv'));
        writetable(struct2table([polynomialResults.metrics]), strcat(typeName{type+1},'PolynomialMetrics.csv'));
    end
end
