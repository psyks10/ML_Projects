
for type = 0:1
    [features, labels] = loadDataset(type);
    linearResults = linearTraining(type, features, labels);
    guassianResults = gaussianTraining(type, features, labels);
    polynomialResults = polynomialTraining(type, features, labels);
end