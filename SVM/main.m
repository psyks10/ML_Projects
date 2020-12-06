
for type = 1:1%2
    [features, labels] = loadDataset(type);
    linearResults = linearTraining(type, features, labels);
    guassianResults = guassianTraining(type, features, labels);
end