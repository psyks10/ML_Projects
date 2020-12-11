function RMSE = calculateRMSE(actual, predicted)

    error = cell2mat(actual) - cell2mat(predicted);
    squaredError = error.^2;
    meanSquaredError = mean(squaredError);
    RMSE = sqrt(meanSquaredError);
    
end