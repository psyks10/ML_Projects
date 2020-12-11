function RMSE = calculateRMSE(actual, predicted)

    error = actual - predicted;
    squaredError = error.^2;
    meanSquaredError = mean(squaredError);
    RMSE = sqrt(meanSquaredError);
    
end