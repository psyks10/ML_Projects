function RMSE = calculateRMSE(error)
    squaredError = error.^2;
    meanSquaredError = mean(squaredError);
    RMSE = sqrt(meanSquaredError);
end