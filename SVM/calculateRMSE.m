function RMSE = calculateRMSE(actual, predicted)

    error = actual - predicted;
    idx = isnan(error);
    [delRows,~] = find(idx);
    error(delRows,:) = [];
    squaredError = error.^2;
    meanSquaredError = mean(squaredError);
    RMSE = sqrt(meanSquaredError);
    
end