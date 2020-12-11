function entropy = calculateEntropy(positiveRatio, negativeRatio)
    % high entropy = low purity 
    if positiveRatio == 1 || negativeRatio == 1
        entropy = 0;
    else
        entropy = - positiveRatio * log2(positiveRatio) - negativeRatio * log2(negativeRatio);
    end
end