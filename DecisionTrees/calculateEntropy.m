function entropy = calculateEntropy(positiveRatio, negativeRatio)
    % high entropy = low purity 
    entropy = - positiveRatio * log2(positiveRatio) - negativeRatio * log2(negativeRatio);
end