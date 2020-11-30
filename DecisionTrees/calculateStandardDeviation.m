function stdev = calculateStandardDeviation(labels)
    % high standard deviation = low purity
    stdev = std(labels.label);
end