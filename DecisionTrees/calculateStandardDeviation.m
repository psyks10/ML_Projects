function stdev = calculateStandardDeviation(labels)
    % high standard deviation = low purity
    stdev = std(cell2mat(labels.label));
end