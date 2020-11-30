function stdev = calculateStandardDeviation(labels)
    % high standard deviation = low purity
    if numel(unique(string(labels.label))) == 1
       stdev = 0;
       return;
    else
        stdev = std(cell2mat(labels.label));
    end
    
end