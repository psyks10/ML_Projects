function possibleSplits = calculatePossibleSplits(feature)
    
    uniqueValues = unique(feature);
    possibleSplits = cell(length(uniqueValues),2);
    
    for i = 1:length(uniqueValues)
        
        left_split = uniqueValues{i};
        right_split = setdiff(uniqueValues, left_split);
        possibleSplits(i, 1) = {left_split};
        possibleSplits(i, 2) = {right_split};
    end

end