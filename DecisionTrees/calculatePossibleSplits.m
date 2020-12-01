function possibleSplits = calculatePossibleSplits(values)
    
    possibleSplits = cell(length(uniqueValues),2);
    
    for i = 1:length(values)
        
        left_split = values(i);
        right_split = setdiff(values, left_split);
        possibleSplits(i, 1) = {left_split};
        possibleSplits(i, 2) = {right_split};
    end

end