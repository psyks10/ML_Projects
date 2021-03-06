function possibleSplits = calculatePossibleSplits(values)
    % Calculates all possible permutations of the values
    % of two splits with one partition being of size 1 

    possibleSplits = cell(length(values),2);
    
    for i = 1:length(values)
        
        left_split = values(i);
        right_split = setdiff(values, left_split);
        possibleSplits(i, 1) = {left_split};
        possibleSplits(i, 2) = {right_split};
    end

end