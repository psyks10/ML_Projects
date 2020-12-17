
types=["Classification/"];

% need to seperate result csvs into Classification and Regression folder
% need to rename file names

for type=types
    filenames = ["ANN","Gaussian","Polynomial","DecisionTree"];
    results = struct();
    comparison = struct("fdist",NaN, "sdist",NaN,"h",NaN,"p",NaN);

    for i = 1:length(filenames)
        iCell = readmatrix(strcat(type, filenames{i}, ".csv"));
        results.(filenames{i}) = iCell;
    end
    count = 0;
    combos = cell(2,6);
    for i = 1:length(filenames)
        for j = i+1:length(filenames)
            count=count+1;
            combos{1, count} = i;
            combos{2, count} = j;
        end
    end
    j=1;
    
    for combo=combos
        fdist = filenames{combo{1}};
        sdist = filenames{combo{2}};
        if (combo{1}==1 || combo{2}==1)
            [h,p] = ttest(results.(fdist), results.(sdist));
        else
            [h,p] = ttest2(results.(fdist), results.(sdist));
        end

        comparison(j).fdist = fdist;
        comparison(j).sdist = sdist;
        comparison(j).h = h;
        comparison(j).p = p;
        j=j+1;
    end

    writetable(struct2table(comparison), strcat(type,"Comparison.csv"));
end

