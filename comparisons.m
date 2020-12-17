
types=["Classification/","Regression"];

% need to seperate result csvs into Classification and Regression folder
% need to rename file names

for type=types
    filenames = ["ANN","Gaussian","Polynomial","DecisionTree"];
    results = struct();
    comparison = struct("fdist",NaN, "sdist",NaN,"h",NaN,"p",NaN);

    for i = 1:length(filenames)
        cell = readcell(strcat(filename{i}, ".csv"));
        results(i).(filename{i}) = cell;
    end

    combos = combntns(1:length(filenames),2); combos
    j=1;
    
    for combo=combos
        fdist = filenames{combo{1}};
        sdist = filenames{combo{2}};
        if ismember(1,combo)
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

