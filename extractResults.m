
%test = struct();
predictions = {};

for fold=1:1:10
    predictions = [predictions cell2mat(trainEvaluationClass(fold).labels)];
    %test(fold) = cell2mat(trainEvaluationClass(1).labels).';
end


