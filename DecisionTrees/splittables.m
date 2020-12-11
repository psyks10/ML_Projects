featurs,labels = loadDataset(1);

edges=[1:ceil(height(table)/10):height(table),height(table)];
discretize(1:height(table), edges).'
splitTables = splitapply(@(x) {cell2table(x, 'VariableNames', table.Properties.VariableNames)}, table2cell(table), discretize(1:height(table), edges).');
