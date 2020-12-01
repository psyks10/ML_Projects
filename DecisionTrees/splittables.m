table = % smthing here
edges=[1:ceil(height(table)/10):height(table),height(table)]
discretize(1:height(table), edges).'
splitTables = splitapply(@(x) {cell2table(x)}, table2cell(table), discretize(1:height(table), edges).');
