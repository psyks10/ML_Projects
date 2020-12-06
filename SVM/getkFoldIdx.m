function kFoldIdx = getkFoldIdx(noOftargets)
    shuffledIndices = randperm(noOftargets);
    edges=[1:ceil(noOftargets/10):noOftargets,noOftargets];
    discretize(1:noOftargets, edges);
    kFoldIdx = splitapply(@(x) {x}, shuffledIndices, discretize(1:noOftargets, edges));
end