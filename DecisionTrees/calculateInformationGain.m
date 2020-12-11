function informationGain = calculateInformationGain(datasetEntropy, ratio)
    informationGain = datasetEntropy - ratio * entropy;
end