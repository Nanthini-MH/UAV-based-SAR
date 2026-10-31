%% ========================================================
% TRAIN EFFICIENTNET-B0 FOR UAV SAR CLASSIFICATION
% Classes are taken from the subfolder names of dataRoot
% ========================================================
clear; clc;

%% 1. DATA
% One subfolder per class: fire / none / people / smoke
dataRoot = '/MATLAB Drive/Claude/OrganizedDataset';

imds = imageDatastore(dataRoot, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Images per class:');
disp(countEachLabel(imds));

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

%% 2. AUGMENTATION (aerial images: rotation/flip are safe)
inputSize = [224 224 3];   % EfficientNet-B0 input

augmenter = imageDataAugmenter( ...
    'RandRotation',    [-180 180], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandScale',       [0.8 1.2]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augVal   = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% 3. CLASS WEIGHTS (handles class imbalance)
counts  = countEachLabel(imdsTrain).Count;
weights = sum(counts) ./ (numClasses * counts);

%% 4. MODEL: replace the classification head
net    = efficientnetb0;
lgraph = layerGraph(net);

fcName = lgraph.Layers(end-2).Name;   % fully connected
clName = lgraph.Layers(end).Name;     % classification output

newFC = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newCL = classificationLayer('Name', 'new_class', ...
    'Classes', classNames, 'ClassWeights', weights);

lgraph = replaceLayer(lgraph, fcName, newFC);
lgraph = replaceLayer(lgraph, clName, newCL);

%% 5. TRAIN
options = trainingOptions('adam', ...
    'InitialLearnRate',   1e-4, ...
    'MiniBatchSize',      32, ...
    'MaxEpochs',          8, ...
    'Shuffle',            'every-epoch', ...
    'ValidationData',     augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose',            true, ...
    'Plots',              'training-progress');

netTrained = trainNetwork(augTrain, lgraph, options);

%% 6. EVALUATE
predVal = classify(netTrained, augVal);
accuracy = mean(predVal == imdsVal.Labels);
fprintf('\nValidation accuracy: %.2f %%\n', accuracy * 100);

figure('Name', 'Confusion Matrix');
confusionchart(imdsVal.Labels, predVal);

%% 7. SAVE (network + class names together)
save('trainedEfficientNet.mat', 'netTrained', 'classNames');
disp('Saved trainedEfficientNet.mat');

%% ========================================================
% USE IN testEfficientNet.m:
%   S = load('trainedEfficientNet.mat');
%   I = imread('your_image.jpg');
%   [label, scores] = classify(S.netTrained, imresize(I, [224 224]));
%   disp(table(S.classNames, scores', 'VariableNames', {'Class','Score'}));
% ========================================================