 %% ========================================================
% COMPUTE ed (missed detection) AND ef (false alarm) FROM THE
% TRAINED EFFICIENTNET, ONE VALUE PER CLASS
%
%   ed = P(no detection | target present) = FN / (TP + FN)
%   ef = P(detection    | no target)      = FP / (FP + TN)
%
% (Paper Eq. 15 and 16. The paper uses fixed numbers; here they come
%  from our own model, so they change with the mission/class.)
% ========================================================
clear; clc;

%% 1. SETTINGS
% Folder with images the network was NOT trained on, one subfolder per
% class. Subfolder names MUST match the training classes
% (fire / none / people / smoke).
valDir = '/MATLAB Drive/Claude/ValidationSet';

defaultEd = 0.10;    % used only if a class has no positive images
defaultEf = 0.05;    % used only if a class has no negative images

%% 2. LOAD MODEL AND IMAGES
S = load('trainedEfficientNet.mat');
net        = S.netTrained;
classNames = S.classNames;

imds   = imageDatastore(valDir, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
augVal = augmentedImageDatastore([224 224 3], imds, ...
    'ColorPreprocessing', 'gray2rgb');

pred  = classify(net, augVal);
trueL = string(imds.Labels);
predL = string(pred);

%% 3. ed AND ef PER CLASS
nC = numel(classNames);
ed = zeros(nC, 1);
ef = zeros(nC, 1);

for k = 1:nC
    cls = string(classNames{k});
    isT = (trueL == cls);
    isP = (predL == cls);

    TP = sum(isT & isP);   FN = sum(isT & ~isP);
    FP = sum(~isT & isP);  TN = sum(~isT & ~isP);

    if TP + FN == 0
        warning('No "%s" images in validation set: using default ed.', cls);
        ed(k) = defaultEd;
    else
        ed(k) = FN / (TP + FN);
    end

    if FP + TN == 0
        warning('No negative images for "%s": using default ef.', cls);
        ef(k) = defaultEf;
    else
        ef(k) = FP / (FP + TN);
    end
end

% avoid exact 0 / 1 (a zero would make Bayesian updates degenerate)
ed = min(max(ed, 0.01), 0.99);
ef = min(max(ef, 0.01), 0.99);

%% 4. REPORT + SAVE
T = table(string(classNames), ed, ef, ed + ef, ...
    'VariableNames', {'Class', 'ed_missed', 'ef_falseAlarm', 'ed_plus_ef'});
disp(T);

if any(ed + ef >= 1)
    warning('ed + ef >= 1 for some class: the paper''s convergence condition fails there.');
end

save('edef.mat', 'classNames', 'ed', 'ef');
disp('Saved edef.mat');