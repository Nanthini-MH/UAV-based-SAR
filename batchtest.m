%% ========================================================
% BATCH TEST: does the system say FOUND only where a target really is?
%
% Put the test images in InputImages and name them by what they contain:
%     person_1.jpg, person_2.jpg   -> a person is in the image
%     fire_1.jpg                   -> a fire is in the image
%     none_1.jpg, none_2.jpg ...   -> nothing to find (use many of these!)
%
% Every image is run through BOTH missions (snow = people, fire = fire).
%% 
% A mission says FOUND if at least one tile passes the detection rule.
% (With full coverage the UAV visits every cell, so this is the same
%  verdict searchMission.m ends with. The cell can differ: check it there.)
% ========================================================
clear; clc;

%% 1. SETTINGS (keep the same values as searchMission.m)
inputDir    = 'InputImages';
gridRows    = 6;
gridCols    = 8;
tileOverlap = 0.25;
minScore    = 0.80;        % target-class score needed
minViews    = 3;           % views (of 4) that must agree

missionNames = {'snow', 'fire'};
targetNames  = {'people', 'fire'};

%% 2. LOAD
S = load('trainedEfficientNet.mat');
net = S.netTrained;  classNames = S.classNames;
tIdx = [find(strcmp(classNames, 'people')), find(strcmp(classNames, 'fire'))];
if numel(tIdx) < 2, error('classNames must contain people and fire'); end

imds = imageDatastore(inputDir);
if isempty(imds.Files), error('No images in %s', inputDir); end

%% 3. RUN EVERY IMAGE
rows = {};
for i = 1:numel(imds.Files)
    I = imread(imds.Files{i});
    if size(I, 3) == 1, I = repmat(I, [1 1 3]); end
    I = I(:, :, 1:3);
    [~, name] = fileparts(imds.Files{i});
    expected = [startsWith(name, 'person'), startsWith(name, 'fire')];

    nDet = zeros(1, 2);  bestScore = zeros(1, 2);  bestCell = zeros(2, 2);
    for r = 1:gridRows
        for c = 1:gridCols
            tile  = imresize(cutTile(I, r, c, gridRows, gridCols, tileOverlap), [224 224]);
            batch = cat(4, tile, fliplr(tile), flipud(tile), rot90(tile));
            [~, sc] = classify(net, batch);              % 4 x nClasses
            [~, idx] = max(sc, [], 2);
            for m = 1:2
                t = tIdx(m);
                if idx(1) == t && sc(1, t) >= minScore && sum(idx == t) >= minViews
                    nDet(m) = nDet(m) + 1;
                    if sc(1, t) > bestScore(m)
                        bestScore(m) = sc(1, t);  bestCell(m, :) = [r c];
                    end
                end
            end
        end
    end

    for m = 1:2
        found = nDet(m) > 0;
        if found, verdict = "FOUND"; else, verdict = "NOT FOUND"; end
        if expected(m), exp_ = "FOUND"; else, exp_ = "NOT FOUND"; end
        if found == expected(m), ok = "yes"; else, ok = "NO"; end
        if found
            cellStr = sprintf('r%d,c%d', bestCell(m, 1), bestCell(m, 2));
        else
            cellStr = '-';
        end
        rows(end + 1, :) = {name, missionNames{m}, char(verdict), nDet(m), ...
            cellStr, 100 * bestScore(m), char(exp_), char(ok)};      %#ok<SAGROW>
    end
end

%% 4. REPORT
T = cell2table(rows, 'VariableNames', {'Image', 'Mission', 'Verdict', ...
    'DetectedTiles', 'BestCell', 'BestScore_pct', 'Expected', 'Correct'});
disp(T);

nOK = sum(strcmp(T.Correct, 'yes'));
fprintf('\n%d of %d checks correct.\n', nOK, height(T));
fprintf('False alarms (FOUND but nothing there): %d\n', ...
    sum(strcmp(T.Verdict, 'FOUND') & strcmp(T.Expected, 'NOT FOUND')));
fprintf('Misses (NOT FOUND but target present) : %d\n', ...
    sum(strcmp(T.Verdict, 'NOT FOUND') & strcmp(T.Expected, 'FOUND')));

%% ========================================================
function tile = cutTile(I, r, c, R, C, ov)
[Hh, Ww, ~] = size(I);
ch = Hh / R;  cw = Ww / C;
y1 = max(1,  round((r - 1) * ch - ov * ch) + 1);
y2 = min(Hh, round(r * ch + ov * ch));
x1 = max(1,  round((c - 1) * cw - ov * cw) + 1);
x2 = min(Ww, round(c * cw + ov * cw));
tile = I(y1:y2, x1:x2, :);
end