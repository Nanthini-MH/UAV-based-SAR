%% ========================================================
% STAGE 2: MISSION-BASED GRID DETECTION
% mission 'snow' -> look for people
% mission 'fire' -> look for fire
% ========================================================
clear; clc;

%% 1. SETTINGS
mission   = 'snow';              % 'snow' or 'fire'
imgPath   = 'test_image.jpg';    % CHANGE to your UAV image
gridRows  = 6;
gridCols  = 8;
threshold = 0.50;                % cell is "detected" above this probability

switch mission
    case 'snow', targetClass = 'people';
    case 'fire', targetClass = 'fire';
    otherwise,   error('mission must be ''snow'' or ''fire''');
end

%% 2. LOAD TRAINED NETWORK
S = load('trainedEfficientNet.mat');
net        = S.netTrained;
classNames = S.classNames;

targetIdx = find(strcmp(classNames, targetClass));
if isempty(targetIdx)
    error('Class "%s" not found. Classes are: %s', ...
        targetClass, strjoin(classNames', ', '));
end

%% 3. LOAD IMAGE
I = imread(imgPath);
if size(I, 3) == 1, I = repmat(I, [1 1 3]); end
[H, W, ~] = size(I);

rowEdges = round(linspace(0, H, gridRows + 1));
colEdges = round(linspace(0, W, gridCols + 1));

%% 4. CLASSIFY EVERY CELL
P = zeros(gridRows, gridCols);   % probability of target in each cell

for r = 1:gridRows
    for c = 1:gridCols
        tile = I(rowEdges(r)+1:rowEdges(r+1), ...
            colEdges(c)+1:colEdges(c+1), :);
        tile = imresize(tile, [224 224]);
        [~, scores] = classify(net, tile);
        P(r, c) = scores(targetIdx);
    end
end

%% 5. RESULTS
[maxP, linIdx] = max(P(:));
[bestR, bestC] = ind2sub(size(P), linIdx);
detected = P >= threshold;

fprintf('\n=====================================\n');
fprintf(' MISSION : %s  (target = %s)\n', upper(mission), targetClass);
fprintf('=====================================\n');
fprintf('Cells above threshold : %d of %d\n', nnz(detected), numel(P));
fprintf('Best cell             : row %d, col %d  (%.2f %%)\n', ...
    bestR, bestC, maxP * 100);

%% 6. VISUALISE
figure('Name', 'Grid Detection', 'Position', [100 100 1100 450]);

subplot(1, 2, 1);
imshow(I); hold on;
for r = 1:gridRows
    for c = 1:gridCols
        x = colEdges(c) + 1;  y = rowEdges(r) + 1;
        w = colEdges(c+1) - colEdges(c);
        h = rowEdges(r+1) - rowEdges(r);
        if detected(r, c)
            rectangle('Position', [x y w h], 'EdgeColor', 'r', 'LineWidth', 2);
        else
            rectangle('Position', [x y w h], 'EdgeColor', [1 1 1], 'LineWidth', 0.5);
        end
    end
end
title(sprintf('%s mission: %s detections', upper(mission), targetClass));

subplot(1, 2, 2);
imagesc(P, [0 1]); colormap('hot'); colorbar;
axis image;
title('Target probability per cell');
xlabel('Column'); ylabel('Row');