
clear; clc; close all;

%% 1. SETTINGS
mission    = 'snow';              
imgPath    = fullfile('InputImages', 'test_image.jpg');   % CHANGE per image
tileDir    = 'GridTiles';         
autoSplit  = true;                
gridRows   = 6;                   
gridCols   = 8;
W          = 2;                   % window size (paper: W x W cells per region)
startCell  = [gridRows 1];        % [row col]; paper starts bottom-left

priorType     = 'gaussuniform';
priorPeaks    = [5 7 1.5 1.0];    % one row per peak: [row col sigma weight]
uniformWeight = 0.3;              % share of the uniform part (gaussuniform)
targetCell = [6 8];               % true target cell(s), one [r c] per row.
                                  % A detection in any other cell is a false
                                  % alarm (rescue team finds nothing).
                                  % [] : first detection ends the mission
pauseTime  = 0.4;                 % seconds between animation frames

maxSteps       = 10 * gridRows * gridCols;
batteryMin     = 35;              % full-battery endurance (minutes)
batteryReserve = 0.20;             % keep 20 % for return-to-home


edMin = 0.15;
efMin = 0.05;
tileOverlap = 0.25;
minScore = 0.80;
minViews = 3;

coverageLooks = 1;


H        = 25;                    % altitude (m)
alphaDeg = 60;  betaDeg = 80;     % camera angles: SET TO YOUR CAMERA
rx = 0.10;  ry = 0.10;            % overlap between neighbouring footprints
v        = 20;                    % UAV speed (m/s)
tProc    = 3;                     

R = gridRows;  C = gridCols;  M = R * C;


if autoSplit
    makeTiles(imgPath, tileDir, R, C, tileOverlap);
end


switch mission
    case 'snow', targetClass = 'people';
    case 'fire', targetClass = 'fire';
    otherwise,   error('mission must be ''snow'' or ''fire''');
end

S = load('trainedEfficientNet.mat');
net        = S.netTrained;
classNames = S.classNames;

E = load('edef.mat');             % made by computeEdEf.m
targetIdx = find(strcmp(classNames, targetClass));
eIdx      = find(strcmp(E.classNames, targetClass));
if isempty(targetIdx) || isempty(eIdx)
    error('Class "%s" not found in the network / edef.mat', targetClass);
end
ed = max(E.ed(eIdx), edMin);                  % missed detection probability, Eq. (15)
ef = max(E.ef(eIdx), efMin);                  % false alarm probability,      Eq. (16)

fprintf('\n=====================================================\n');
fprintf(' MISSION : %s   target class = %s\n', upper(mission), targetClass);
fprintf(' ed (missed detection) = %.3f   ef (false alarm) = %.3f\n', ed, ef);
fprintf('=====================================================\n');

%% 3. GEOMETRY (cell sizes from camera FOV, Eq. 12-13)
camW = 2 * H * tand(alphaDeg / 2);
camL = 2 * H * tand(betaDeg  / 2);
rowSpacing = camW * (1 - ry);     % metres between cell centres (rows)
colSpacing = camL * (1 - rx);     % metres between cell centres (cols)

%% 4. DISPLAY SETUP
if isfile(imgPath)
    Iimg = imread(imgPath);
    [imgH, imgW, ~] = size(Iimg);
else
    Iimg = [];  imgH = 600;  imgW = 800;
end
st.Iimg = Iimg;  st.imgH = imgH;  st.imgW = imgW;
st.rowEdges = linspace(0, imgH, R + 1);
st.colEdges = linspace(0, imgW, C + 1);
st.mission = mission;  st.targetClass = targetClass;

fig = figure('Name', 'Search mission', 'Position', [40 40 1350 540]);
ax1 = subplot(1, 2, 1);
ax2 = subplot(1, 2, 2);

% manual stop button
setappdata(fig, 'stopRequested', false);
uicontrol(fig, 'Style', 'pushbutton', 'String', 'STOP MISSION', ...
    'Position', [10 10 140 32], 'FontWeight', 'bold', ...
    'BackgroundColor', [0.85 0.2 0.2], 'ForegroundColor', 'w', ...
    'Callback', @(~, ~) setappdata(fig, 'stopRequested', true));

%% 5. INITIAL BELIEF MAP
[cx, cy] = meshgrid(1:C, 1:R);
G = zeros(R, C);                      % Gaussian mixture
for i = 1:size(priorPeaks, 1)
    G = G + priorPeaks(i, 4) * exp(-((cx - priorPeaks(i, 2)).^2 + ...
        (cy - priorPeaks(i, 1)).^2) / (2 * priorPeaks(i, 3)^2));
end
G = G / sum(G(:));
U = ones(R, C) / M;                   % uniform

switch priorType
    case 'uniform',      P = U;
    case 'gaussian',     P = G + 1e-3 * U;
    case 'gaussuniform', P = (1 - uniformWeight) * G + uniformWeight * U;
    otherwise, error('priorType must be uniform, gaussian or gaussuniform');
end
P = P / sum(P(:));

%% 6. ALGORITHM 1: PATH PLANNING LOOP
cfm        = struct('minScore', minScore, 'minViews', minViews);
pos        = startCell;
path       = pos;
labelGrid  = repmat("--", R, C);
logRows    = {};
found      = false;
stopReason = '';
looks      = zeros(R, C);         % times each cell was observed
falseAlarms = 0;
totalDist  = 0;
step       = 0;
first      = true;
Rn         = [ceil(pos(1) / W), ceil(pos(2) / W)];   % current region

while true
    for k = 1:W^2                                   % W^2 observations in R_n
        if first
            first = false;                          % observe start cell (step 0)
        else
            nxt = chooseNextCell(P, pos, Rn, W);    % Eq. (30)-(32)
            totalDist = totalDist + norm([(nxt(1) - pos(1)) * rowSpacing, ...
                                          (nxt(2) - pos(2)) * colSpacing]);
            pos  = nxt;
            step = step + 1;
            path(end + 1, :) = pos;                 %#ok<SAGROW>
        end

        % Algorithm 2: image from folder -> EfficientNet -> belief update
        [P, found, fa, det, label, conf] = observeAndUpdate(P, pos, net, ...
            tileDir, classNames, targetIdx, ed, ef, targetCell, cfm);   % Eq. (23)
        falseAlarms = falseAlarms + fa;
        labelGrid(pos(1), pos(2)) = string(label);
        looks(pos(1), pos(2)) = looks(pos(1), pos(2)) + 1;
        logRows(end + 1, :) = {step, (pos(1) - 1) * C + pos(2), pos(1), ...
            pos(2), label, conf * 100, det};        %#ok<SAGROW>

        fprintf('step %3d | cell %2d (r%d,c%d) | %-7s %6.2f %% | detected=%d\n', ...
            step, (pos(1) - 1) * C + pos(2), pos(1), pos(2), label, conf * 100, det);

        % battery used so far (flying + imaging time against the budget)
        elapsed     = totalDist / v + (step + 1) * tProc;
        batteryUsed = 100 * elapsed / (batteryMin * 60);

        if ~isvalid(fig), stopReason = 'figure closed (manual stop)'; break; end
        render(ax1, ax2, st, P, path, labelGrid, step, found, batteryUsed);
        pause(pauseTime);

        if ~isvalid(fig)
            stopReason = 'figure closed (manual stop)';
        elseif found
            stopReason = 'target found';
        elseif all(looks(:) >= coverageLooks)
            stopReason = 'whole area searched, target not found';
        elseif step >= maxSteps
            stopReason = 'max steps reached';
        elseif batteryUsed >= 100 * (1 - batteryReserve)
            stopReason = 'battery low, return to home';
        elseif getappdata(fig, 'stopRequested')
            stopReason = 'stopped manually';
        end
        if ~isempty(stopReason), break; end
    end
    if ~isempty(stopReason), break; end

    % next region: Eq. (26)-(29)
    Rn = chooseRegion(P, [ceil(pos(1) / W), ceil(pos(2) / W)], W);
end

%% 7. RESULTS
totalTime = totalDist / v + (step + 1) * tProc;

fprintf('\n=====================================================\n');
if found
    fprintf(' TARGET FOUND in cell %d (row %d, col %d)\n', ...
        (pos(1) - 1) * C + pos(2), pos(1), pos(2));
else
    fprintf(' TARGET NOT FOUND\n');
end
fprintf(' Stopped because           : %s\n', stopReason);
fprintf(' Battery used              : %.1f %%  (of %d min endurance)\n', ...
    100 * totalTime / (batteryMin * 60), batteryMin);
fprintf(' Steps (cell moves)        : %d\n', step);
fprintf(' Cells observed            : %d  (%d distinct)\n', step + 1, ...
    size(unique(path, 'rows'), 1));
fprintf(' False alarms              : %d\n', falseAlarms);
fprintf(' Flight distance           : %.1f m\n', totalDist);
fprintf(' Mission time              : %.1f s  (%.1f s flying + %.1f s imaging)\n', ...
    totalTime, totalDist / v, (step + 1) * tProc);
fprintf('=====================================================\n');

[~, order] = sort(P(:), 'descend');
fprintf('\nHighest final belief cells:\n');
for i = 1:3
    [rr, cc] = ind2sub([R C], order(i));
    fprintf('  cell %2d (r%d,c%d) : %.2f %%\n', order(i), rr, cc, P(order(i)) * 100);
end

fprintf('\nWhat each observed grid cell contains (-- = not visited):\n');
for r = 1:R
    fprintf('%s\n', strjoin(compose('%-8s', labelGrid(r, :)), ' '));
end

T = cell2table(logRows, 'VariableNames', ...
    {'Step', 'CellID', 'GridRow', 'GridCol', 'Predicted', 'Conf_pct', 'Detected'});
disp(T);

%% ========================================================
% LOCAL FUNCTIONS
% ========================================================

function [P, found, falseAlarm, det, label, conf] = observeAndUpdate( ...
        P, pos, net, tileDir, classNames, targetIdx, ed, ef, targetCell, cfm)
    [det, label, conf] = observeCell(pos(1), pos(2), net, tileDir, ...
        classNames, targetIdx, cfm);
    found = false;  falseAlarm = 0;
    if det
        if isempty(targetCell) || ismember(pos, targetCell, 'rows')
            found = true;                         % reported to ground station
            return;
        end
        falseAlarm = 1;                           % rescue team finds nothing
        P(pos(1), pos(2)) = 0;
        P = P / sum(P(:));
    else
        % Eq. (23): update after "no detection" in the visited cell
        pi_ = P(pos(1), pos(2));
        b   = ed * pi_ + (1 - ef) * (1 - pi_);
        P   = P * (1 - ef) / b;                   % j ~= i
        P(pos(1), pos(2)) = pi_ * ed / b;         % j == i
    end
end

function [det, label, conf] = observeCell(r, c, net, tileDir, classNames, targetIdx, cfm)
    fname = fullfile(tileDir, sprintf('tile_r%02d_c%02d.jpg', r, c));
    if ~isfile(fname), error('Tile not found: %s', fname); end
    I = imread(fname);
    if size(I, 3) == 1, I = repmat(I, [1 1 3]); end
    [det, label, conf] = confirmDetection(net, I(:, :, 1:3), classNames, ...
        targetIdx, cfm.minScore, cfm.minViews);
end

function [det, label, conf] = confirmDetection(net, tile, classNames, ...
        targetIdx, minScore, minViews)
    
    tile  = imresize(tile, [224 224]);
    batch = cat(4, tile, fliplr(tile), flipud(tile), rot90(tile));
    [~, sc] = classify(net, batch);              % 4 x nClasses
    [m, idx] = max(sc, [], 2);
    idx0   = idx(1);
    conf   = m(1);
    tScore = sc(1, targetIdx);
    agree  = sum(idx == targetIdx);
    det    = (idx0 == targetIdx) && tScore >= minScore && agree >= minViews;
    label  = classNames{idx0};
    if idx0 == targetIdx && ~det
        label = [label '?'];
    end
end

function makeTiles(imgPath, tileDir, R, C, ov)
    if ~isfile(imgPath), error('Input image not found: %s', imgPath); end
    if exist(tileDir, 'dir')
        delete(fullfile(tileDir, 'tile_r*_c*.jpg'));      % remove old tiles
    else
        mkdir(tileDir);
    end
    I = imread(imgPath);
    if size(I, 3) == 1, I = repmat(I, [1 1 3]); end
    [Hh, Ww, ~] = size(I);
    rowEdges = round(linspace(0, Hh, R + 1));
    colEdges = round(linspace(0, Ww, C + 1));
    for r = 1:R
        for c = 1:C
            ch = Hh / R;  cw = Ww / C;               % cell size in pixels
            y1 = max(1,  round(rowEdges(r)     - ov * ch) + 1);
            y2 = min(Hh, round(rowEdges(r + 1) + ov * ch));
            x1 = max(1,  round(colEdges(c)     - ov * cw) + 1);
            x2 = min(Ww, round(colEdges(c + 1) + ov * cw));
            tile = I(y1:y2, x1:x2, :);
            imwrite(tile, fullfile(tileDir, sprintf('tile_r%02d_c%02d.jpg', r, c)));
        end
    end
end

function Rn = chooseRegion(P, Rc, W)
    % Eq. (26): region probabilities
    [R, C] = size(P);
    nRr = ceil(R / W);  nRc = ceil(C / W);
    PR = zeros(nRr, nRc);
    for a = 1:nRr
        for b = 1:nRc
            rows = (a - 1) * W + 1 : min(a * W, R);
            cols = (b - 1) * W + 1 : min(b * W, C);
            PR(a, b) = sum(sum(P(rows, cols)));
        end
    end
    % Eq. (27): region with the highest probability
    [pmax, li] = max(PR(:));
    [ra, rb] = ind2sub(size(PR), li);
    Rmax = [ra rb];
    if isequal(Rmax, Rc), Rn = Rc; return; end

    % Eq. (28): best adjacent region
    adj = zeros(0, 2);
    for dr = -1:1
        for dc = -1:1
            q = Rc + [dr dc];
            if (dr ~= 0 || dc ~= 0) && q(1) >= 1 && q(1) <= nRr && ...
                    q(2) >= 1 && q(2) <= nRc
                adj(end + 1, :) = q;              %#ok<AGROW>
            end
        end
    end
    pa = PR(sub2ind(size(PR), adj(:, 1), adj(:, 2)));
    [plocal, j] = max(pa);
    Rloc = adj(j, :);

    % R_p: adjacent region on the straight line from R_c towards R_max
    Rp = Rc + sign(Rmax - Rc);

    % Eq. (29): is flying to R_max worth the extra distance?
    dmax = norm(Rmax - Rc);
    dloc = norm(Rloc - Rc);
    if (pmax / plocal) > (dmax / dloc)
        Rn = Rp;
    else
        Rn = Rloc;
    end
end

function nxt = chooseNextCell(P, pos, Rn, W)
    % Eq. (30)-(32): among straight paths of length W from the current
    % cell, move to the adjacent cell whose window minimises E[T_a].
    [R, C] = size(P);  M = R * C;
    r1 = (Rn(1) - 1) * W + 1;  r2 = min(Rn(1) * W, R);
    c1 = (Rn(2) - 1) * W + 1;  c2 = min(Rn(2) * W, C);
    ctr = [(r1 + r2) / 2, (c1 + c2) / 2];
    inReg = pos(1) >= r1 && pos(1) <= r2 && pos(2) >= c1 && pos(2) <= c2;

    dirs = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    cand = zeros(0, 2);  ET = zeros(0, 1);  ok = false(0, 1);

    for i = 1:size(dirs, 1)
        a = pos + dirs(i, :);
        if a(1) < 1 || a(1) > R || a(2) < 1 || a(2) > C, continue; end

        cells = zeros(0, 2);                      % window of up to W cells
        for k = 1:W
            q = pos + k * dirs(i, :);
            if q(1) < 1 || q(1) > R || q(2) < 1 || q(2) > C, break; end
            cells(end + 1, :) = q;                %#ok<AGROW>
        end
        n = size(cells, 1);
        p = P(sub2ind([R C], cells(:, 1), cells(:, 2)));

        % Eq. (32)
        ET(end + 1, 1) = sum((1:n)' .* p) + (M - n) * (1 - sum(p)); %#ok<AGROW>
        cand(end + 1, :) = a;                     %#ok<AGROW>

        inside = a(1) >= r1 && a(1) <= r2 && a(2) >= c1 && a(2) <= c2;
        closer = norm(a - ctr) < norm(pos - ctr);
        if inReg
            ok(end + 1, 1) = inside;              %#ok<AGROW>
        else
            ok(end + 1, 1) = closer;              %#ok<AGROW>
        end
    end

    if ~any(ok), ok(:) = true; end
    ET(~ok) = inf;
    [~, best] = min(ET);
    nxt = cand(best, :);
end

function render(ax1, ax2, st, P, path, labelGrid, step, found, batteryUsed)
    [R, C] = size(P);
    xc = @(c) (st.colEdges(c) + st.colEdges(c + 1)) / 2;
    yc = @(r) (st.rowEdges(r) + st.rowEdges(r + 1)) / 2;
    px = arrayfun(xc, path(:, 2));
    py = arrayfun(yc, path(:, 1));

    % ---- left: image, grid, class labels, UAV path ----
    cla(ax1);
    if isempty(st.Iimg)
        imshow(ones(st.imgH, st.imgW, 3), 'Parent', ax1);
    else
        imshow(st.Iimg, 'Parent', ax1);
    end
    hold(ax1, 'on');
    for c = 1:C + 1
        plot(ax1, [st.colEdges(c) st.colEdges(c)], [0 st.imgH], 'y-', 'LineWidth', 0.5);
    end
    for r = 1:R + 1
        plot(ax1, [0 st.imgW], [st.rowEdges(r) st.rowEdges(r)], 'y-', 'LineWidth', 0.5);
    end
    cw  = st.colEdges(2) - st.colEdges(1);     % cell width  (pixels)
    chh = st.rowEdges(2) - st.rowEdges(1);     % cell height (pixels)
    for r = 1:R
        for c = 1:C
            if labelGrid(r, c) ~= "--"
                % class label in the TOP-LEFT corner: the path never covers it
                text(ax1, st.colEdges(c) + 0.06 * cw, st.rowEdges(r) + 0.06 * chh, ...
                    char(labelGrid(r, c)), 'Color', 'w', ...
                    'BackgroundColor', [0 0 0], 'Margin', 1, 'FontSize', 7, ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
                if labelGrid(r, c) == string(st.targetClass)
                    % cell where the target class was reported
                    % (a false alarm if the mission keeps going)
                    rectangle(ax1, 'Position', [st.colEdges(c) st.rowEdges(r) cw chh], ...
                        'EdgeColor', 'r', 'LineWidth', 1.5);
                end
            end
        end
    end
    plot(ax1, px, py, '-', 'Color', [0.2 0.9 0.2], 'LineWidth', 1);
    plot(ax1, px, py, '.', 'Color', [0.2 0.9 0.2], 'MarkerSize', 8);
    plot(ax1, px(end), py(end), 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
    if found
        r = path(end, 1);  c = path(end, 2);
        rectangle(ax1, 'Position', [st.colEdges(c) st.rowEdges(r) ...
            st.colEdges(c + 1) - st.colEdges(c), ...
            st.rowEdges(r + 1) - st.rowEdges(r)], ...
            'EdgeColor', 'r', 'LineWidth', 3);
    end
    title(ax1, sprintf('%s mission: UAV path (step %d, battery used %.0f %%)', ...
        upper(st.mission), step, batteryUsed));

    cla(ax2);
    imagesc(ax2, P);
    colormap(ax2, hot);
    caxis(ax2, [0 max(P(:))]);
    colorbar(ax2);
    hold(ax2, 'on');
    for r = 1:R
        for c = 1:C
            if P(r, c) > 0.5 * max(P(:)), tc = 'k'; else, tc = 'w'; end
            text(ax2, c - 0.46, r - 0.44, sprintf('%.1f', P(r, c) * 100), ...
                'Color', tc, 'FontSize', 7, ...
                'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
        end
    end
    plot(ax2, path(:, 2), path(:, 1), '-', 'Color', [0.2 0.9 0.2], 'LineWidth', 1);
    plot(ax2, path(:, 2), path(:, 1), '.', 'Color', [0.2 0.9 0.2], 'MarkerSize', 8);
    plot(ax2, path(end, 2), path(end, 1), 'cp', 'MarkerSize', 15, 'MarkerFaceColor', 'c');
    axis(ax2, 'image');
    set(ax2, 'XTick', 1:C, 'YTick', 1:R);
    title(ax2, sprintf('Belief map of "%s" in %% (step %d)', st.targetClass, step));
    drawnow;
end