%% ========================================================
% COMPARE:  PROPOSED (windowing + Bayesian belief update)  vs  ZIGZAG
% Monte-Carlo simulation like the paper's Sec. V-C (Tables VI / VII):
%   - the target cell is drawn at random (from the prior, as in the paper)
%   - the sensor is simulated with the ed / ef measured on OUR classifier:
%       target in cell -> detected with probability 1 - ed
%       no target      -> false alarm with probability ef
%   - both algorithms see the SAME target and the SAME random numbers
% ========================================================
clear; clc; close all;

%% 1. SETTINGS
mission    = 'snow';              % 'snow' or 'fire' (picks ed/ef of that class)
gridRows   = 6;   gridCols = 8;
W          = 2;                   % window size of the proposed algorithm
startCell  = [gridRows 1];        % bottom-left corner
nRuns      = 100;                 % paper used 25 experiments per case
targetDist = 'prior';             % 'prior' (as in paper) or 'uniform'
priorTypes = {'uniform', 'gaussian', 'gaussuniform'};
priorPeaks = [5 7 1.5 1.0];       % [row col sigma weight] per peak
uniformWeight = 0.3;
edManual   = [];                  % e.g. 0.10 to override the measured ed
efManual   = [];                  % e.g. 0.05 to override the measured ef

% camera / flight (same as searchMission.m)
H = 25;  alphaDeg = 60;  betaDeg = 80;  rx = 0.10;  ry = 0.10;
v = 20;  tProc = 3;

% rotary-wing power model, paper Eq. (33). Typical values for a small
% rotary-wing UAV from Zeng et al. [68]: CHECK them against that paper.
Epar = struct('P0', 79.86, 'Pi', 88.63, 'Utip', 120, 'v0', 4.03, ...
              'd0', 0.6, 'rho', 1.225, 's', 0.05, 'A', 0.503);

R = gridRows;  C = gridCols;  M = R * C;
maxSteps = 20 * M;
rng(1);                           % same results every run

%% 2. ed / ef OF THE MISSION'S TARGET CLASS
switch mission
    case 'snow', targetClass = 'people';
    case 'fire', targetClass = 'fire';
    otherwise,   error('mission must be ''snow'' or ''fire''');
end
E = load('edef.mat');
eIdx = find(strcmp(E.classNames, targetClass));
ed = E.ed(eIdx);  ef = E.ef(eIdx);
if ~isempty(edManual), ed = edManual; end
if ~isempty(efManual), ef = efManual; end
fprintf('Mission %s (target = %s):  ed = %.3f   ef = %.3f\n', ...
    upper(mission), targetClass, ed, ef);

rowSp = 2 * H * tand(alphaDeg / 2) * (1 - ry);
colSp = 2 * H * tand(betaDeg  / 2) * (1 - rx);

%% 3. MONTE-CARLO
names = {'Proposed', 'Zigzag'};
nP = numel(priorTypes);
Tm = zeros(nRuns, 2, nP);   % mission time (s)
St = zeros(nRuns, 2, nP);   % steps
Di = zeros(nRuns, 2, nP);   % flight distance (m)
En = zeros(nRuns, 2, nP);   % energy (kJ)
FA = zeros(nRuns, 2, nP);   % false alarms
Fd = zeros(nRuns, 2, nP);   % target found (1/0)

for p = 1:nP
    P0 = makePrior(priorTypes{p}, R, C, priorPeaks, uniformWeight);
    if strcmp(targetDist, 'uniform')
        Ptar = ones(R, C) / M;
    else
        Ptar = P0;
    end
    cdf = cumsum(Ptar(:));

    for run = 1:nRuns
        tIdx = find(rand <= cdf, 1);
        [tr, tc] = ind2sub([R C], tIdx);
        t = [tr tc];
        u = rand(1, maxSteps + 1);

        [pathA, fdA, faA] = runProposed(P0, startCell, W, t, ed, ef, u, maxSteps);
        [pathB, fdB, faB] = runZigzag(R, C, startCell, t, ed, ef, u, maxSteps);

        paths = {pathA, pathB};  fds = [fdA fdB];  fas = [faA faB];
        for a = 1:2
            [tt, dd, ee] = pathMetrics(paths{a}, rowSp, colSp, v, tProc, Epar);
            Tm(run, a, p) = tt;
            St(run, a, p) = size(paths{a}, 1) - 1;
            Di(run, a, p) = dd;
            En(run, a, p) = ee;
            FA(run, a, p) = fas(a);
            Fd(run, a, p) = fds(a);
        end
    end
end

%% 4. RESULTS TABLE  (paper Table VI / VII style)
rows = {};
for p = 1:nP
    for a = 1:2
        rows(end + 1, :) = {priorTypes{p}, names{a}, ...
            mean(Tm(:, a, p)), std(Tm(:, a, p)), mean(St(:, a, p)), ...
            mean(Di(:, a, p)), mean(En(:, a, p)), mean(FA(:, a, p)), ...
            100 * (1 - mean(Fd(:, a, p)))};            %#ok<SAGROW>
    end
end
Tbl = cell2table(rows, 'VariableNames', {'Prior', 'Algorithm', 'Time_s', ...
    'Time_std', 'Steps', 'Distance_m', 'Energy_kJ', 'FalseAlarms', 'NotFound_pct'});
disp(Tbl);

fprintf('\nProposed vs Zigzag  (%d runs, target drawn from "%s"):\n', nRuns, targetDist);
for p = 1:nP
    fprintf('  %-13s : %.2fx faster, %.2fx less energy, proposed wins in %.0f %% of runs\n', ...
        priorTypes{p}, mean(Tm(:, 2, p)) / mean(Tm(:, 1, p)), ...
        mean(En(:, 2, p)) / mean(En(:, 1, p)), ...
        100 * mean(Tm(:, 1, p) < Tm(:, 2, p)));
end

%% 5. FIGURES
figure('Name', 'Proposed vs Zigzag', 'Position', [60 60 1100 420]);

subplot(1, 2, 1);
mt = squeeze(mean(Tm, 1))';  sd = squeeze(std(Tm, 0, 1))';
if nP == 1, mt = mt(:)'; sd = sd(:)'; end
b = bar(mt);  hold on;
for a = 1:2
    errorbar(b(a).XEndPoints, mt(:, a), sd(:, a), 'k.');
end
set(gca, 'XTickLabel', priorTypes);
ylabel('Average mission time (s)');
legend(names, 'Location', 'northwest');
title('Time to find the target');

subplot(1, 2, 2);
me = squeeze(mean(En, 1))';
if nP == 1, me = me(:)'; end
bar(me);
set(gca, 'XTickLabel', priorTypes);
ylabel('Average energy (kJ)');
legend(names, 'Location', 'northwest');
title('Energy to find the target');

%% ========================================================
% LOCAL FUNCTIONS
% ========================================================

function P = makePrior(type, R, C, peaks, uw)
    [cx, cy] = meshgrid(1:C, 1:R);
    G = zeros(R, C);
    for i = 1:size(peaks, 1)
        G = G + peaks(i, 4) * exp(-((cx - peaks(i, 2)).^2 + ...
            (cy - peaks(i, 1)).^2) / (2 * peaks(i, 3)^2));
    end
    G = G / sum(G(:));
    U = ones(R, C) / (R * C);
    switch type
        case 'uniform',      P = U;
        case 'gaussian',     P = G + 1e-3 * U;
        case 'gaussuniform', P = (1 - uw) * G + uw * U;
        otherwise, error('unknown prior type');
    end
    P = P / sum(P(:));
end

function [path, found, fa] = runProposed(P, start, W, t, ed, ef, u, maxSteps)
    pos = start;  path = pos;  found = false;  fa = 0;
    step = 0;  first = true;
    Rn = [ceil(pos(1) / W), ceil(pos(2) / W)];
    while true
        for k = 1:W^2
            if first
                first = false;
            else
                pos = chooseNextCell(P, pos, Rn, W);
                step = step + 1;
                path(end + 1, :) = pos;                 %#ok<AGROW>
            end
            isT = isequal(pos, t);
            if isT, det = u(step + 1) > ed; else, det = u(step + 1) < ef; end
            if det
                if isT, found = true; return; end
                fa = fa + 1;                            % false alarm
                P(pos(1), pos(2)) = 0;
                P = P / sum(P(:));
            else
                pi_ = P(pos(1), pos(2));                % Eq. (23)
                b   = ed * pi_ + (1 - ef) * (1 - pi_);
                P   = P * (1 - ef) / b;
                P(pos(1), pos(2)) = pi_ * ed / b;
            end
            if step >= maxSteps, return; end
        end
        Rn = chooseRegion(P, [ceil(pos(1) / W), ceil(pos(2) / W)], W);
    end
end

function [path, found, fa] = runZigzag(R, C, start, t, ed, ef, u, maxSteps)
    % back-and-forth (serpentine) sweep from the start corner, repeated
    if start(1) == 1, rows = 1:R; else, rows = R:-1:1; end
    if start(2) == 1, cols = 1:C; else, cols = C:-1:1; end
    seq = zeros(R * C, 2);  n = 0;
    for i = 1:R
        if mod(i, 2) == 1, cc = cols; else, cc = fliplr(cols); end
        for c = cc
            n = n + 1;  seq(n, :) = [rows(i) c];
        end
    end
    cyc = [seq; seq(end - 1:-1:2, :)];     % forth, then back
    L = size(cyc, 1);

    path = zeros(maxSteps + 1, 2);  found = false;  fa = 0;
    for step = 0:maxSteps
        pos = cyc(mod(step, L) + 1, :);
        path(step + 1, :) = pos;
        isT = isequal(pos, t);
        if isT, det = u(step + 1) > ed; else, det = u(step + 1) < ef; end
        if det
            if isT
                found = true;
                path = path(1:step + 1, :);
                return;
            end
            fa = fa + 1;
        end
    end
end

function [time, dist, energy] = pathMetrics(path, rowSp, colSp, v, tProc, E)
    d = diff(path, 1, 1);
    dist = sum(sqrt((d(:, 1) * rowSp).^2 + (d(:, 2) * colSp).^2));
    nObs = size(path, 1);
    tFly = dist / v;
    time = tFly + nObs * tProc;
    energy = (uavPower(v, E) * tFly + uavPower(0, E) * nObs * tProc) / 1000;  % kJ
end

function p = uavPower(v, E)
    % paper Eq. (33): blade profile + induced + parasite power
    p = E.P0 * (1 + 3 * v^2 / E.Utip^2) ...
      + E.Pi * sqrt(sqrt(1 + v^4 / (4 * E.v0^4)) - v^2 / (2 * E.v0^2)) ...
      + 0.5 * E.d0 * E.rho * E.s * E.A * v^3;
end

function Rn = chooseRegion(P, Rc, W)
    [R, C] = size(P);
    nRr = ceil(R / W);  nRc = ceil(C / W);
    PR = zeros(nRr, nRc);
    for a = 1:nRr
        for b = 1:nRc
            rows = (a - 1) * W + 1 : min(a * W, R);
            cols = (b - 1) * W + 1 : min(b * W, C);
            PR(a, b) = sum(sum(P(rows, cols)));         % Eq. (26)
        end
    end
    [pmax, li] = max(PR(:));                            % Eq. (27)
    [ra, rb] = ind2sub(size(PR), li);
    Rmax = [ra rb];
    if isequal(Rmax, Rc), Rn = Rc; return; end

    adj = zeros(0, 2);                                  % Eq. (28)
    for dr = -1:1
        for dc = -1:1
            q = Rc + [dr dc];
            if (dr ~= 0 || dc ~= 0) && q(1) >= 1 && q(1) <= nRr && ...
                    q(2) >= 1 && q(2) <= nRc
                adj(end + 1, :) = q;                    %#ok<AGROW>
            end
        end
    end
    pa = PR(sub2ind(size(PR), adj(:, 1), adj(:, 2)));
    [plocal, j] = max(pa);
    Rloc = adj(j, :);
    Rp = Rc + sign(Rmax - Rc);

    dmax = norm(Rmax - Rc);                             % Eq. (29)
    dloc = norm(Rloc - Rc);
    if (pmax / plocal) > (dmax / dloc), Rn = Rp; else, Rn = Rloc; end
end

function nxt = chooseNextCell(P, pos, Rn, W)
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
        cells = zeros(0, 2);
        for k = 1:W
            q = pos + k * dirs(i, :);
            if q(1) < 1 || q(1) > R || q(2) < 1 || q(2) > C, break; end
            cells(end + 1, :) = q;                      %#ok<AGROW>
        end
        n = size(cells, 1);
        p = P(sub2ind([R C], cells(:, 1), cells(:, 2)));
        ET(end + 1, 1) = sum((1:n)' .* p) + (M - n) * (1 - sum(p)); %#ok<AGROW>
        cand(end + 1, :) = a;                           %#ok<AGROW>
        inside = a(1) >= r1 && a(1) <= r2 && a(2) >= c1 && a(2) <= c2;
        closer = norm(a - ctr) < norm(pos - ctr);
        if inReg, ok(end + 1, 1) = inside; else, ok(end + 1, 1) = closer; end %#ok<AGROW>
    end

    if ~any(ok), ok(:) = true; end
    ET(~ok) = inf;
    [~, best] = min(ET);
    nxt = cand(best, :);
end