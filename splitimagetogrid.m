%% ========================================================
% SPLIT A WHOLE IMAGE INTO GRID TILES
% - saves every tile in a folder  (tile_r01_c03.jpg = row 1, col 3)
% - writes gridIndex.csv with cell ID and pixel coordinates
% - shows the image with cell numbers drawn on it
% ========================================================
clear; clc;

%% 1. SETTINGS
imgPath  = 'test_image.jpg';      % CHANGE: your whole UAV image
outDir   = 'GridTiles';           % folder where tiles are saved
gridRows = 6;
gridCols = 8;

if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 2. LOAD IMAGE
I = imread(imgPath);
if size(I, 3) == 1, I = repmat(I, [1 1 3]); end
[H, W, ~] = size(I);

rowEdges = round(linspace(0, H, gridRows + 1));
colEdges = round(linspace(0, W, gridCols + 1));

%% 3. SPLIT, SAVE AND ASSIGN GRID IDs
cellID = [];  rowNo = [];  colNo = [];
x1 = [];  y1 = [];  x2 = [];  y2 = [];  fileName = strings(0, 1);

id = 0;
for r = 1:gridRows
    for c = 1:gridCols
        id = id + 1;

        ys = rowEdges(r) + 1;   ye = rowEdges(r + 1);
        xs = colEdges(c) + 1;   xe = colEdges(c + 1);

        tile = I(ys:ye, xs:xe, :);

        fname = sprintf('tile_r%02d_c%02d.jpg', r, c);
        imwrite(tile, fullfile(outDir, fname));

        cellID(end+1, 1) = id;     %#ok<SAGROW>
        rowNo(end+1, 1)  = r;      %#ok<SAGROW>
        colNo(end+1, 1)  = c;      %#ok<SAGROW>
        x1(end+1, 1) = xs;  y1(end+1, 1) = ys;   %#ok<SAGROW>
        x2(end+1, 1) = xe;  y2(end+1, 1) = ye;   %#ok<SAGROW>
        fileName(end+1, 1) = string(fname);      %#ok<SAGROW>
    end
end

gridTable = table(cellID, rowNo, colNo, x1, y1, x2, y2, fileName);
writetable(gridTable, fullfile(outDir, 'gridIndex.csv'));

fprintf('Saved %d tiles in folder "%s"\n', id, outDir);
fprintf('Grid index saved: %s\n', fullfile(outDir, 'gridIndex.csv'));

%% 4. SHOW IMAGE WITH CELL NUMBERS
figure('Name', 'Grid Assignment');
imshow(I); hold on;
for k = 1:height(gridTable)
    rectangle('Position', [x1(k), y1(k), x2(k)-x1(k), y2(k)-y1(k)], ...
        'EdgeColor', 'y', 'LineWidth', 1);
    text((x1(k)+x2(k))/2, (y1(k)+y2(k))/2, ...
        sprintf('%d', cellID(k)), 'Color', 'c', ...
        'FontSize', 10, 'HorizontalAlignment', 'center');
end
title(sprintf('%d x %d grid  (cell ID numbers shown)', gridRows, gridCols));
saveas(gcf, fullfile(outDir, 'gridOverlay.png'));