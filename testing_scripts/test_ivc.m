function [singles, pairs] = test_ivc(runnum, imgnum, expected_num)
% Image View & Compare :: A command line program for viewing and comparing indexed or non-indexed (raster or non-raster) TIF images that exist in a test file directory speicifed by test_setGlobals.m.

command_args = ["list"];
preselect = false;
concurrent = false;
if exist('imgnum', 'var')
    command_args = [command_args, "add"];
    preselect = true;
end
if exist('runnum', 'var')
    runnum = num2str(runnum);
    if isempty(runnum)
        runnum = '[]';
    end
    command_args = [command_args, runnum];
end
if exist('imgnum', 'var')
    imgnum = num2str(imgnum);
    if isempty(imgnum)
        imgnum = '[]';
    end
    command_args = [command_args, imgnum];
end
if exist('expected_num', 'var')
    concurrent = true;
    if isempty(expected_num)
        clear expected_num;
    elseif expected_num < 1
        error('Argument expected_num must be greater than or equal to 1.');
    end
end
command = strjoin(command_args);

test_setGlobals();
global TESTDIR;


FILE_COMPARE_WAIT  = [TESTDIR,'/','COMPARE_WAIT'];
FILE_COMPARE_READY = [TESTDIR,'/','COMPARE_READY'];

if concurrent
    fid = fopen(FILE_COMPARE_WAIT, 'wt');
    fclose(fid);
    fid = fopen(FILE_COMPARE_READY, 'wt');
    fclose(fid);
end

fprintf(['' ...
        '\n----- COMMANDS -----\n' ...
        '(browse/list) [runnum] [imgnum]\n' ...
        'add [browse_1] [browse_2] ... [browse_n]\n' ...
        'remove [selection_1] [selection_2] ... [selection_n]\n' ...
        'refresh\n' ...
        'view {raster/image} [selection_1] [selection_2] ... [selection_n]\n' ...
        'compare {raster/image} [selection_1 selection_2]\n' ...
        'auto {raster/image}\n' ...
        'figclose\n' ...
        'quit\n']);
    
indexstr = '';
browseFiles = [];
browse = [];
selection = [];
view = [];
compare = [];
errmsg = '';

ready = false;
while ~ready
    arg_nums = cell2mat(arrayfun(@(x) str2double(char(x)), command_args, 'UniformOutput', false));
    arg_nums(isnan(arg_nums)) = [];
    try
        
        if strcmp(command_args(1), 'figclose')
            close all;
        end
        
        if strcmp(command_args(1), 'refresh')
            if length(command_args) < 2 || (~strcmp(command_args(2), 'browse') && ~strcmp(command_args(2), 'list'))
                browseFiles = dir([TESTDIR,'/',indexstr,'*.tif']);
                browse = setdiff({browseFiles.name}, selection);
            end
            testFiles = dir(TESTDIR);
            selection = intersect(selection, {testFiles.name});
            
            command_args(1) = [];
        end
        
        if strcmp(command_args(1), 'browse') || strcmp(command_args(1), 'list')
            runnum = [];
            imgnum = [];
            arg = 1;
            for i = 2:length(command_args)
                num = str2double(char(command_args(i)));
                if ~isnan(num) || strcmp(command_args(i), '[]')
                    
                    if arg == 1
                        if isnan(num)
                            num = test_getLastRunnum();
                            if isempty(num)
                                break;
                            end
                        end
                        runnum = num;
                        arg = arg + 1;
                        
                    elseif arg == 2
                        if isnan(num)
                            num = test_getNextImgnum(runnum, true, false);
                        end
                        imgnum = num;
                        arg = arg + 1;
                        
                    else
                        fprintf("Ignoring excess number arguments to 'browse' command.\n");
                        break;
                    end
                end
            end
            
            indexstr_run = '';
            indexstr_img = '';
            if ~isempty(runnum)
                indexstr_run = sprintf('run%03d_', runnum);
                if ~isempty(imgnum)
                    indexstr_img = sprintf('%03d_', imgnum);
                end
            end
            
            indexstr = [indexstr_run,indexstr_img];
            browseFiles = dir([TESTDIR,'/',indexstr,'*.tif']);
            browse = setdiff({browseFiles.name}, selection);
            
            command_args(1) = [];
            arg_nums = [];
        end

        if strcmp(command_args(1), 'add') || strcmp(command_args(1), 'remove')
            if strcmp(command_args(1), 'add')
                if isempty(arg_nums)
                    selection = [selection, browse];
                else
                    selection = [selection, browse(arg_nums)];
                end
            elseif strcmp(command_args(1), 'remove')
                if isempty(arg_nums)
                    selection = [];
                else
                    selection(arg_nums) = [];
                end
            end
            
            browse = setdiff({browseFiles.name}, selection);
            
            command_args(1) = [];
        end
        
        if strcmp(command_args(1), 'view') || strcmp(command_args(1), 'compare') || strcmp(command_args(1), 'auto')
            view = [];
            compare = [];
            
            if isempty(selection)
                errmsg = 'No images are selected for viewing/comparing.\n';
                error('CUSTOM MESSAGE');
            else
            
                if strcmp(command_args(1), 'view')
                    if isempty(arg_nums)
                        view_list = selection;
                    else
                        view_list = selection(arg_nums);
                    end
                    view = repmat("", [length(view_list), 2]);
                    view(:,1) = view_list;
                    
                elseif strcmp(command_args(1), 'compare')
                    if isempty(arg_nums)
                        [~, compare] = test_matchFnames(selection);
                    elseif length(arg_nums) == 1
                        errmsg = "Zero or two number arguments must be given to 'compare' command.\n";
                        error('CUSTOM MESSAGE');
                    else
                        compare = repmat("", [1, 2]);
                        compare(1,1) = selection(arg_nums(1));
                        compare(1,2) = selection(arg_nums(2));
                        if length(arg_nums) > 2
                            fprintf("Ignoring excess number arguments to 'compare' command.\n");
                        end
                    end
                    
                elseif strcmp(command_args(1), 'auto')
                    [view, compare] = test_matchFnames(selection);
                end
                
                set_image_type = false;
                image_type = -1;
                if length(command_args) > 1
                    if strcmp(command_args(2), 'image') || strcmp(command_args(2), 'img')
                        image_type = 0;
                    elseif strcmp(command_args(2), 'raster') || strcmp(command_args(2), 'ras')
                        image_type = 1;
                    end
                end
                if image_type ~= -1
                    set_image_type = true;
                end
                
                compare_args = vertcat(view, compare);
                
                compare_total = compare_args.size(1);
                digits = floor(log10(compare_total)) + 1;
                num_format = ['%',num2str(digits),'d'];
                
                fprintf("\n");
                for i = 1:compare_total
                    if strcmp(compare_args(i,2), "")
                        figtitle = compare_args(i,1);
                    else
                        figtitle = test_normalizeTestFname(compare_args(i,1));
                    end
                    
                    if ~set_image_type
                        if (contains(figtitle, '_ras_') || contains(figtitle, 'testRaster_'))
                            image_type = 1;
                        else
                            image_type = 0;
                        end
                    end
                    
                    progress = sprintf(['(',num_format,'/',num_format,')'], i, compare_total);
                    fprintf("Running %s test_compareImages('%s', '%s', '%s', %d)\n", ...
                        progress, compare_args(i,1), compare_args(i,2), figtitle, image_type);
                    test_compareImages(compare_args(i,1), compare_args(i,2), figtitle, image_type);
                end
                
            end
            
            if set_image_type
                command_args(2) = [];
            end
            command_args(1) = [];
        end
        
        if strcmp(command_args(1), 'quit')
            if concurrent && exist(FILE_COMPARE_WAIT, 'file') == 2
                delete(FILE_COMPARE_WAIT);
            end
            while exist(FILE_COMPARE_READY, 'file') == 2
                ;
            end
            return;
        end
        
    catch ME
        if strcmp(ME.message, 'CUSTOM MESSAGE')
        elseif strcmp(ME.message, 'Index exceeds matrix dimensions.') ...
            || strcmp(ME.message, 'Matrix index is out of range for deletion.') ...
            || strcmp(ME.message, 'Subscript indices must either be real positive integers or logicals.')
            if ~isempty(command_args)
                errmsg = 'One or more number arguments is illegal.\n';
            end
        else
            rethrow(ME);
        end
    end
    
    indexstr_display = '(';
    if ~isempty(runnum)
        indexstr_display = [indexstr_display, sprintf('run%03d', runnum)];
        if ~isempty(imgnum)
            indexstr_display = [indexstr_display, sprintf(', img%03d', imgnum)];
        end
    else
        indexstr_display = [indexstr_display, 'all'];
    end
    indexstr_display = [indexstr_display, ')'];
    
    if ~preselect && ~exist('expected_num', 'var')
        fprintf('\n\n--- BROWSE %s ---\n', indexstr_display);
        if isempty(browse)
            fprintf('(empty)\n');
        else
            digits = floor(log10(length(browse))) + 1;
            num_format = ['%-',num2str(digits),'d'];
            for i=1:length(browse)
                fprintf([num_format,': %s\n'], i, string(browse(i)));
            end
        end
    end
        
    if preselect || exist('expected_num', 'var')
        fprintf('\n\n--- SELECTION %s ---\n', indexstr_display);
    else
        fprintf('\n--- SELECTION ---\n');
    end
    if isempty(selection)
        fprintf('(empty)\n');
    else
        digits = floor(log10(length(selection))) + 1;
        num_format = ['%-',num2str(digits),'d'];
        for i=1:length(selection)
            fprintf([num_format,': %s\n'], i, string(selection(i)));
        end
    end
    fprintf('\n');
    
    if preselect
        preselect = false;
    end
    
    if ~strcmp(errmsg, '')
        fprintf(2, errmsg);
        errmsg = '';
    end
    
    if exist('expected_num', 'var')
        if length(selection) ~= expected_num
            fprintf(2, '%d images are expected for comparison.\n', expected_num);
            if length(selection) < expected_num
                fprintf('Press [ENTER] to update selection.\n\n');
            else
                fprintf(2, 'WARNING: More images are selected than expected.\n');
                clear expected_num;
            end
        else
            clear expected_num;
        end
    end
    
    if ~isempty(selection) && ~exist('expected_num', 'var')
        fprintf('(Press [ENTER] to start auto view/compare of selected images)\n\n');
    end
    
    next_command = input('>> ', 's');
    
    if exist('expected_num', 'var')
        if strcmp(next_command, '')
            next_command = command;
        else
            clear expected_num;
        end
    end
    command = next_command;
    
    if ~isempty(selection) && strcmp(command, '')
        command = 'auto';
    end
    
    command_args = string(strsplit(strtrim(command), ' '));
end

[singles, pairs] = test_matchFnames(selection);
