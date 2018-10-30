function [path_full] = test_validateTestFileSave(path, allow_existing)

if ~exist('allow_existing', 'var') || isempty(allow_existing)
    allow_existing = false;
end

test_setGlobals();
global TESTDIR;


[path_dir, path_fname, path_ext] = fileparts(path);
path_basename = [path_fname, path_ext];
if strcmp(path_basename, path)
    path_full = fullfile(TESTDIR, path);
else
    path_full = path;
end

if ~allow_existing
    while exist(path_full, 'file') == 2
        opt = input(sprintf('Test file "%s" already exists. Overwrite/append? (y/n): ', strrep(path_full, TESTDIR, '{TESTDIR}')), 's');
        if strcmpi(opt, 'y')
            break;
        else
            opt = input('Append description to filename (or press [ENTER] to cancel): ', 's');
            if isempty(opt)
                path_full = [];
                return;
            else
                path_fname_root = path_fname;
                path_full = sprintf('%s~%s%s', path_fname_root, strrep(opt,' ','-'), path_ext);
            end
        end
    end
end
