function [singles, pairs] = test_matchFnames(full_fnames)

norm_fnames = arrayfun(@(fname) test_normalizeTestFname(fname), full_fnames, 'UniformOutput', false);
    
match_markers = zeros(1, length(norm_fnames));
matchnum = 1;
for i = 1:length(norm_fnames)
    if ~isempty(cell2mat(norm_fnames(i)))
        for j = (i-1):-1:1
            if strcmp(norm_fnames(i), norm_fnames(j))
                match_markers(i) = match_markers(j);
                break;
            end
        end
        if match_markers(i) == 0
            match_markers(i) = matchnum;
            matchnum = matchnum + 1;
        end
    end
end

match_fnames = repmat("", [max(match_markers), 2]);
offset = 0;
for matchnum = 1:max(match_markers)
    match = full_fnames(match_markers == matchnum);
    if length(match) == 1
        match_fnames(matchnum+offset,1) = match;
    elseif length(match) == 2
        match_fnames(matchnum+offset,:) = match;
    else
        combos = combnk(match, 2);
        extra_entries = length(combos) - 1;
        match_fnames = vertcat(match_fnames, repmat("", [extra_entries, 2]));
        for i = 0:extra_entries
            match_fnames(matchnum+offset+i,:) = combos(i+1,:);
        end
        offset = offset + extra_entries;
    end
end

single_markers = cell2mat(arrayfun(@(fname) strcmp(char(fname), ""), match_fnames(:,2), 'UniformOutput', false));
singles = match_fnames(single_markers,:);
pairs = match_fnames(~single_markers,:);
