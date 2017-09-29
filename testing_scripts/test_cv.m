% test_cv :: test_checkVars

array_vars = [
        "x", "y", "z", "m", "o", "md", "-", ...
        "X", "Y", "Z", "M", "O", "-", ...
        "Xsub", "Ysub", "Zsub", "Msub", "Osub"
    ];

fprintf('\n');
for i=1:length(array_vars)
    var = char(array_vars(i));
    if exist(var, 'var')
        
        expr = sprintf('class(%s)', var);
        dtype = eval(expr);
        if strcmp(dtype, 'single')
            dtype = 'float32';
        elseif strcmp(dtype, 'double')
            dtype = 'float64';
        end
        fprintf(2, '> %s.dtype = %s\n', var, dtype);
        
        expr = sprintf('size(%s)', var);
        shape = eval(expr);
        fprintf('    shape = (%d, %d)\n', shape(1), shape(2));
        
        expr = sprintf('min(%s(:))', var);
        fprintf('    min = %f\n', eval(expr));
        
        expr = sprintf('max(%s(:))', var);
        fprintf('    max = %f\n', eval(expr));
        
    elseif strcmp(var, '-')
        fprintf('------------------\n');
    end
end
fprintf('\n');
