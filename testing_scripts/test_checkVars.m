array_vars = ["x", "y", "z", "m", "o", "md", ...
        "X", "Y", "Z", "M", "O"];

fprintf('\n');
for i=1:length(array_vars)
    var = char(array_vars(i));
    if exist(var, 'var')
        
        expr = sprintf('class(%s)', var);
        fprintf(2, '> %s.dtype = %s\n', var, eval(expr));
        
        expr = sprintf('min(%s(:))', var);
        fprintf('    min = %f\n', eval(expr));
        
        expr = sprintf('max(%s(:))', var);
        fprintf('    max = %f\n', eval(expr));
        
    end
end
fprintf('\n');
