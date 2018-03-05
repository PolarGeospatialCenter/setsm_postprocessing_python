% Check Vars ***to be executed while debugging ***

cv_test_vars = [];
cv_test_var = [];
cv_test_expr = [];
cv_shape = [];
cv_dtype = [];
cv_i = [];

cv_test_vars = [
        "x", "y", "z", "m", "o", "md", "-", ...
        "X", "Y", "Z", "M", "O", "-", ...
        "Xsub", "Ysub", "Zsub", "Msub", "Osub"
    ];

fprintf('\n');
for cv_i=1:length(cv_test_vars)
    cv_test_var = char(cv_test_vars(cv_i));
    if exist(cv_test_var, 'var')
        
        cv_test_expr = sprintf('class(%s)', cv_test_var);
        cv_dtype = eval(cv_test_expr);
        if strcmp(cv_dtype, 'single')
            cv_dtype = 'float32';
        elseif strcmp(cv_dtype, 'double')
            cv_dtype = 'float64';
        end
        fprintf(2, '> %s.dtype = %s\n', cv_test_var, cv_dtype);
        
        cv_test_expr = sprintf('size(%s)', cv_test_var);
        cv_shape = eval(cv_test_expr);
        fprintf('    shape = (%d, %d)\n', cv_shape(1), cv_shape(2));
        
        cv_test_expr = sprintf('min(%s(:))', cv_test_var);
        fprintf('    min = %f\n', eval(cv_test_expr));
        
        cv_test_expr = sprintf('max(%s(:))', cv_test_var);
        fprintf('    max = %f\n', eval(cv_test_expr));
        
    elseif strcmp(cv_test_var, '-')
        fprintf('------------------\n');
    end
end
fprintf('\n');

clear('cv_test_vars', 'cv_test_var', 'cv_test_expr', 'cv_shape', 'cv_dtype', 'cv_i');
