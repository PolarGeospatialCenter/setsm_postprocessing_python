function [named_struct] = test_namedStruct(name, var)

eval(['named_struct.',name,' = var;']);
