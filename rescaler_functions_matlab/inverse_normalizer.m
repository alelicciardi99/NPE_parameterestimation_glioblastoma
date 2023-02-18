function inv_val = inverse_normalizer(val,range)
% range is a 2-elements vector
% range(1) is the min value
% range(2) is the max value
% val is a single value, or a vector in [0,1]
inv_val=(range(2)-range(1))*val+range(1);
end

