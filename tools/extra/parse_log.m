function [iter,top1,top5] = parse_log(txtName)
% txtName = 'log_88.6.txt.test';
f = fopen(txtName,'r');
fgetl(f);
A = fscanf(f,'%d %f %f %f %f %f %f %f');

iter = A(1:8:end);
top1 = A(3:8:end);
top5 = A(4:8:end);
len =length(top1);
iter = iter(1:len);
end