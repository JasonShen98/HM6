close all; clear all; clc
A1 = [2;10];
A2 = [2;5];
A3 = [8;4];
A4 = [5;8];
A5 = [7;5];
A6 = [6;4];
A7 = [1;2];
A8 = [4;9];
Map = [A1,A2,A3,A4,A5,A6,A7,A8];
init_cluster = [A3,A5,A6];
iter_count = 0;
old_cluster = init_cluster;
new_cluster = ucl(init_cluster,Map);
while (~isequal(old_cluster,new_cluster))
    iter_count = iter_count + 1;
    old_cluster = new_cluster;
    new_cluster = ucl(new_cluster,Map);
end
disp(iter_count)
disp(new_cluster)

function cal_d = d(i,j)
x1 = i(1);
y1 = i(2);
x2 = j(1);
y2 = j(2);
cal_d = sqrt((x1-x2)^2 + (y1-y2)^2);
end

function Assignment = assign(C,M,i)
c1 = C(:,1);
c2 = C(:,2);
c3 = C(:,3);
p = M(:,i);
D = [d(p,c1),d(p,c2),d(p,c3)];
[m,c] = min(D);
Assignment = c;
end

function UpdateCluster = ucl(C,M)
Assign = zeros(1,8);
for i=1:8
    Assign(i) = assign(C,M,i);
end
c = zeros(2,3);
cout_c = zeros(1,3);
for i = 1:8
    if (Assign(i) == 1)
        c(1,1) = c(1,1) + M(1,i);
        c(2,1) = c(2,1) + M(2,i);
        cout_c(1) = cout_c(1) + 1;
    elseif (Assign(i) == 2)
        c(1,2) = c(1,2) + M(1,i);
        c(2,2) = c(2,2) + M(2,i);
        cout_c(2) = cout_c(2) + 1;
    elseif (Assign(i) == 3)
        c(1,3) = c(1,3) + M(1,i);
        c(2,3) = c(2,3) + M(2,i);
        cout_c(3) = cout_c(3) + 1;
    end
end
c(1,1) = c(1,1)/cout_c(1);
c(2,1) = c(2,1)/cout_c(1);
c(1,2) = c(1,2)/cout_c(2);
c(2,2) = c(2,2)/cout_c(2);
c(1,3) = c(1,3)/cout_c(3);
c(2,3) = c(2,3)/cout_c(3);
UpdateCluster = c;
end