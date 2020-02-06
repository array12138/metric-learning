load('newTwomoons.mat');
data = newdata;
label = newlabel;
oneClass = find(label==1);
twoClass = find(label==2);

hold on;
plot(data(oneClass(:),1),data(oneClass(:),2),'g.');
plot(data(twoClass(:),1),data(twoClass(:),2),'y.');
plot(U_matrix(oneClass(:),1),U_matrix(oneClass(:),2),'ko');
plot(U_matrix(twoClass(:),1),U_matrix(twoClass(:),2),'bo');
