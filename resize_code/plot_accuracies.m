a_all = [0.0005356596265687174, 0.04438322620140802, 0.001147842056932966, 0.03168044077134986, 0.02288031833486379, 0.018901132537496174, 0.10529537802265075, 0.10261707988980716];
names_all = {'Random', 'KNN', 'NB', 'RF', 'Simple CNN1', 'Simple CNN2', 'AE', 'MLP'};

a_single1 = [0.18947046219773492, 0.6502142638506275, 0.3525405570860116, 0.37894092439546984, 0.2209213345576982, 0.22053872053872053, 0.7075298438934803, 0.89, 0.91];
names_single = {'Random', 'KNN', 'NB', 'RF', 'Simple CNN1', 'Simple CNN2', 'MLP', 'Resnet', 'SI Resnet'};

%%
[Y, I] = sort(a_all, 'descend');
names_s = names_all(I);
X = categorical(names_s, names_s);

bar(X, Y*100)
ylabel('Accuracy')
saveas(gcf, '../images/accuracy_all.png')
close gcf

%%
[Y, I] = sort(a_single1, 'descend');
names_s = names_single(I);
X = categorical(names_s, names_s);

bar(X, Y*100)
ylabel('Accuracy')
saveas(gcf, '../images/accuracy_single.png')
close gcf