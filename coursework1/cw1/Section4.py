# do the experiments on Dropout, L1, L2, Label Smoothing separately

# Dropout

# L1

# L2

# Label Smoothing

# Fill Table3

# Find italic and bold

# Plot Figure4 with Known Value from Table3

# do other experiments on different (p, lambda) for L1 and (p, lambda) for L2

from matplotlib import pyplot as plt

from Experiment import Exp_Dropout, Exp_L1, Exp_L2, Exp_Dropout_L1, Exp_Dropout_L2, Exp_Label_Smoothing

done1 = True
done2 = True
done3 = True
done4 = True

# Fill Table3
# Dropout, p = 0.7
if not done1:
    experiment = Exp_Dropout()
    experiment(prob=0.7)

# L1, Lambda = 1e-3
if not done2:
    experiment = Exp_L1()
    experiment(lamda=1e-3)

# L2, Lambda = 1e-3
if not done3:
    experiment = Exp_L2()
    experiment(lamda=1e-3)

# Label Smoothing, alpha = 0.1
if not done4:
    experiment = Exp_Label_Smoothing()
    experiment()

# Find italic and Bold

# Plot Figure4 munually
fig_1 = plt.figure(figsize=(7, 7))
ax_1 = fig_1.add_subplot(111)

X_dropout = [0.6, 0.7, 0.85, 0.97]
Y_val_acc = [80.7, 83.2, 85.1, 85.4]
Y_gap = [0.593 - 0.549, 0.504 - 0.444, 0.434 - 0.329, 0.457 - 0.244]

line1 = ax_1.plot(X_dropout, Y_val_acc, label="Val. Acc.", linestyle="-")

_ax_1 = ax_1.twinx()
line2 = _ax_1.plot(X_dropout, Y_gap, label="Gap", linestyle="-", color="#348ABD")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax_1.legend(lns, labs, loc=0)

ax_1.set_xlabel('Dropout value')
ax_1.set_ylabel('Accuracy')
_ax_1.set_ylabel('Generalization gap')

# plt.savefig("./report/figures/Task2_dropout_plot.pdf")

fig_2 = plt.figure(figsize=(6, 6))
ax_2 = fig_2.add_subplot(111)

X_wd = [5e-4, 1e-3, 5e-3, 5e-2]
Y_val_acc_L1 = [79.5, 75.1, 2.41, 2.20]
Y_gap_L1 = [0.658 - 0.642, 0.849 - 0.841, 3.850 - 3.850, 3.850 - 3.850]
Y_val_acc_L2 = [85.1, 85.0, 81.3, 39.2]
Y_gap_L2 = [0.460 - 0.306, 0.456 - 0.361, 0.607 - 0.586, 2.256 - 2.258]

line1 = ax_2.plot(X_wd, Y_val_acc_L1, label="L1 Val. Acc.", linestyle="-")
line2 = ax_2.plot(X_wd, Y_val_acc_L2, label="L2 Val. Acc.", linestyle="-")

_ax_2 = ax_2.twinx()
line3 = _ax_2.plot(X_wd, Y_gap_L1, label="L1 Gap", linestyle="--")
line4 = _ax_2.plot(X_wd, Y_gap_L2, label="L2 Gap", linestyle="--")
        
lns = line1 + line2 + line3 + line4
labs = [l.get_label() for l in lns]
ax_2.legend(lns, labs, loc=0)

ax_2.set_xscale('log')
ax_2.set_xlabel('Weight decay value')
ax_2.set_ylabel('Accuracy')
_ax_2.set_ylabel('Generalization gap')

# plt.savefig("./report/figures/Task2_wd_plot.pdf")

# Do other experiments (p, lambda) 
