import pandas as pd
import matplotlib.pyplot as plt

pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'

all_acc_names = ['aero', 'srp', 'albedo']
all_acc_vecs = []

for acc_name in all_acc_names:

    filename = pwd + 'csvoutput/' + acc_name + '_accel_2015-01-01.csv'
    df = pd.read_csv(filename, skiprows=5)
    t_vec = df['Secs_from_Epoch']/3600 # min
    acc_vec = df[' AMAG']
    all_acc_vecs.append(acc_vec)
    plt.semilogy(t_vec, acc_vec, label=acc_name)

plt.xlabel('Time since t_0 (h)')
plt.ylabel("Acc. norm (m/s^2)")
plt.legend()
plt.grid()
plt.savefig( pwd + '/pngoutput/' + 'combined_accel_plot.png') #, format='svg' )
#plt.show()


print('Done!')