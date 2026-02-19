import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import os
#from . import mst_old
import pyshtools as pysh
import sys
import os
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from mst import mst_old
from mst import mstloop

folder = './mst/trajectory/propagate/'
sys.path.append(os.path.abspath(folder))

def open_input_file():

    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    file_path = pwd + "sb_inputs.py"

    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()

    return content


def write_file(content):

    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    file_path = pwd + "sb_inputs.py"

    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
        #f.flush()
        #os.fsync(f.fileno())
        print("sb_inputs re-writing complete")


def set_Am_factor_input_file(factor):

    original_Am = 0.7854
    new_area = original_Am*factor

    content = open_input_file()
    new_content = re.sub(r"('area'\s*:\s*)\d+(?:\.\d+)?", r"\g<1>" + str(new_area), content)
    write_file(new_content)

    
def set_sma_extra_input_file(extra_a):

    original_sma = 7178.14
    new_sma = original_sma + extra_a

    content = open_input_file()
    new_content = re.sub(r"('a'\s*:\s*)\d+(?:\.\d+)?", r"\g<1>" + str(new_sma), content)
    write_file(new_content)


def set_e_input_file(new_e):

    content = open_input_file()
    new_content = re.sub(r"('e'\s*:\s*)\d+(?:\.\d+)?", r"\g<1>" + str(new_e), content)
    write_file(new_content)

def set_longNode_input_file(o):

    content = open_input_file()
    new_content = re.sub(r"('o'\s*:\s*)\d+(?:\.\d+)?", r"\g<1>" + str(o), content)
    write_file(new_content)


def set_albedo_to_lw():
    
    content = open_input_file()
    new_content = re.sub(r"('albedothermalforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(True), content)
    new_content = re.sub(r"('albedoonlyforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(False), new_content)
    new_content = re.sub(r"('albedoforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(False), new_content)
    write_file(new_content)


def set_albedo_to_sw():
    
    content = open_input_file()
    new_content = re.sub(r"('albedothermalforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(False), content)
    new_content = re.sub(r"('albedoonlyforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(True), new_content)
    new_content = re.sub(r"('albedoforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(False), new_content)
    write_file(new_content)

def set_albedo_to_both():
    content = open_input_file()
    new_content = re.sub(r"('albedothermalforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(False), content)
    new_content = re.sub(r"('albedoonlyforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(False), new_content)
    new_content = re.sub(r"('albedoforce'\s*:\s*)(True|False)", lambda m: m.group(1) + str(True), new_content)
    write_file(new_content)


def get_amag_vec_from_filename(filename):

    df = pd.read_csv(filename, skiprows=5)
    t_vec = np.array(df['Secs_from_Epoch']/3600) # min
    acc_norm = np.array(df[' AMAG'])
    
    return acc_norm, t_vec


def get_lonlat():
    
    df = pd.read_csv('./mst/geodeticht_longlat.csv', skiprows=5)
    t_vec = np.array(df['Secs_from_Epoch']/3600) # min
    lon_vec = np.array(df[' Long'])
    lat_vec = np.array(df[' Lat'])

    return lon_vec, lat_vec, t_vec


def get_arad_vec_from_filename(filename):

    df = pd.read_csv(filename, skiprows=5)
    t_vec = np.array(df['Secs_from_Epoch']/3600) # h
    acc_R = np.array(df[' Ar'])
    
    return acc_R, t_vec


def run_CLI_command(cmd_array, print_output=True):
    
    process = subprocess.Popen(cmd_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if print_output:
        # Read and print output line by line
        for line in process.stdout:
            print(line.strip())

def run_and_make_acc_plot(AM_factor, sma_alt_km, e, figs_folder, plot_num):

    set_Am_factor_input_file(AM_factor)
    set_e_input_file(e)
    set_sma_extra_input_file(sma_alt_km - 800)

    set_albedo_to_sw()
    #run_CLI_command(["mpython", "mstloop.py", "sb_inputs"])
    mstloop.main('.sb_inputs')
    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    
    all_acc_vec = []
    all_R_acc_vec = []

    for acc_name in ['aero', 'srp', 'albedo']:

        filename = pwd + 'csvoutput/' + acc_name + '_accel_2018-01-01.csv'
        acc_norm, t_vec = get_amag_vec_from_filename(filename)
        all_acc_vec.append(acc_norm)

        acc_R, t_vec = get_arad_vec_from_filename(filename)
        all_R_acc_vec.append(acc_R)

    set_albedo_to_lw()
    run_CLI_command(["mpython", "mstloop.py", "sb_inputs"])

    filename = pwd + 'csvoutput/albedo_accel_2018-01-01.csv'
    acc_norm, t_vec = get_amag_vec_from_filename(filename)
    all_acc_vec.append(acc_norm)

    acc_R, t_vec = get_arad_vec_from_filename(filename)
    all_R_acc_vec.append(acc_R)

    all_acc_names = ['Drag', 'SRP', 'Albedo SW', 'Albedo LW']
    title = "a = $R_E$+" + f"{sma_alt_km:.1f}" + "km;      e = " + f"{e:.4f}" + ";      $A/m$ = $\pi$*(0.5m)$^2$/90kg * " + f"{AM_factor:.4f}"
    
    y_label = "Acc. norm (m/s$^2$)"
    fig_name = '/combined_accel_plot_' + str(plot_num) + '.png'
    make_acc_plot(t_vec, all_acc_vec, all_acc_names, y_label, title, 'log', fig_name)
    
    y_label = "Radial acc. (m/s$^2$)"
    fig_name = '/combined_R_accel_plot_' + str(plot_num) + '.png'
    make_acc_plot(t_vec, all_R_acc_vec, all_acc_names, y_label, title, 'symlog', fig_name)

    
    lon_vec, lat_vec, t_vec_gt = get_lonlat()  # groundtracks
    make_groundtrack_plot(lon_vec, lat_vec, figs_folder)


    return all_acc_names, all_acc_vec, all_R_acc_vec


def make_groundtrack_plot(lon_vec, lat_vec, figs_folder):

    #fig, ax_coast = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}) # PlateCarree
    #ax_coast.coastlines()
    #gl = ax_coast.gridlines(draw_labels=True)
    #gl.top_labels = False
    #gl.right_labels = False
    #fig = ax_coast.get_figure()

    new_ax = plt.figure()
    plt.plot(lon_vec, lat_vec)
    
    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    os.makedirs(pwd + '/pngoutput/' + figs_folder, exist_ok=True)
    plt.savefig( pwd + '/pngoutput/' + figs_folder + 'groundtrack.png') #, format='svg' )


def make_acc_plot(t_vec, all_acc_vec, all_acc_names, y_label, title, scale, fig_name):


    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Default Matplotlib colors
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for (acc_norm, acc_name, col) in zip(all_acc_vec, all_acc_names, colors):
        ax1.plot(t_vec, acc_norm, label=acc_name, c=col)

        #ax2.plot(t_vec, acc_R, label=acc_name, c=col)
        #ax.semilogy(t_vec, acc_R, label='_nolegend_', c=col, linestyle='--', alpha=0.5, )

    ax1.plot(t_vec, np.sum(all_acc_vec, axis=0), c='k', label='Total', linewidth=2, linestyle='--')
    #ax1.plot(t_vec, np.sum(all_R_acc_vec, axis=0), c='k', label='Total', linewidth=2, linestyle='--')
    
    if scale=="symlog":
        plt.yscale('symlog', linthresh=1e-12)

    elif scale=="log":
        plt.yscale('log')
        plt.ylim((1e-13, 1e-6))

    plt.xlabel('Time since t_0 (h)')
    plt.ylabel(y_label)
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.2, linestyle='--')
    #plt.ylim((1e-13, 1e-6))
    plt.xlim((t_vec[0], t_vec[-1]))
    plt.title(title)
    plt.tight_layout()

    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    os.makedirs(pwd + '/pngoutput/' + figs_folder, exist_ok=True)
    plt.savefig( pwd + '/pngoutput/' + figs_folder + fig_name) #, format='svg' )


def make_sma_sens_plot(sma_alt_vec, all_RMS_array, all_acc_names, figs_folder, fig_name):
    all_RMS_mat = np.transpose(np.array(all_RMS_array))
    plt.figure()
    for name, RMS_vec in zip(all_acc_names, all_RMS_mat):
        plt.semilogy(sma_alt_vec, RMS_vec, label=name)

    albedo_rad_min_sma = all_RMS_array[0][2]/8
    inv_square_vec = np.array([albedo_rad_min_sma*(6378 + sma_alt_vec[0])**2/((6378 + sma)**2) for sma in sma_alt_vec])
    plt.semilogy(sma_alt_vec, inv_square_vec, label='$\sim 1/r^2$', linestyle='--')

    plt.xlabel('Semimajor axis - $R_E$ (km)')
    plt.ylabel("RMS of acc. norm (m/s$^2$)")
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.2, linestyle='--')
    plt.xlim((sma_alt_vec[0], sma_alt_vec[-1]))
    plt.tight_layout()

    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    plt.savefig(pwd + '/pngoutput/' + figs_folder + fig_name) # 


def make_Am_sens_plot(Am_fact_vec, all_RMS_array, all_acc_names, figs_folder, fig_name):

    all_RMS_mat = np.transpose(np.array(all_RMS_array))
    plt.figure()
    for name, RMS_vec in zip(all_acc_names, all_RMS_mat):
        plt.loglog(Am_fact_vec, RMS_vec, label=name)

    plt.xlabel('(A/m)/(A/m nom.) ()')
    plt.ylabel("RMS of acc. norm (m/s$^2$)")
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.2, linestyle='--')
    plt.xlim((Am_fact_vec[0], Am_fact_vec[-1]))
    plt.tight_layout()

    pwd = '/home/llbu9955/Space-Balls-MST/sbfinal-master/mst/'
    plt.savefig(pwd + '/pngoutput/' + figs_folder + fig_name) # '/Am_sensitivity_plot.png'


if __name__ == "__main__":

    # Nominal settings:
    AM_factor = 1
    sma_alt = 800
    e = 0

    # sma sensitivity analysis:

    n = 2
    AM_factor = 1
    # sma_alt_vec = np.linspace(500, 5000, n)
    sma_alt_vec = np.logspace(np.log10(500), np.log10(5000), n)
    print(sma_alt_vec)
    all_RMS_array = [None] * n
    all_RMS_rad_array = [None] * n
    figs_folder = 'sma_sensitivity'

    for i, sma_alt in enumerate(sma_alt_vec):

        print(AM_factor)
        print(sma_alt)

        all_acc_names, all_acc_vec, all_R_acc_vec = run_and_make_acc_plot(AM_factor, sma_alt, e, figs_folder, i)
        
        all_RMS = [np.sqrt(np.mean(arr**2)) for arr in all_acc_vec]
        all_RMS_array[i] = all_RMS
        
        all_RMS_rad = [np.sqrt(np.mean(arr**2)) for arr in all_R_acc_vec]
        all_RMS_rad_array[i] = all_RMS_rad

    make_sma_sens_plot(sma_alt_vec, all_RMS_array, all_acc_names, figs_folder, '/sma_sensitivity_plot.png')
    make_sma_sens_plot(sma_alt_vec, all_RMS_rad_array, all_acc_names, figs_folder, '/sma_sensitivity_plot_radial.png')


    # A/m sensitivity analysis:
    sma_alt = 800

    n = 2
    Am_fact_vec = np.logspace(-2, 2, n)
    all_RMS_array = [None] * n
    all_RMS_rad_array = [None] * n
    figs_folder = 'Am_sensitivity'

    for i, AM_factor in enumerate(Am_fact_vec):

        all_acc_names, all_acc_vec, all_R_acc_vec = run_and_make_acc_plot(AM_factor, sma_alt, e, figs_folder, i)
            
        all_RMS = [np.sqrt(np.mean(arr**2)) for arr in all_acc_vec]
        all_RMS_array[i] = all_RMS
        
        all_RMS_rad = [np.sqrt(np.mean(arr**2)) for arr in all_R_acc_vec]
        all_RMS_rad_array[i] = all_RMS_rad

    make_Am_sens_plot(Am_fact_vec, all_RMS_array, all_acc_names, figs_folder, '/Am_sensitivity_plot.png')
    make_Am_sens_plot(Am_fact_vec, all_RMS_rad_array, all_acc_names, figs_folder, '/Am_sensitivity_plot_radial.png')





