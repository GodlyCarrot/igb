import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import io

#Initialize a hidden tkinter root window so only the dialogs appear
root = tk.Tk()
root.withdraw()

#asks user to select a .npy file
filename = filedialog.askopenfilename(title="Select MINFLUX data file", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])

if (filename == ""):
    print("No file selected. Exiting script.")
    exit()

#asks user for labeling inputs
user_input = simpledialog.askstring("Labels", "Enter labels separated by commas:", initialvalue="x-axis, y-axis")

if user_input:
    labels = [l.strip() for l in user_input.split(',')]
else:
    labels = ['x-axis', 'y-axis']

#loads data
mfx_data = np.load(filename)
print(f"Successfully loaded: {filename}")

u_tid = np.unique(mfx_data['tid'])
cutoff = 100
common_frequency = np.linspace(0, 200, 1000)
summed_power = np.zeros_like(common_frequency)

#Plots Raw Signal -- asked ai to do this part
plt.figure(figsize=(10, 4))
for T in u_tid:
    mask = mfx_data['tid'] == T
    if np.size(mfx_data['tim'][mask]) > cutoff and \
       np.max(mfx_data['loc'][mask, 0]) - np.min(mfx_data['loc'][mask, 0]) < 1e-7:
        
        t = mfx_data['tim'][mask][cutoff:] - mfx_data['tim'][mask][cutoff]
        locs = mfx_data['loc'][mask, :][cutoff:]
        y = locs[:, 0] - np.mean(locs[:, 0])
        
        if np.size(t) > 1:
            plt.plot(t, y)
            
            #Lomb-Scargle Calculation
            frequency, power = LombScargle(t, y).autopower()
            power_interp = np.interp(common_frequency, frequency, power)
            summed_power += power_interp

#plots graph 1
plt.xlabel(labels[0])#used to be time
plt.ylabel(labels[1]) #used to be signal
plt.title(f"Raw Signal Overlaid: {labels[0]}")
plt.grid(True)
#plt.show() #opens the firstwindow

#plots graph 2
plt.figure(figsize=(10, 5))
plt.plot(common_frequency, summed_power, label="Summed Power Spectrum", color='purple')
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.ylim([0, 0.25])

median_eco = np.round(np.median(mfx_data['eco']))
plt.title(f"Lomb-Scargle Power Spectrum on {labels[0]} (Pho N={median_eco})")
plt.legend()
plt.grid(True)
plt.show() #opens both?