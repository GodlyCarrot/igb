import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import matplotlib.backends.backend_tkagg as tkagg
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import io
import pandas as pd
import os

### TO DO:
# scaling the graphs
# export as csv

#Initialize a hidden tkinter root window so only the dialogs appear
root = tk.Tk()
#root.withdraw()
root.title("MINFLUX Fourier Transform")

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
fig, (g1, g2) = plt.subplots(2, 1, figsize=(10, 8))
#plt.figure(figsize=(10, 4))
raw_data_points = []
for T in u_tid:
    mask = mfx_data['tid'] == T
    if np.size(mfx_data['tim'][mask]) > cutoff and \
       np.max(mfx_data['loc'][mask, 0]) - np.min(mfx_data['loc'][mask, 0]) < 1e-7:
        
        t = mfx_data['tim'][mask][cutoff:] - mfx_data['tim'][mask][cutoff]
        locs = mfx_data['loc'][mask, :][cutoff:]
        y = locs[:, 0] - np.mean(locs[:, 0])

        for time_val, signal_val in zip(t, y):
                raw_data_points.append([time_val, signal_val])

        if np.size(t) > 1:
            g1.plot(t, y)
            frequency, power = LombScargle(t, y).autopower()
            summed_power += np.interp(common_frequency, frequency, power)

#plots graph 1
g1.set_xlabel(labels[0])#used to be time
g1.set_ylabel(labels[1]) #used to be signal
g1.set_title(f"Raw Signal Overlaid: {labels[0]}")
g1.grid(True)
#plt.show() #opens the firstwindow

#plots graph 2
#g2.figure(figsize=(10, 5))
g2.plot(common_frequency, summed_power, label="Summed Power Spectrum", color='purple')
#g2.legend()
#plt.show() #opens both?
#g2.plot(common_frequency, summed_power, color='purple')
g2.set_xlabel(labels[0])
g2.set_ylabel(labels[1])
g2.set_ylim([0, 0.25])
median_eco = np.round(np.median(mfx_data['eco']))
g2.set_title(f"Lomb-Scargle Power Spectrum (Pho N={median_eco})")
g2.grid(True)
fig.tight_layout()

canvas = tkagg.FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.draw()

def export_raw_csv():
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save Raw Signal Data")
    if path:
        np.savetxt(path, np.array(raw_data_points), delimiter=",", header=labels[0] + ", " + labels[1], comments='')
        messagebox.showinfo("Export Successful", f"Raw signals saved to:\n{path}")

def export_spectrum_csv():
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save Power Spectrum Data")
    if path:
        data = np.column_stack((common_frequency, summed_power))
        np.savetxt(path, data, delimiter=",", header=labels[0] + ", " + labels[1], comments='')
        messagebox.showinfo("Export Successful", f"Power spectrum saved to:\n{path}")

control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

btn_raw = tk.Button(control_frame, text="Export Raw Signal", command=export_raw_csv, bg="lightgrey")
btn_raw.pack(side=tk.LEFT, padx=5)

btn_spec = tk.Button(control_frame, text="Export Power Spectrum", command=export_spectrum_csv, bg="lightgrey")
btn_spec.pack(side=tk.LEFT, padx=5)

# Add the navigation toolbar (zoom, pan, save) to the same window
toolbar = tkagg.NavigationToolbar2Tk(canvas, root)
toolbar.update()

# Ensure clean exit when the window is closed: close the Matplotlib figure
# and destroy the Tk root so the Python process can terminate. - used ai
def on_closing():
    try:
        plt.close(fig)
    except Exception:
        pass
    try:
        root.quit()
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()