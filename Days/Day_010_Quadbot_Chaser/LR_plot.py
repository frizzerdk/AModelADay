import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import LogLocator, FuncFormatter

def warmup_linear_schedule(initial_lr: float, peak_lr: float, final_lr: float, warmup_fraction: float = 0.1):
    def schedule(progress_remaining: float) -> float:
        if progress_remaining > 1 - warmup_fraction:
            # Warmup phase
            warmup_progress = (progress_remaining - (1 - warmup_fraction)) / warmup_fraction
            return initial_lr + (peak_lr - initial_lr) * (1 - warmup_progress)
        else:
            # Linear decay phase
            return final_lr + (progress_remaining / (1 - warmup_fraction)) * (peak_lr - final_lr)
    return schedule

# Initial parameters
initial_lr = 3e-5
peak_lr = 3e-4
final_lr = 3e-5
warmup_fraction = 0.1

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.15, bottom=0.3)

# Generate initial plot
progress_values = np.linspace(1, 0, 1000)
lr_schedule = warmup_linear_schedule(initial_lr, peak_lr, final_lr, warmup_fraction)
lr_values = [lr_schedule(p) for p in progress_values]
line, = ax.plot(1 - progress_values, lr_values)

# Set up the plot
ax.set_title('Learning Rate Schedule')
ax.set_xlabel('Training Progress')
ax.set_ylabel('Learning Rate')
ax.set_yscale('log')
ax.grid(True)

# Custom y-axis tick locator and formatter
y_locator = LogLocator(base=10, numticks=20)
y_formatter = FuncFormatter(lambda y, _: f'{y:.1e}')
ax.yaxis.set_major_locator(y_locator)
ax.yaxis.set_major_formatter(y_formatter)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

# Create sliders
slider_color = 'lightgoldenrodyellow'
slider_ax_initial = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=slider_color)
slider_ax_peak = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=slider_color)
slider_ax_final = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=slider_color)
slider_ax_warmup = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=slider_color)

slider_initial = Slider(slider_ax_initial, 'Initial LR', 1e-6, 1e-3, valinit=initial_lr, valstep=1e-6)
slider_peak = Slider(slider_ax_peak, 'Peak LR', 1e-5, 1e-2, valinit=peak_lr, valstep=1e-5)
slider_final = Slider(slider_ax_final, 'Final LR', 1e-6, 1e-3, valinit=final_lr, valstep=1e-6)
slider_warmup = Slider(slider_ax_warmup, 'Warmup Fraction', 0.01, 0.5, valinit=warmup_fraction, valstep=0.01)

# Update function
def update(val):
    new_initial_lr = slider_initial.val
    new_peak_lr = slider_peak.val
    new_final_lr = slider_final.val
    new_warmup_fraction = slider_warmup.val
    
    new_lr_schedule = warmup_linear_schedule(new_initial_lr, new_peak_lr, new_final_lr, new_warmup_fraction)
    new_lr_values = [new_lr_schedule(p) for p in progress_values]
    
    line.set_ydata(new_lr_values)
    ax.relim()
    ax.autoscale_view()
    
    # Update y-axis limits
    ax.set_ylim(min(new_lr_values) * 0.9, max(new_lr_values) * 1.1)
    
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_initial.on_changed(update)
slider_peak.on_changed(update)
slider_final.on_changed(update)
slider_warmup.on_changed(update)

# Add a reset button
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color=slider_color, hovercolor='0.975')

def reset(event):
    slider_initial.reset()
    slider_peak.reset()
    slider_final.reset()
    slider_warmup.reset()

reset_button.on_clicked(reset)

plt.show()
