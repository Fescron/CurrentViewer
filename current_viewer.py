#!/usr/bin/env python
# Copyright (c) Marius Gheorghescu. All rights reserved.
# Modifications added by Brecht Van Eeckhoudt
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
import sys
import time
import serial
import logging
from logging.handlers import RotatingFileHandler
import argparse
import platform
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.animation as animation
from matplotlib.dates import num2date, DateFormatter
from matplotlib.widgets import Button
from datetime import datetime, timedelta
from threading import Thread
from os import path
from matplotlib.ticker import EngFormatter, FuncFormatter


version = "1.1.1-BVE"
connected_device = "CurrentRanger"

# Default serial connection settings
port = ""
baud = 115200

# Default plot refresh interval
refresh_interval = 66 # 66ms = 15fps

# Controls the default window size (and memory usage). 100k samples = 3 minutes
buffer_max_samples = 100000

# Controls how many samples to display (by default) in the chart (and CPU usage). Ie 4k display should be ok with 2k samples
chart_max_samples = 2048

# Set to "True" to compute median instead of average (less noise, more CPU)
median_filter = False

# How many samples to average (median)
max_supersampling = 16

# Global variables used for storing samples to a file
save_file = None
save_format = None

# Default logfile size
log_size_bytes = 1024*1024

# Global variables used for the linear current scaling
linear_current_axis = False
autoscale_current = True

# Set to "False" to default to the dark theme
light_theme = True

# Used to print the total chart length on the plot
chart_length_s = 0

# Set to "False" to display the program name and GitHub repository on the plot
hide_info_on_plot = True


class CRPlot:
    """
    A class for managing live data streaming and plotting from a CurrentRanger device.

    Attributes:
    - port: The serial port to connect to.
    - baud: The baud rate for the serial connection.
    - thread: The thread for streaming data.
    - stream_data: A flag indicating whether data streaming is active.
    - pause_chart: A flag indicating whether the chart is paused.
    - sample_count: The number of samples received.
    - animation_index: The index for saving animations.
    - max_samples: The maximum number of samples to store in memory.
    - data: A deque for storing current values.
    - timestamps: A deque for storing timestamps.
    - dataStartTS: The timestamp when data streaming started.
    - serialConnection: The serial connection object.
    - framerate: The framerate for animations.
    """

    def __init__(self, sample_buffer = 100):
        """
        Initializes the CRPlot object with default settings.

        Parameters:
        - sample_buffer: The maximum number of samples to store in memory.
        """
        self.port = "/dev/ttyACM0"
        self.baud = 9600
        self.thread = None
        self.stream_data = True
        self.pause_chart = False
        self.sample_count = 0
        self.animation_index = 0
        self.max_samples = sample_buffer
        self.data = collections.deque(maxlen=sample_buffer)
        self.timestamps = collections.deque(maxlen=sample_buffer)
        self.dataStartTS = None
        self.serialConnection = None
        self.framerate = 30

    def serialStart(self, port, speed = 115200):
        """
        Starts the serial connection and data streaming.

        Parameters:
        - port: The serial port to connect to.
        - speed: The baud rate for the serial connection.

        Returns:
        True if the connection is successful, False otherwise.
        """
        self.port = port
        self.baud = speed
        logging.info(f"Trying to connect to port='{port}' with baud='{speed}'")
        try:
            self.serialConnection = serial.Serial(self.port, self.baud, timeout=5)
            logging.info(f"Connected to {port} at {speed} baud")
        except serial.SerialException as e:
            logging.error(f"Error connecting to serial port: {e}")
            return False
        except:
            logging.error(f"Error connecting to serial port, unexpected exception:{sys.exc_info()}")
            return False

        if self.thread == None:
            self.thread = Thread(target=self.serialStream)
            self.thread.start()

            print("Initializing data capture:", end="")
            wait_timeout = 100
            while wait_timeout > 0 and self.sample_count == 0:
                print(".", end="", flush=True)
                time.sleep(0.01)
                wait_timeout -= 1

            if (self.sample_count == 0):
                logging.error("Error: No data samples received. Aborting")
                return False

            print("OK\n")
            return True

    def pauseRefresh(self, state):
        """
        Toggles the pause state of the chart.

        Parameters:
        - state: The current state of the pause button.

        This function updates the chart title and button label based on the pause state.
        """
        logging.debug(f"pause {state}")
        self.pause_chart = not self.pause_chart
        if self.pause_chart:
            if not light_theme:
                self.ax.set_title("<Paused>", color="yellow")
            else:
                self.ax.set_title("<Paused>")
            self.bpause.label.set_text("Resume")
        else:
            if not light_theme:
                self.ax.set_title(f"Streaming: {connected_device}", color="white")
            else:
                self.ax.set_title(f"Streaming: {connected_device}")
            self.bpause.label.set_text("Pause")

    def saveAnimation(self, state):
        """
        Saves the current animation as a GIF file.

        Parameters:
        - state: The current state of the save button.

        This function starts a background thread to save the animation to a file.
        """
        logging.debug(f"save {state}")

        def save_gif():
            self.bsave.label.set_text("Saving...")
            plt.gcf().canvas.draw()
            filename = None
            while True:
                filename = f"current{self.animation_index}.gif"
                self.animation_index += 1
                if not path.exists(filename):
                    break
            try:
                logging.info(f"Animation saving started to '{filename}'")
                self.anim.save(filename, writer="imagemagick", fps=self.framerate)
                logging.info("Animation saved to '{filename}'")
            except Exception as e:
                logging.error(f"Failed to save animation: {e}")
            finally:
                self.bsave.label.set_text("GIF")
                plt.gcf().canvas.draw()

        Thread(target=save_gif, daemon=True).start()

    def toggle_autoscale_current(self, state):
        """
        Toggles the autoscale mode for the current axis.

        Parameters:
        - state: The current state of the autoscale button.

        This function switches between manual and automatic scaling for the y-axis.
        """
        global autoscale_current
        autoscale_current = not autoscale_current
        toolbar = plt.get_current_fig_manager().toolbar
        if autoscale_current:
            self.b_autoscale_current.label.set_text("Manual\nCurr. Scale")
            if toolbar.mode == "zoom rect":
                toolbar.zoom() # Unselect the zoom tool
        else:
            self.b_autoscale_current.label.set_text("Automatic\nCurr. Scale")
            if toolbar.mode != "zoom rect":
                toolbar.zoom() # Select the zoom tool (because it is not already selected)

    def chartSetup(self, refresh_interval=100):
        """
        Sets up the live chart for streaming data.

        Parameters:
        - refresh_interval: The refresh interval for the chart in milliseconds.

        This function initializes the matplotlib figure, axes, and buttons for
        interacting with the live chart.
        """
        if not light_theme:
            plt.style.use("dark_background")

        fig = plt.figure(num=f"CurrentViewer v{version}", figsize=(10, 6))
        self.ax = plt.axes()
        ax = self.ax

        setup_plot_style(ax, fig, title=f"Streaming: {connected_device}")

        ax.set_xlim(datetime.now(), datetime.now() + timedelta(seconds=10))

        lines = ax.plot([], [], label="Current")[0]

        lastText = ax.text(0.50, 0.95, "", transform=ax.transAxes)
        self.anim = animation.FuncAnimation(fig, self.getSerialData, fargs=(lines, plt.legend(), lastText), interval=refresh_interval, cache_frame_data=False)

        apause = plt.axes([0.91, 0.15, 0.08, 0.07])
        if not light_theme:
            self.bpause = Button(apause, label="Pause", color="0.2", hovercolor="0.1")
            self.bpause.label.set_color("yellow")
        else:
            self.bpause = Button(apause, label="Pause")
        self.bpause.on_clicked(self.pauseRefresh)

        aanimation = plt.axes([0.91, 0.25, 0.08, 0.07])
        if not light_theme:
            self.bsave = Button(aanimation, "GIF", color="0.2", hovercolor="0.1")
            self.bsave.label.set_color("yellow")
        else:
            self.bsave = Button(aanimation, "GIF")
        self.bsave.on_clicked(self.saveAnimation)

        if linear_current_axis:
            a_autoscale_current = plt.axes([0.91, 0.35, 0.08, 0.07])
            if not light_theme:
                self.b_autoscale_current = Button(a_autoscale_current, "Manual\nCurr. Scale", color="0.2", hovercolor="0.1")
                self.b_autoscale_current.label.set_color("yellow")
            else:
                self.b_autoscale_current = Button(a_autoscale_current, "Manual\nCurr. Scale")
            self.b_autoscale_current.on_clicked(self.toggle_autoscale_current)

        crs = mplcursors.cursor(ax, hover=True)
        @crs.connect("add")
        def _(sel):
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="yellow", alpha=.4)
            sel.annotation.set_text(textAmp(sel.target[1]))

        self.framerate = 1000/refresh_interval
        plt.gcf().autofmt_xdate()
        plt.show()

    def serialStream(self):
        """
        Reads data from the serial connection and processes it.

        This function runs in a background thread and continuously reads data
        from the serial port, parses it, and stores it in memory for plotting.
        """
        # Set data streaming mode on CurrentRanger (assuming it was off)
        self.serialConnection.write(b"u")

        self.serialConnection.reset_input_buffer()
        self.sample_count = 0
        line_count = 0
        error_count = 0
        self.dataStartTS = datetime.now()

        # Data timeout threshold (seconds) - bails out of no samples received
        data_timeout_ths = 0.5

        line = None
        device_data = bytearray()

        logging.info("Starting USB streaming loop")

        while (self.stream_data):
            try:
                # Get the timestamp before the data string, likely to align better with the actual reading
                ts = datetime.now()

                chunk_len = device_data.find(b"\n")
                if chunk_len >= 0:
                    line = device_data[:chunk_len]
                    device_data = device_data[chunk_len+1:]
                else:
                    line = None
                    while line == None and self.stream_data:
                        chunk_len = max(1, min(4096, self.serialConnection.in_waiting))
                        chunk = self.serialConnection.read(chunk_len)
                        chunk_len = chunk.find(b"\n")
                        if chunk_len >= 0:
                            line = device_data + chunk[:chunk_len]
                            device_data[0:] = chunk[chunk_len+1:]
                        else:
                            device_data.extend(chunk)

                if line == None:
                    continue

                line = line.decode(encoding="ascii", errors="strict")

                if (line.startswith("USB_LOGGING")):
                    if (line.startswith("USB_LOGGING_DISABLED")):
                        # Must have been left open by a different process/instance
                        logging.info("CurrentRanger USB Logging was disabled. Re-enabling")
                        self.serialConnection.write(b"u")
                        self.serialConnection.flush()
                    continue

                data = float(line)
                self.sample_count += 1
                line_count += 1

                if save_file:
                    if save_format == "CSV":
                        save_file.write(f"{ts},{data}\n")
                    elif save_format == "JSON":
                        save_file.write("{}{{\"time\":\"{}\",\"current\":\"{}\"}}".format(",\n" if self.sample_count>1 else "", ts, data))

                if data < 0.0: # TODO Allow negative values with a new mode
                    # This happens too often (negative values)
                    self.timestamps.append(np.datetime64(ts))
                    self.data.append(1.0e-11)
                    logging.debug(f"Unexpected value='{line.strip()}'")
                else:
                    self.timestamps.append(np.datetime64(ts))
                    self.data.append(data)
                    logging.debug(f"#{self.sample_count}:{ts}: {data}")

                if (self.sample_count % 1000 == 0):
                    logging.debug(f"{ts.strftime('%H:%M:%S.%f')}: '{line.rstrip()}' -> {data}")
                    dt = datetime.now() - self.dataStartTS
                    logging.info(f"Received {self.sample_count} samples in {1000*dt.total_seconds():.0f}ms ({self.sample_count/dt.total_seconds():.2f} samples/second)")
                    print(f"Received {self.sample_count} samples in {1000*dt.total_seconds():.0f}ms ({self.sample_count/dt.total_seconds():.2f} samples/second)")

            except KeyboardInterrupt:
                logging.info("Terminated by user")
                break

            except ValueError:
                logging.error(f"Invalid data format: '{line}': {sys.exc_info()}")
                error_count += 1
                last_sample = (np.datetime64(datetime.now()) - (self.timestamps[-1] if self.sample_count else np.datetime64(datetime.now())))/np.timedelta64(1, "s")
                if (error_count > 100) and  last_sample > data_timeout_ths:
                    logging.error(f"Aborting. Error rate is too high ({error_count} errors), last valid sample received {last_sample} seconds ago")
                    self.stream_data = False
                    break
                pass

            except serial.SerialException as e:
                logging.error(f"Serial read error: {e.strerror}: {sys.exc_info()}")
                self.stream_data = False
                break

        self.stream_data = False

        # Stop streaming so the device shuts down if in auto mode
        logging.info("Telling CurrentRanger to stop USB streaming")

        try:
            # This will throw if the device has failed.disconnected already
            self.serialConnection.write(b"u")
        except:
            logging.warning("Was not able to clean disconnect from the device")

        logging.info("Serial streaming terminated")

    def getSerialData(self, frame, lines, legend, lastText):
        """
        Updates the chart with the latest data.

        Parameters:
        - frame: The current frame of the animation.
        - lines: The matplotlib Line2D object for the data line.
        - legend: The legend object for the chart.
        - lastText: The text object for displaying the sample rate.

        This function processes the data and updates the chart elements.
        """
        global autoscale_current

        if (self.pause_chart or len(self.data) < 2):
            lastText.set_text("")
            return

        if not self.stream_data:
            self.ax.set_title("<Disconnected>", color="red")
            lastText.set_text("")
            return

        dt = datetime.now() - self.dataStartTS

        # Capped at buffer_max_samples
        sample_set_size = len(self.data)

        timestamps = []
        samples = [] #np.arange(chart_max_samples, dtype="float64")

        subsamples = max(1, min(max_supersampling, int(sample_set_size/chart_max_samples)))
        
        # Sub-sampling for longer window views without the redraw perf impact
        for i in range(0, chart_max_samples):
            sample_index = int(sample_set_size*i/chart_max_samples)
            timestamps.append(self.timestamps[sample_index])
            supersample = np.array([self.data[i] for i in range(sample_index, sample_index+subsamples)])
            samples.append(np.median(supersample) if median_filter else np.average(supersample))

        self.ax.set_xlim(timestamps[0], timestamps[-1])

        # Some machines max out at 100fps, so this should react in 0.5-5 seconds to actual speed
        sps_samples = min(512, sample_set_size);
        dt_sps = (np.datetime64(datetime.now()) - self.timestamps[-sps_samples])/np.timedelta64(1, "s");

        # If more than 1 second since last sample, automatically set SPS to 0 so we don't have until it slowly decays to 0
        sps = sps_samples/dt_sps if ((np.datetime64(datetime.now()) - self.timestamps[-1])/np.timedelta64(1, "s")) < 1 else 0.0
        lastText.set_text(f"{sps:.1f} SPS")
        if sps > 500:
            if light_theme:
                lastText.set_color("black")
            else:
                lastText.set_color("white")
        elif sps > 100:
            if light_theme:
                lastText.set_color("orange")
            else:
                lastText.set_color("yellow")
        else:
            lastText.set_color("red")

        if autoscale_current and linear_current_axis:
            self.ax.relim()
            self.ax.set_ylim(bottom=0, top=max(samples) * 1.1)

        logging.debug(f"Drawing chart: range {samples[0]}@{timestamps[0]} .. {samples[-1]}@{timestamps[-1]}")
        lines.set_data(timestamps, samples)
        self.ax.legend(labels=[f"Last: {textAmp(samples[-1])}\nAvg.: {textAmp(sum(samples)/len(samples))}\nTime: {round(chart_length_s, 2):.2f} s"])


    def isStreaming(self) -> bool:
        """
        Checks whether data streaming is active.

        Returns:
        True if streaming is active, False otherwise.
        """
        return self.stream_data

    def close(self):
        """
        Closes the serial connection and stops data streaming.

        This function ensures that the background thread is stopped and the
        serial connection is properly closed.
        """
        self.stream_data = False

        if self.thread != None:
            self.thread.join()

        if self.serialConnection != None:
            self.serialConnection.close()

        logging.info("Connection closed.")

def setup_plot_style(ax, fig, title):
    """
    Configures the style and appearance of a plot.

    Parameters:
    - ax: The matplotlib Axes object to configure.
    - fig: The matplotlib Figure object to configure.
    - title: The title of the plot.

    This function applies consistent styling to the plot, including grid lines,
    axis labels, title, and footer text. It also adjusts the y-axis scale based
    on the `linear_current_axis` setting.
    """
    if not light_theme:
        ax.set_title(title, color="white")
        if not hide_info_on_plot:
            fig.text (0.2, 0.88, f"CurrentViewer v{version}", color="white",  verticalalignment="bottom", horizontalalignment="center", fontsize=9, alpha=0.5)
            fig.text (0.89, 0.0, f"github.com/MGX3D/CurrentViewer", color="white",  verticalalignment="bottom", horizontalalignment="center", fontsize=9, alpha=0.5)
    else:
        ax.set_title(title)
        if not hide_info_on_plot:
            fig.text (0.2, 0.88, f"CurrentViewer {version}", verticalalignment="bottom", horizontalalignment="center", fontsize=9, alpha=0.5)
            fig.text (0.89, 0.0, f"github.com/MGX3D/CurrentViewer", verticalalignment="bottom", horizontalalignment="center", fontsize=9, alpha=0.5)

    ax.set_ylabel("Current")
    currentFormatter = EngFormatter(unit="A")
    if not linear_current_axis:
        ax.set_yscale("log", nonpositive="clip")
        ax.set_ylim(1e-10, 1e1)
        ax.yaxis.set_minor_formatter(currentFormatter)
    ax.yaxis.set_major_formatter(currentFormatter)

    if not light_theme:
        ax.grid(axis="y", which="both", color="yellow", alpha=.3, linewidth=.5)
    else:
        ax.grid(axis="y", which="both", alpha=.3, linewidth=.5)

    ax.set_xlabel("Time")
    plt.xticks(rotation=20)

    if not light_theme:
        ax.grid(axis="x", color="green", alpha=.4, linewidth=2, linestyle=":")
    else:
        ax.grid(axis="x", alpha=.3, linewidth=1, linestyle=":")

    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))

    def on_xlims_change(event_ax):
        global chart_length_s
        logging.debug(f"Interactive zoom: {num2date(event_ax.get_xlim()[0])} .. {num2date(event_ax.get_xlim()[1])}")

        chart_length_s = (num2date(event_ax.get_xlim()[1]) - num2date(event_ax.get_xlim()[0])).total_seconds()

        if chart_length_s < 5:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: num2date(x).strftime("%H:%M:%S.%f").rstrip("0").rstrip(".")))
        else:
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: num2date(x).strftime("%H:%M:%S.%f").rstrip("0").rstrip(".")))

    ax.callbacks.connect("xlim_changed", on_xlims_change)

def textAmp(curr):
    """
    Converts a current value into a human-readable string with appropriate units.

    Parameters:
    - curr: The current value in amperes.

    Returns:
    A string representation of the current value with units (A, mA, µA, or nA).
    """
    if (abs(curr) > 1.0):
        return f"{curr:.3f} A"
    if (abs(curr) > 0.001):
        return f"{curr*1000:.2f} mA"
    if (abs(curr) > 0.000001):
        return f"{curr*1000*1000:.1f} \u00B5A"
    return f"{curr*1000*1000*1000:.1f} nA"

def plot_from_file(file_path):
    """
    Plots data from an existing CSV file.

    Parameters:
    - file_path: The path to the CSV file containing the data.

    This function reads the CSV file, extracts timestamps and current values,
    and plots them using matplotlib. It applies consistent styling to the plot.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=[0])
        timestamps = pd.to_datetime(data.iloc[:, 0])
        currents = data.iloc[:, 1]

        if not light_theme:
            plt.style.use("dark_background")
        fig, ax = plt.subplots(num=f"CurrentViewer v{version}", figsize=(10, 6))
        ax.plot(timestamps, currents, label="Current")
        plt.legend(loc="upper right", framealpha=0.5)

        setup_plot_style(ax, fig, f"File: {file_path}")

        crs = mplcursors.cursor(ax, hover=True)
        @crs.connect("add")
        def _(sel):
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="yellow", alpha=.4)
            sel.annotation.set_text(textAmp(sel.target[1]))

        plt.show()

    except Exception as e:
        logging.error(f"Failed to plot from file '{file_path}': {e}")
        print(f"Error: Could not plot data from file '{file_path}'. Check the logs for details.", file=sys.stderr)

def init_argparse() -> argparse.ArgumentParser:
    """
    Initializes and configures the argument parser for the script.

    Returns:
    An argparse.ArgumentParser object with all the supported command-line arguments.

    This function defines the command-line arguments for the script, including
    options for serial port, baud rate, input file, output file, and various
    configuration settings.
    """
    parser = argparse.ArgumentParser(
        usage="\t%(prog)s -p <port> [OPTION(s)]\n\t%(prog)s -i <file> [OPTION(s)]",
        description="CurrentRanger R3 Viewer"
    )

    parser.add_argument("--version", action="version", version = f"{parser.prog} version {version}")
    parser.add_argument("-p", "--port", metavar='<port>', nargs=1, help="Set the serial port (backed by USB or Bluetooth) to connect to(example: /dev/ttyACM0 or COM3)")
    parser.add_argument("-s", "--baud", metavar='<n>', type=int, nargs=1, help=f"Set the serial baud rate (default: {baud})")
    parser.add_argument("-i", "--input", metavar='<file>', nargs=1, help="Plot data from an existing CSV file")
    parser.add_argument("-o", "--out", metavar='<file>', nargs=1, help=f"Save the output samples to <file>.csv/json")
    parser.add_argument("-g", "--no-gui", dest="gui", action="store_false", help="Do not display the GUI / Interactive Chart. Useful for automation")
    parser.add_argument("-b", "--buffer", metavar='<samples>', type=int, nargs=1, help=f"Set the chart buffer size (window size) in # of samples (default: {buffer_max_samples})")
    parser.add_argument("-m", "--max-chart", metavar='<samples>', type=int, nargs=1, help=f"Set the chart max # samples displayed (default: {chart_max_samples})")
    parser.add_argument("-r", "--refresh", metavar='<ms>', type=int, nargs=1, help=f"Set the live chart refresh interval in milliseconds (default: {refresh_interval})")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Show the debug messages in the console (can be specified at most 3 times to increase logging verbosity)")
    parser.add_argument("--log-size", metavar='<Mb>', type=float, nargs=1, help=f"Set the log maximum size in megabytes (default: {log_size_bytes/1024/1024:.0f})")
    parser.add_argument("-l", "--log-file", metavar="<file>", nargs=1, help=f"Set the debug log filename and start logging to it (always with the highest verbose level)")
    parser.add_argument("--linear", default=False, action="store_true", help="Use a linear current-axis (with toggle-able autoscaling)")
    parser.add_argument("--switch-theme", default=False, action="store_true", help=f"Switch from the dark to the light theme or vice-versa (currently: {'Light' if light_theme else 'Dark'})")

    parser.set_defaults(gui=True)
    return parser

def main():
    """
    The main entry point of the script.

    This function parses command-line arguments, initializes logging, and
    either starts live data streaming or plots data from a file. It handles
    both GUI and non-GUI modes and manages the lifecycle of the CRPlot object.
    """
    global log_size_bytes

    print(f"CurrentViewer v{version}")

    parser = init_argparse()
    args = parser.parse_args()

    ignore_string = ""

    if not (args.port or args.input):
        parser.error("Use (at least) one of the above calls to use this script")

    if args.port and args.input:
        ignore_string = ignore_string + f"-p|--port {args.port[0]} "

    if args.log_file:
        logfile = args.log_file[0]
        file_logger = RotatingFileHandler(logfile, maxBytes=log_size_bytes, backupCount=1)
        file_logger.setLevel(logging.DEBUG)
        file_logger.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(threadName)s:%(message)s"))
        logging.getLogger().addHandler(file_logger)

    if args.log_size:
        log_size_bytes = 1024*1024*args.log_size[0]

    if args.refresh:
        global refresh_interval
        if args.input:
            ignore_string = ignore_string + f"-r|--refresh {args.refresh[0]} "
        refresh_interval = args.refresh[0]

    if args.baud:
        global baud
        if args.input:
            ignore_string = ignore_string + f"-s|--baud {args.baud[0]} "
        baud = args.baud[0]

    if args.max_chart and args.max_chart[0] > 10:
        global chart_max_samples
        if args.input:
            ignore_string = ignore_string + f"-m|--max-chart {args.max_chart[0]} "
        chart_max_samples = args.max_chart[0]

    if args.buffer:
        global buffer_max_samples
        if args.input:
            ignore_string = ignore_string + f"-b|--buffer {args.buffer[0]} "
        buffer_max_samples = args.buffer[0]
        if buffer_max_samples < chart_max_samples:
            print("Command line error: Buffer size cannot be smaller than the chart sample size", file=sys.stderr)
            return -1

    logging_level = logging.DEBUG if args.verbose>2 else (logging.INFO if args.verbose>1 else (logging.WARNING if args.verbose>0 else logging.ERROR))

    # Disable matplotlib logging for fonts, seems to be quite noisy
    logging.getLogger("matplotlib.font_manager").disabled = True

    if (args.verbose>0) or args.log_file:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.verbose>0:
        print("Enabling console logging")
        console_logger = logging.StreamHandler()
        console_logger.setLevel(logging_level)
        console_logger.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logging.getLogger().addHandler(console_logger)

    if args.linear:
        global linear_current_axis
        linear_current_axis = True
    
    if args.switch_theme:
        global light_theme
        light_theme = not light_theme

    global save_file
    global save_format

    if args.out:
        if not args.input:
            output_file_name = args.out[0]
            save_file = open(output_file_name, "w+")

            if not save_format:
                save_format = "CSV" if output_file_name.upper().endswith(".CSV") else "JSON"
                logging.info(f"Save format automatically set to {save_format} for {args.out[0]}")

            if save_format == "CSV":
                save_file.write("DateTime [YYYY-MM-DD HH:MM:SS.ms],Current [A]\n")
            elif save_format == "JSON":
                save_file.write("{\n\"data\":[\n")
        else:
            ignore_string = ignore_string + f"-o|--out {args.out[0]} "

    if args.input:
        file_path = args.input[0]
        if not path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
            return -1
        if ignore_string != "":
            print(f"(ignoring the following arguments since we're plotting from a file: {ignore_string[:-1]})") # "[:-1]" to remove the last blank space
        print(f"Plotting data from file: {file_path}")
        plot_from_file(file_path)
        return 0

    logging.info(f"CurrentViewer v{version}. System: {platform.system()}, Platform: {platform.platform()}, Machine: {platform.machine()}, Python: {platform.python_version()}")

    csp = CRPlot(sample_buffer=buffer_max_samples)

    if csp.serialStart(port=args.port[0], speed=baud):
        if args.gui:
            print("Starting live chart...")
            csp.chartSetup(refresh_interval=refresh_interval)
        else:
            print("Running with no GUI (press Ctrl-C to stop)...")
            try:
                while csp.isStreaming():
                    time.sleep(0.01)
            except KeyboardInterrupt:
                logging.info("Terminated")
                csp.close()

            print("Done.")
    else:
        print(f"Fatal: Could not connect to USB/BT COM port {args.port[0]}. Check the logs for more information", file=sys.stderr)

    csp.close()

    if save_file:
        if save_format == "JSON":
            save_file.write("\n]\n}\n")
        save_file.close()

if __name__ == "__main__":
  main()
