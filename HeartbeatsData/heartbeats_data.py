from calendar import day_abbr
from collections import Counter
from datetime import datetime, date
from typing import Callable

import numpy as np
from numpy import zeros, newaxis, sum as npsum

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LinearLocator
from matplotlib.widgets import Slider

_SECS_IN_MIN = 60
_MINS_IN_HOUR = 60
_SECS_IN_HOUR = _MINS_IN_HOUR * _SECS_IN_MIN
_HOURS_IN_DAY = 24
_SECS_IN_DAY = _SECS_IN_HOUR * _HOURS_IN_DAY
_DAYS_IN_WEEK = 7


def show():
    plt.show()


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)


class HeartbeatData:
    DEFAULT_LEGEND_LENGTH = 10
    DEFAULT_TIMEOUT = 15 * _SECS_IN_MIN

    fig: Figure = None
    ax: Axes = None

    hb_type_counter = Counter()
    hb_types = []
    dates = []
    secs_since_midnights = []

    hb_type_color_map: dict[str, tuple[float, float, float]] = None  # Set by legend
    colors: list[tuple[float, float, float]] = None  # Set by legend
    durations: dict[date, list[tuple[str, int, int]]] = None  # Set by calc_durations
    duration_counts = None  # Set by calc_durations
    timeout_slider: Slider = None  # Set by show_timeout_slider

    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def add_hb(self, hb_type: str, timestamp: datetime):
        if hb_type == "":
            hb_type = "Other"
        else:
            self.hb_type_counter[hb_type] += 1
        self.hb_types.append(hb_type)
        self.dates.append(timestamp.date())
        self.secs_since_midnights.append(
            timestamp.hour * _SECS_IN_HOUR + timestamp.minute * _SECS_IN_MIN + timestamp.second)

    def calc_durations(self, timeout=DEFAULT_TIMEOUT):
        print("Calculating durations")
        self.durations = {}
        heartbeats = zip(self.hb_types, self.dates, self.secs_since_midnights)
        initial = next(heartbeats)
        curr_hb_type = initial[0]
        curr_date = initial[1]
        curr_start = initial[2]
        curr_end = curr_start
        for hb_type, hb_date, secs_since_midnight in heartbeats:
            if hb_type != curr_hb_type or hb_date != curr_date or secs_since_midnight > curr_end + timeout:
                self.durations.setdefault(curr_date, []).append((curr_hb_type, curr_start, curr_end - curr_start))
                curr_hb_type = hb_type
                curr_date = hb_date
                curr_start = secs_since_midnight
            curr_end = secs_since_midnight

    def calc_duration_counts(self, timeout=DEFAULT_TIMEOUT):
        print("Calculating duration counts")
        self.duration_counts = zeros((_DAYS_IN_WEEK, _SECS_IN_DAY - 1), np.int16)
        heartbeats = zip(self.hb_types, self.dates, self.secs_since_midnights)
        initial = next(heartbeats)
        curr_date = initial[1]
        curr_start = initial[2]
        curr_end = curr_start
        for hb_type, hb_date, secs_since_midnight in heartbeats:
            if hb_date != curr_date or secs_since_midnight > curr_end + timeout:
                self.duration_counts[curr_date.weekday(), curr_start:curr_end + 1] += 1
                curr_date = hb_date
                curr_start = secs_since_midnight
            curr_end = secs_since_midnight

    def legend(self, legend_length=DEFAULT_LEGEND_LENGTH, other_name="Other", color_map="tab20", **kwargs):
        color_map = colormaps[color_map].colors
        self.hb_type_color_map = {other_name: color_map[0]}
        for i, (hb_type, _) in enumerate(self.hb_type_counter.most_common(legend_length - 1)):
            self.hb_type_color_map[hb_type] = color_map[i + 1]
        for i, hb_type in enumerate(self.hb_types):
            if hb_type not in self.hb_type_color_map:
                self.hb_types[i] = other_name
        self.colors = [self.hb_type_color_map.get(hb_type, color_map[0]) for hb_type in self.hb_types]
        self.ax.legend(handles=[Patch(color=color, label=hb_type) for hb_type, color in self.hb_type_color_map.items()],
                       **kwargs)

    def plot_dates(self):
        self.ax.yaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        self.ax.set_ylabel('Date')
        self.ax.invert_yaxis()

    def plot_times(self, weekdays=False):
        def format_time(x):
            if weekdays:
                return '{} {:02d}:{:02d}'.format(
                    day_abbr[x // _SECS_IN_DAY],
                    (x // _SECS_IN_HOUR) % _HOURS_IN_DAY,
                    (x % _SECS_IN_HOUR) // _MINS_IN_HOUR
                )
            return '{:02d}:{:02d}'.format(
                x // _SECS_IN_HOUR,
                (x % _SECS_IN_HOUR) // _MINS_IN_HOUR
            )

        self.ax.margins(x=0)
        self.ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: format_time(int(x))
        ))
        self.ax.xaxis.set_major_locator(LinearLocator(8))  # If over 5, should be 7n+1
        self.ax.set_xlabel('Time of day')

    def plot_scatter(self, **kwargs):
        if self.colors is not None:
            kwargs["color"] = self.colors
        self.plot_dates()
        self.plot_times()
        self.ax.scatter(self.secs_since_midnights, mdates.date2num(self.dates), **kwargs)

    def plot_durations(self, timeout_slider=True, plot_kwargs=None, slider_kwargs=None):
        def refresh():
            if self.durations is None:
                raise ValueError("Tried to plot durations before durations have been calculated")
            for hb_date, durations in self.durations.items():
                if self.colors is not None:
                    plot_kwargs["facecolors"] = [self.hb_type_color_map[duration[0]] for duration in durations]
                self.ax.broken_barh(
                    [duration[1:] for duration in durations],
                    (mdates.date2num(hb_date), 1),
                    **plot_kwargs
                )
            self.plot_dates()
            self.plot_times()

        if plot_kwargs is None:
            plot_kwargs = {}
        if slider_kwargs is None:
            slider_kwargs = {}
        if timeout_slider:
            self.show_timeout_slider(self.calc_durations, refresh, **slider_kwargs)
        else:
            refresh()

    def plot_duration_counts(self, as_heatmap=False, weekly=False, timeout_slider=True, plot_kwargs: dict[str, any] = None, slider_kwargs: dict[str, any] = None):
        def refresh():
            if self.duration_counts is None:
                raise ValueError("Tried to plot duration counts before duration counts have been calculated")
            y = self.duration_counts.flatten() if weekly else npsum(self.duration_counts, axis=0)
            if as_heatmap:
                self.ax.imshow(y[newaxis, :], aspect="auto", **plot_kwargs)
                self.ax.set_yticks([])
            else:
                self.ax.set_ylabel("How many durations include that time")
                self.ax.plot(y, **plot_kwargs)
            self.plot_times(weekly)

        if plot_kwargs is None:
            plot_kwargs = {}
        if slider_kwargs is None:
            slider_kwargs = {}
        if timeout_slider:
            self.show_timeout_slider(self.calc_duration_counts, refresh, **slider_kwargs)
        else:
            refresh()

    def show_timeout_slider(self, calc_fn: Callable, plot_fn: Callable, default=DEFAULT_TIMEOUT, rect: tuple[float, float, float, float] = None):
        def refresh(timeout):
            calc_fn(timeout)
            print("Refreshing plot")
            self.ax.clear()
            plot_fn()
            self.fig.canvas.draw_idle()

        if rect is None:
            rect = [0.15, 0.1, 0.75, 0.03]
        calc_fn(default)
        plot_fn()
        self.fig.subplots_adjust(bottom=rect[1] * 2 + rect[3])
        self.timeout_slider = Slider(self.fig.add_axes(rect), 'Timeout', 1, _SECS_IN_DAY, valinit=default)
        self.timeout_slider.on_changed(refresh)
