""" plot_utils.py: 
Provides functionality to generate simple plots
"""

__maintainer__ = "Ayush"

from collections import defaultdict
import numpy
from scipy import stats
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from numpy import logspace, log10, linspace
from scipy.stats import ttest_ind
from plot_utils import plot_multiple_line as plot_multiple_line_gnuplot, plot_barchart as plot_barchart_gnuplot


plt.style.use('ggplot')
plt.rcParams['font.size'] = 30
plt.rcParams.update({'font.size': 30})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = 18,12
max_yticks = 5
loc = plt.MaxNLocator(max_yticks)

import plotly.plotly as py
import plotly.graph_objs as go
import plotly

def plot_histogram(val_list, bins, title, xlabel, ylabel, fig_name, y_logscale=False):
    plt.hist(val_list, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_logscale:
        plt.gca().set_yscale('symlog')
    plt.savefig(fig_name)
    plt.close()


def plot_histogram_plotly(val_list, bins, title, xlabel, ylabel, fig_name, y_logscale=False, x_logscale=False):
    data = [go.Histogram(
        x=val_list,
        nbinsx=bins,
        # autobinx=False,
        # xbins=dict(start=min(val_list), end=max(val_list), size=10),
        marker=dict(
            line=dict(
                color='rgb(255, 217, 102)'
                ),
            )
        )]

    x_dict = dict(title=xlabel)
    if x_logscale:
        x_dict.update(type='log', autorange=True)
    y_dict = dict(title=ylabel)
    if y_logscale:
        y_dict.update(type='log', autorange=True)
    layout = go.Layout(
        font=dict(size=36),
        # font=dict(size=26),
        xaxis=x_dict,
        yaxis=y_dict,
        title=title,
        bargap=0.25,
    )

    fig = dict(data=data, layout=layout)
    url = plotly.offline.plot(fig, filename=fig_name, auto_open=False)

def plot_barchart_plotly(heights, labels, xlabel, ylabel, fig_name, title='', y_logscale=False):
    new_labels = []
    for label in labels:
        new_labels.append(label.replace(' ', '<br>'))
    layout = dict(
        font=dict(
                size=32
                ),
        xaxis = dict(
                title = xlabel
            ),
        yaxis = dict(
                title = ylabel
            ),
        #yaxis=dict(
            title=title
        #    )
        )
    if y_logscale:
        layout['yaxis']['type'] = 'log'
    data = [go.Bar(
            x=new_labels,
            y=heights
        )]
    fig = dict(data=data, layout=layout)
    url = plotly.offline.plot(fig, filename=fig_name + ".html", auto_open=False)
    

def plot_barchart(heights, labels, xlabel, ylabel, fig_name, title='', y_logscale=False):
    N = len(heights)
    ind = range(N)
    width = 0.35
    rects = plt.bar(ind, heights, width)
    if y_logscale:
        plt.gca().set_yscale('symlog')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels:



        plt.gca().set_xticks([x + width / 2.0 for x in ind])
        plt.gca().set_xticklabels(tuple(labels))

    plt.savefig(fig_name)
    plt.close()



def plot_barchart_horizontal_plotly(heights, labels, ylabel, xlabel, fig_name, title='', y_logscale=False):
    new_labels = []
    for label in labels:
        new_labels.append(label.replace(' ', '<br>'))
    layout = dict(
        font=dict(
                family='Arial',
                size=36
                ),
        xaxis = dict(
                title = xlabel
            ),
        yaxis = dict(
                title = ylabel,
                showticklabels=False
            ),
        #yaxis=dict(
            title=title
        #    )
        )
    if y_logscale:
        layout['xaxis']['type'] = 'log'
    data = [go.Bar(
            y=new_labels,
            x=heights,
            orientation='h', 
            marker=dict(
                color='#8b1a0e'
            )
        )]
    xmin, xmax = min(heights), max(heights)
    x_range = xmax - xmin
    annotations = []
    for l, h in zip(labels, heights):
        d = dict(
            xref='x', yref='y',
            y = l, x = h,
            text = '<b>{}</b>'.format(l), 
            showarrow=False,
            xanchor='right',
            font=dict(size=42)
        )
        if not y_logscale and h < 0.25*x_range:
            d['x'] = h + 0.01*x_range
            d['xanchor'] = 'left'
        else:
            d['x'] = 0.98 * h
            d['xanchor'] = 'right' 
            d['font'] = dict(color='rgb(220,220,220)')
        annotations.append(d)
    layout['annotations'] = annotations


    fig = dict(data=data, layout=layout)
    url = plotly.offline.plot(fig, filename=fig_name)


def plot_barchart_horizontal(heights, labels, xlabel, ylabel, fig_name, title='', y_logscale=False):
    N = len(heights)
    ind = numpy.arange(N) + 0.5
    width = 0.5
    rects = plt.barh(ind, heights, height=width, align='center')
    if y_logscale:
        plt.gca().set_xscale('symlog')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.title(title)

    xmin, xmax = plt.gca().get_xlim()
    x_range = xmax - xmin
    
    ind = 0
    rect_labels = []
    for rect in rects:

        if labels is None or ind == len(labels):
            break
        
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        
        width = rect.get_width()
        channel_name = labels[ind]
        ind += 1

        ## TODO:
        ## Handle log plots more elegantly
        ##

        # The bars aren't wide enough to print the ranking inside
        if (not y_logscale) and (width < 0.25 * x_range):
            # Shift the text to the right side of the right edge
            xloc = width + 0.01 * x_range
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = 0.98*width
            # White on magenta
            clr = 'black'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height()/2.0
        label = plt.gca().text(xloc, yloc, channel_name, horizontalalignment=align,
                         verticalalignment='center', color=clr, weight='bold',
                         clip_on=True)
        rect_labels.append(label)
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(fig_name)
    plt.close()

def plot_scatter_plotly(xs, ys, xlabel, ylabel, fig_name, title='', x_logscale=False, y_logscale=False):
    layout = dict(
        font=dict(
                size=36
                ),
        xaxis = dict(
                title = xlabel
            ),
        yaxis = dict(
                title = ylabel
            ),
        #yaxis=dict(
            title=title
        #    )
        )
    if y_logscale:
        layout['yaxis']['type'] = 'log'
    if x_logscale:
        layout['xaxis']['type'] = 'log'

    data = []

    for i in range(len(xs)):
        data.append(go.Scatter(
                x = xs[i],
                y = ys[i],
                mode = 'markers'
            )
        )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=fig_name, auto_open=False)


def plot_stacked_chart_plotly(
    xlabels, stacks, ylabels,
    title, fig_name, normalized=False
):
    # stacks = [[], [], []]
    # for each ylabels[i], stacks[i] is the list of lengths corresponding
    # to ylabels[i]'s contribution to xlabels[i]
    assert len(ylabels) == len(stacks)
    for stack in stacks:
        assert len(stack) == len(xlabels)

    if normalized:
        heights = [[stacks[i][j] for j in range(len(stacks[i]))] for i in range(len(stacks))]
        for i in range(len(stacks)):
            for j in range(len(stacks[i])):
                try:
                    heights[i][j] = stacks[i][j]*100.0/(sum(stacks[k][j] for k in range(len(stacks))))
                except ZeroDivisionError:
                    heights[i][j] = 0
        stacks = heights 
        

    data = []
    for i in range(len(ylabels)):
        data.append(
            go.Bar(
                x=xlabels,
                y=stacks[i],
                name=ylabels[i]
            )
        )

    layout = go.Layout(
        barmode='stack'
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=fig_name, auto_open=False)


def plot_multiple_scatter(
    xs, ys, title, xlabel, ylabel, legends, fig_name, y_logscale=False,
    styles=[], leg_pos=None,
    add_trend='linear_reg'
):
    assert len(xs) == len(ys)
    if legends:
        assert len(xs) == len(legends)
        if not leg_pos:
            leg_pos = 'upper right'
    if styles:
        assert len(xs) == len(styles)
    else:
        styles = []
        for i in range(len(xs)):
            styles.append('o')

    if add_trend == 'median':
        # add a median line for each x, y plot
        old_xs = xs[:]
        old_ys = ys[:]
        for i in range(len(old_xs)):
            ys_for_x = defaultdict(list)
            # xs.append([])  # mean
            xs.append([])  # median
            # ys.append([])  # mean
            ys.append([])  # median
            # styles.append('-')  # mean
            styles.append('-')  # median
            # legends.append('mean_{}'.format(legends[i]))  # mean
            # legends.append('median_{}'.format(legends[i]))  # median
            for j, x in enumerate(old_xs[i]):
                ys_for_x[x].append(old_ys[i][j])
            for x in sorted(ys_for_x.keys()):
                # xs[len(xs) - 2].append(x)
                xs[len(xs) - 1].append(x)
                # ys[len(xs) - 2].append(np.mean(ys_for_x[x]))
                ys[len(xs) - 1].append(numpy.median(ys_for_x[x]))
    elif add_trend == 'linear_reg':
        # add a linear reg line for each x, y plot
        # legend = correlation coeff
        old_xs = xs[:]
        old_ys = ys[:]
        if not legends:
            for i in range(len(old_xs)):
                legends.append(str(i + 1))
        for i in range(len(old_xs)):
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(old_xs[i], old_ys[i])
            min_x = min(old_xs[i])
            max_x = max(old_xs[i])
            if numpy.isnan(slope):
                xs.append([min_x, max_x])
                ys.append([min(old_ys[i]), max(old_ys[i])])
            else:
                if not y_logscale:
                    xs.append([min_x, max_x])
                    ys.append([
                        min_x * slope + intercept,
                        max_x * slope + intercept
                    ])
                else:
                    num_discrete_steps = 100
                    xs.append([])
                    ys.append([])
                    for d in range(num_discrete_steps + 1):
                        next_x = min_x + \
                            (d / float(num_discrete_steps)) * float(max_x - min_x)
                        xs[len(xs) - 1].append(next_x)
                        ys[len(ys) - 1].append(next_x * slope + intercept)
            legends.append('{}_LR({:2.3f})({:2.3f})'.format(legends[i], r_value, p_value))
            styles.append('-')

    # sort plots (x, y) in (xs, ys) by number of points, and plot lower number lines later
    # (to prevent overlapping / covering all less frequent ones)
    lens_and_ids = []
    for i, x in enumerate(xs):
        lens_and_ids.append((len(x), i))
    sorted_ids = [
        i
        for length, i in sorted(lens_and_ids, key=(lambda x: x[0]), reverse=True)]
    sorted_xs = []
    sorted_ys = []
    if legends:
        sorted_legends = []
    sorted_styles = []
    for orig_index in sorted_ids:
        sorted_xs.append(xs[orig_index])
        sorted_ys.append(ys[orig_index])
        if legends:
            sorted_legends.append(legends[orig_index])
        if styles:
            sorted_styles.append(styles[orig_index])
    plt.figure()
    for i in range(len(sorted_xs)):
        style = 'o' if not styles else sorted_styles[i]
        plt.plot(sorted_xs[i], sorted_ys[i], style)
    if y_logscale:
        plt.gca().set_yscale('symlog')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # TODO: parametrize below
    if legends:
        fontP = FontProperties()
        fontP.set_size('small')

        # plt.legend(sorted_legends, loc=leg_pos)
        plt.legend(sorted_legends, prop=fontP)

    plt.savefig(fig_name)
    plt.close()


def plot_multiple_line(
    xs, ys, title, xlabel, ylabel, legends, fig_name, x_logscale=False, y_logscale=False, 
    leg_pos=None, styles=[]
):
    assert len(xs) == len(ys)
    if not leg_pos:
        leg_pos = 'lower right'

    plt.figure()
    for i in range(len(xs)):
        if i >= len(styles):
            styles.append('-')
        plt.plot(xs[i], ys[i], styles[i])
    if x_logscale:
        plt.gca().set_xscale('symlog')
    if y_logscale:
        plt.gca().set_yscale('symlog')
    #plt.ylim([0, 1.1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legends is not None:
        plt.legend(legends, loc=leg_pos)
    plt.savefig(fig_name)
    plt.close()

def plot_multiple_line_plotly(
    xs, ys, title, xlabel, ylabel, legends, fig_name, x_logscale=False, y_logscale=False,
    reference_lines = []):
    assert len(xs) == len(ys)
    layout = dict(
        font=dict(
                size=36
                ),
        xaxis = dict(
                title = xlabel
            ),
        yaxis = dict(
                title = ylabel
            ),
        #yaxis=dict(
            title=title
        #    )
        )
    if reference_lines:
        layout['shapes'] = []
        for (x0,y0,x1,y1) in reference_lines:
            layout['shapes'].append(
                {
                    'type': 'line',
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1,
                    'line': 
                        {
                            'dash':'dot'
                        }
                }
            )
    if y_logscale:
        layout['yaxis']['type'] = 'log'
    if x_logscale:
        layout['xaxis']['type'] = 'log'

    if legends is None:
        layout['showlegend'] = False
    data = []

    for i in range(len(xs)):
        d = dict(
                x = xs[i],
                y = ys[i],
                mode = 'lines+markers'
            )
        if legends is not None:
            d['name'] = legends[i]
        data.append(go.Scatter(d))

    fig = dict(data=data, layout=layout)
    url = plotly.offline.plot(fig, filename=fig_name, auto_open=False)


def plot_CDF_plotly(attrib_vals, metric_vals, bin_funcs, bin_labels,
    metric_name, fig_name, inverse_cdf=False, metric_logscale=False,
    reference_lines=[], mid_line=False, width=-3,
    xtick_font_size=None):
    assert len(attrib_vals) == len(metric_vals)

    step_size = 0.02
    if mid_line:
        # add a vertical line in the middle of the plot
        x_min = min(metric_vals)
        x_max = max(metric_vals)
        increment = step_size*(x_max - x_min)
        if metric_logscale:
            ref_x_nums = logspace(log10(max(0.1, x_min)), log10(x_max), 50)
        else:
            ref_x_nums = numpy.arange(x_min, x_max, increment)
        middle_x = ref_x_nums[len(ref_x_nums) / 2]
        if reference_lines:
            # passing ref each time to plot_CDF_plotly
            reference_lines = reference_lines[:] + \
                reference_lines.append((middle_x, 0.0, middle_x, 1.0))
        else:
            reference_lines = [(middle_x, 0.0, middle_x, 1.0)]

    observations = []
    data = []

    xs = []
    ys = []
    legends = []

    for func_i in range(len(bin_funcs)):

        func = bin_funcs[func_i]
        filtered_metric_vals = []
        for i in range(len(attrib_vals)):
            if func(attrib_vals[i]):
                filtered_metric_vals.append(metric_vals[i])

        print 'Stats for bin number {} ({}), num items: {}'.format(
            func_i, bin_labels[func_i], len(filtered_metric_vals))
        print 'average: {}, median: {}, min: {}, max: {}, std: {}'.format(
            numpy.mean(filtered_metric_vals),
            numpy.median(filtered_metric_vals),
            min(filtered_metric_vals),
            max(filtered_metric_vals),
            numpy.std(filtered_metric_vals))

        # TODO: figure out why empty metric_vals would even come here
        # (for bins = 2 especially)
        # temporary hack
        if not filtered_metric_vals:
            continue
        x_min = min(filtered_metric_vals)
        x_max = max(filtered_metric_vals)
        increment = step_size*(x_max - x_min)
        if metric_logscale:
            x_nums = logspace(log10(max(0.1, x_min)), log10(x_max), 1.0 / step_size)
        else:
            x_nums = numpy.arange(x_min, x_max + increment, increment)
            try:
                assert x_nums[-1] == x_max
            except AssertionError:
                print 'min={}, max={}, x_nums[0]={}, x_nums[-1]={}'.format(
                    x_min, x_max, x_nums[0], x_nums[-1])
        # print 'min={}, max={}, x_nums[0]={}, x_nums[-1]={}'.format(
        #     x_min, x_max, x_nums[0], x_nums[-1])
        y_nums = []

        if not inverse_cdf:
            for x in x_nums:
                y_nums.append(sum([1 for i in range(len(filtered_metric_vals)) if filtered_metric_vals[i] <= x]))
            y_nums = [y*1.0/len(filtered_metric_vals) for y in y_nums]
            # print [met for met in filtered_metric_vals if 0.48 < met and met <= 0.5]
        else:
            for x in x_nums:
                y_nums.append(sum([1 for i in range(len(filtered_metric_vals)) if filtered_metric_vals[i] > x]))
            y_nums = [y*1.0/len(filtered_metric_vals) for y in y_nums]
            # print [met for met in filtered_metric_vals if 0.52 >= met and met > 0.5]



        if len(bin_funcs) == 2:
            # observations.append(y_nums[:])
            observations.append(filtered_metric_vals)
        data.append(go.Scatter(
                x = x_nums,
                y = y_nums,
                name = bin_labels[func_i]
            ))
        xs.append(x_nums)
        ys.append(y_nums)
        legends.append(bin_labels[func_i])

    if inverse_cdf:
        ylabel = '1 - cdf'
    else:
        ylabel = 'cdf'

    layout = dict(
        font=dict(
                size=28
                ),
        xaxis = dict(
                title = metric_name
            ),
        yaxis = dict(
                title = ylabel
            )
        )

    if reference_lines:
        layout['shapes'] = []
        for (x0,y0,x1,y1) in reference_lines:
            layout['shapes'].append(
                {
                    'type': 'line',
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1,
                    'line': 
                        {
                            'dash':'dot'
                        }
                }
            )
    if metric_logscale:
        layout['xaxis']['type'] = 'log'
    fig = dict(data=data, layout=layout)
    url = plotly.offline.plot(fig, filename=fig_name + '.html', auto_open=False)

    plot_multiple_line_gnuplot(
    xs, ys, '', metric_name, None, legends, fig_name, x_logscale=metric_logscale, y_logscale=False,
    reference_lines=reference_lines, linetypes=['lines']*len(xs), scale=1.0/3,
    width=width, xtick_font_size=xtick_font_size)

    if len(observations) == 2:
        #_, p_value = ttest_ind(observations[0], observations[1])
        #print "p-value (assuming equal variance of both distributions):", p_value
        _, p_value = ttest_ind(observations[0], observations[1], equal_var=False)
        #print "p-value (Without assumption on variance):", p_value
        return p_value