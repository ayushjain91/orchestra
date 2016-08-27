import datetime
from numpy import histogram, logspace, log10, sqrt
import warnings
from subprocess import call

PLOT_COMMAND_FILE = "test.plt"
PLOT_DATA_FILE = "data.tmp"
PLOT_REFERENCE_LINES_FILE = "reference.tmp"


def template_header(out_file, scale=1, xtick_font_size=None):
    global scale_func
    def scale_func(x, scale=scale):
        return round(scale*x)

    fig_height = 12
    fig_width = 18
    font_size = 96
    if xtick_font_size is None:
        xtick_font_size = font_size

    header = ""
    header += "set terminal postscript eps enhanced color solid size {}, {} font 'Arial, {}'\n".format(
            scale_func(fig_width), scale_func(fig_height), scale_func(font_size)
        )

    header += "set style line 31 lc rgb '#808080' lt 1\n"
    header += "set border 3 back ls 31\n"
    header += "set tics nomirror font ', {}'\n".format(
        scale_func(xtick_font_size))
    header += "set style line 32 lc rgb '#c0c0c0' lt 0 lw 1\n"
    header += "set grid back ls 32\n"

    header += "set style line 1 lc rgb '#8b1a0e' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 2 lc rgb '#5e9c36' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 3 lc rgb '#f2520d' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 4 lc rgb '#6228d7' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 5 lc rgb '#1972e6' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 6 lc rgb '#468db9' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 7 lc rgb '#55AA97' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 8 lc rgb '#946b93' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 9 lc rgb '#8ffab3' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 10 lc rgb '#ed9821' pt 7 ps 2 lt 1 lw 8\n"
    header += "set style line 11 lc rgb '#CEBD1B' pt 7 ps 2 lt 1 lw 8\n"

    header += "set linetype 1 lc rgb '#8b1a0e'\n"
    header += "set linetype 2 lc rgb '#5e9c36'\n"
    header += "set linetype 3 lc rgb '#f2520d'\n"
    header += "set linetype 4 lc rgb '#6228d7'\n"
    header += "set linetype 5 lc rgb '#1972e6'\n"
    header += "set linetype 6 lc rgb '#468db9'\n"
    header += "set linetype 7 lc rgb '#55AA97'\n"
    header += "set linetype 8 lc rgb '#946b93'\n"
    header += "set linetype 9 lc rgb '#8ffab3'\n"
    header += "set linetype 10 lc rgb '#ed9821'\n"
    header += "set linetype 11 lc rgb '#CEBD1B'\n"

    # Reference lines
    header += "set style line 21 lc rgb '#000000' pt 0 ps 4 lt 0 lw 6\n"

    header += "set output '" + out_file + ".eps'\n"
    header += "set datafile sep '\t'\n"
    return header


def template_axis_labels(xlabel, ylabel, title=None, x_logscale=False, y_logscale=False, xformat=None, yformat=None):
    body = ""
    if xlabel is not None:
        body += "set xlabel '" + xlabel + "'\n"
    if ylabel is not None:
        body += "set ylabel '" + ylabel + "'\n"
    if title is not None:
        body += "set title '" + title + "'\n"
    log_axes = ""
    if x_logscale:
        log_axes += "x"
    if xformat:
        body += "set format x '" + xformat +"'\n"
    if y_logscale:
        log_axes += "y"
    #else:
    #    body += "set yrange [0:]\n"
    if yformat:
        body += "set format y '" + yformat + "'\n"
    if log_axes:
        body += "set logscale " + log_axes + "\n"
    return body


def template_body_barchart_stacked(heights, xlabels, ylabels, pattern=4, index_func=None, normalized=False):
    xtics_size = 72

    body = ""
    # body += "set terminal postscript eps enhanced color solid size 5, 3 font 'Arial, 28'\n"
    body += "set boxwidth 0.5\n"
    body += "set style fill pattern\n"
    body += "set xtics font ', {}'\n".format(scale_func(xtics_size))
    body += "set style data histograms\n"
    body += "set style histogram rowstacked\n"

    # body += "set xtics font ', 24'\n"
    if index_func:
        body += "index(n)=" + index_func + "\n"
    assert len(heights) == len(ylabels)

    
    with open(PLOT_DATA_FILE, "w") as data_file:
        for x_ind in range(len(xlabels)):
            line = xlabels[x_ind]
            for y_ind in range(len(ylabels)):
                if not normalized:
                    line += "\t" + str(heights[y_ind][x_ind])
                else:
                    try:
                        line += "\t" + str(heights[y_ind][x_ind]*100.0/sum(heights[i][x_ind] for i in range(len(heights))))
                    except ZeroDivisionError:
                        line += "\t" + "0.00"
            line += "\n"
            data_file.write(line)

    body += "plot "

    for y_ind in range(len(ylabels)):
        body += "'" + PLOT_DATA_FILE + "' using " 
        body += str(y_ind + 2) + " "
        if y_ind + 1 == len(ylabels):
            body += ":xtic(1) "
        body += "notitle "
        if index_func:
            body += "lt index('" + ylabels[y_ind] + "') "
        if pattern:
            body += "fs pattern " + str(pattern) + " "
        if y_ind + 1 != len(ylabels):
            body += ", "
        else:
            body += "\n"
    return body





def template_body_barchart(heights, labels, pattern=3, index_func=None):
    xtics_size = 72

    body = ""
    body += "set boxwidth 0.5\n"
    body += "set style fill pattern\n"
    
    # if small_fig:
    #     body += "set xtics font ', 24'\n"
    # else: 
    body += "set xtics font ', {}'\n".format(scale_func(xtics_size))
    if index_func:
        body += "index(n)=" + index_func + "\n"

    assert len(heights) == len(labels)
    with open(PLOT_DATA_FILE, "w") as data_file:
        index = 0
        for h, l in zip(heights, labels):
            line = str(index) + "\t" + l.replace(" ", "\\n") + "\t" + str(h)
            data_file.write(line + "\n")
            index += 1
            # if index ==10:
                # break



    body += "plot '" + PLOT_DATA_FILE + "' using "
    if index_func:
        body += "1:3:(index(strcol(2))):xtic(2) with boxes linecolor variable "
    else:
        body += "1:3:xtic(2) with boxes ls 1"
    body += "notitle fs pattern " + str(pattern)
    return body

def template_body_histogram(val_list, nbins, x_logscale=False):
    body = ""
    body += "set boxwidth 0.75 relative\n"
    body += "set style fill solid border -1\n"

    if not x_logscale:
        hist, bin_edges = histogram(val_list, nbins)
    else:
        bin_edges = logspace(log10(max(1, min(val_list))), log10(1.1*max(val_list)), nbins + 1)
        hist = []
        for i in range(len(bin_edges) - 1):
            hist.append(len([v for v in val_list if v<bin_edges[i+1] and v >= bin_edges[i]]))


    with open(PLOT_DATA_FILE, "w") as data_file:
        for i in range(len(hist)):
            if not x_logscale:
                bin_centre = (bin_edges[i] + bin_edges[i+1])*1.0/2 
            else:
                bin_centre = sqrt(bin_edges[i]*bin_edges[i+1])
            line = str(bin_centre) + "\t" + str(hist[i]) + "\n"
            data_file.write(line)

    body += "plot '" + PLOT_DATA_FILE + "' using 1:2 with boxes notitle"
    return body




def template_body_line(
        xs, ys, legends=None, linetypes=None, reference_lines=[],
        width=-3):
    key_font_size = 78

    body = ""

    # Data checks
    assert len(xs) == len(ys)
    if linetypes is not None:
        assert len(xs) == len(linetypes)
    else:
        linetypes = ["linespoints"] * len(xs)
    if legends is not None:
        assert len(xs) == len(legends)
        body += "set key horizontal outside bottom center samplen 2 height 1 maxcols 5 maxrows 1 font 'Arial, {}'\n".format(
            scale_func(key_font_size))
    else:
        legends = [None] * len(xs)
        body += "set key off\n"
    if len(xs) > 11:
        warnings.warn(
            "Maximum 5 linestyles in a plot are defined. Expect weirdness.")

    if isinstance(xs[0][0], datetime.date):
        body += "set terminal postscript eps enhanced color solid size {}, {} font 'Arial, {}'\n".format(
            scale_func(20), scale_func(12), scale_func(84))
        # scale_func(22), scale_func(12), scale_func(84) for label_utils:simple_vs_complex
        # scale_func(20), scale_func(9), scale_func(84) for rest
        body += "set xdata time\n"
        body += "set timefmt '%Y-%m-%d %H:%M:%S'\n"
        body += "set format x \"%b'%y\"\n"

    # Create data file
    with open(PLOT_DATA_FILE, "w") as data_file:
        for i in range(len(xs)):
            for j in range(len(xs[i])):
                line = str(xs[i][j]) + "\t" + str(ys[i][j]) + "\n"
                data_file.write(line)
            data_file.write("\n\n")

        # row = 0
        # while row < max([len(x) for x in xs]):
        #     line = ""
        #     for col in range(len(xs)):
        #         if row < len(xs[col]):
        #             line += str(xs[col][row]) + "\t" + str(ys[col][row])
        #         else:
        #             line += "\t"
        #         if col + 1 < len(xs):
        #             line += "\t"
        #     data_file.write(line + '\n')
        #     print line
        #     row += 1

    with open(PLOT_REFERENCE_LINES_FILE, "w") as data_file:
        line = ""
        for line_num in range(len(reference_lines)):
            line += str(reference_lines[line_num][0]) + \
                "\t" + str(reference_lines[line_num][1])
            if line_num + 1 < len(reference_lines):
                line += "\t"
        line += "\n"
        for line_num in range(len(reference_lines)):
            line += str(reference_lines[line_num][2]) + \
                "\t" + str(reference_lines[line_num][3])
            if line_num + 1 < len(reference_lines):
                line += "\t"
        data_file.write(line)

    # Create template body
    body += "plot "
    for line_number in range(len(xs)):
        # body += "'" + PLOT_DATA_FILE + "' using " + \
        #     str(2 * line_number + 1) + ":" + str(2 * line_number + 2) + " "
        body += "'" + PLOT_DATA_FILE + "' index " + str(line_number) + " using 1:2 "
        if legends[line_number] is not None:
            legend_string = legends[line_number].replace(">=", " {/Symbol \263} ").replace("<=", " {/Symbol \243} ")
            legend_string = legend_string.replace(">", " {/Symbol \076} ").replace("<", " {/Symbol \074} ")
            body += "title '" + legend_string  + "' "
        else:
            body += "notitle "
        body += "with " + linetypes[line_number] + " "
        if line_number <= 11:
            body += "ls " + str(line_number + 1)
        if line_number + 1 < len(xs) or reference_lines:
            body += ", "
    for line_number in range(len(reference_lines)):
        body += "'" + PLOT_REFERENCE_LINES_FILE + "' using " + \
            str(2 * line_number + 1) + ":" + \
            str(2 * line_number + 2) + " notitle with lines ls 21"
        if line_number + 1 < len(reference_lines):
            body += ", "
    return body


def run_gnuplot(plot_commands):
    with open(PLOT_COMMAND_FILE, "w") as plot_file:
        plot_file.write(plot_commands)
    call(['gnuplot', 'test.plt'])
    call(['rm', PLOT_COMMAND_FILE])
    call(['rm', PLOT_DATA_FILE])
    call(['rm', PLOT_REFERENCE_LINES_FILE])


def plot_stacked_chart(
    xlabels, stacks, ylabels,
    title, fig_name, pattern=4, index_func=None, y_logscale=False, normalized=False):
    commands = template_header(fig_name)
    xlabel_format = None
    ylabel_format = '%.1s%c'
    commands += template_axis_labels(None, None, title, False, y_logscale, xlabel_format, ylabel_format)
    commands += template_body_barchart_stacked(stacks, xlabels, ylabels, pattern, index_func, normalized)
    run_gnuplot(commands)
    call(['epstopdf', fig_name + '.eps'])
    call(['rm', fig_name + '.eps'])


def plot_multiple_line(
        xs, ys, title, xlabel, ylabel, legends, fig_name, x_logscale=False,
        y_logscale=False, reference_lines=[], linetypes=None, scale=1,
        width=-3, xtick_font_size=None):

    # if legends:
    #    for i in range(len(legends)):
    #        legends[i] = legends[i].replace('_', '\_')

    commands = template_header(fig_name, scale=scale, xtick_font_size=xtick_font_size)
    xlabel_format = None
    ylabel_format = None
    if x_logscale:
        xlabel_format = '%.0s%c'
    if y_logscale:
        ylabel_format = '%.0s%c'
    commands += template_axis_labels(xlabel,
                                     ylabel, title, x_logscale, y_logscale, xlabel_format, ylabel_format)
    commands += template_body_line(xs, ys, legends=legends,
                              reference_lines=reference_lines, linetypes=linetypes,
                              width=width)
    run_gnuplot(commands)
    call(['epstopdf', fig_name + '.eps'])
    call(['rm', fig_name + '.eps'])


def plot_barchart(heights, labels, xlabel, ylabel, fig_name, title='', y_logscale=False, pattern=3, index_func=None, scale=1):
    # gnuplot '_' is subscript
    commands = template_header(fig_name, scale)
    xlabel_format = None
    ylabel_format = None
    if not y_logscale:
        if max(heights) >= 1000:
            ylabel_format = '%.1s%c'
    else:
        ylabel_format = '%.0s%c'
    commands += template_axis_labels(xlabel,
                                     ylabel, title, False, y_logscale, xlabel_format, ylabel_format)
    commands += template_body_barchart(heights, labels, pattern, index_func)
    run_gnuplot(commands)
    call(['epstopdf', fig_name + '.eps'])
    call(['rm', fig_name + '.eps'])


def plot_histogram(val_list, bins, title, xlabel, ylabel, fig_name, y_logscale=False, x_logscale=False, scale=1):
    commands = template_header(fig_name, scale=scale)
    xlabel_format = None
    ylabel_format = None
    if not y_logscale:
        if max(heights) >= 1000:
            ylabel_format = '%.1s%c'
    else:
        ylabel_format = '%.0s%c'
    if x_logscale:
        xlabel_format = '%.0s%c'
    commands += template_axis_labels(xlabel,
                                     ylabel, title, x_logscale, y_logscale, xlabel_format, ylabel_format)
    commands += template_body_histogram(val_list, bins, x_logscale)
    run_gnuplot(commands)
    call(['epstopdf', fig_name + '.eps'])
    call(['rm', fig_name + '.eps'])