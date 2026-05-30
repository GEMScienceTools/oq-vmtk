import re
import pickle


def import_from_pkl(path):
    """
    Function to import data stored in a pickle object
    -----
    Input
    -----
    :param path:           string                Path to the pickle file

    ------
    Output
    ------
    Pickle file
    """

    # import file
    with open(path, 'rb') as file:
        return pickle.load(file)


def export_to_pkl(path, var):
    """
    Function to store data in a pickle object
    -----
    Input
    -----
    :param path:           string                Path to the pickle file
    :param var:          variable                Variable to store
    ------
    Output
    ------
    Pickle file
    """

    # store file
    with open(path, 'wb') as file:
        return pickle.dump(var, file)


def get_num_modes(number_storeys):
    """Return the recommended number of modes for modal analysis."""
    return 1 if number_storeys == 1 else 3


def quick_line_plot(x, y, xlabel, ylabel, color='#399283', lw=1, fontsize=16):
    """Plot a single line with consistent journal-style formatting and display immediately."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, lw=lw)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(visible=True, which='major')
    ax.grid(visible=True, which='minor')
    ax.set_xlim([0.0, max(x)])
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def sorted_alphanumeric(data):
    """
    Function to sort data alphanumerically
    -----
    Input
    -----
    :param data:             list                Data to be sorted
    ------
    Output
    ------
    Sorted data of the same type as "data"
    """

    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)
