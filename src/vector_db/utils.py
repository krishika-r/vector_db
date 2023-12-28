import os
import fsspec
import re
import yaml
from functools import partial


def load_yml(path, *, fs=None, **kwargs):
    """Load a yml file from the input `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        dictionary of the loaded yml file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return yaml.safe_load(fp, **kwargs)

def config_init(
    user_config_path=None,
    data_config_path=None,
    model_config_path=None,
):
    """
    Initialize the configs.

    Parameters
    ----------
    user_config_path : str
        The path to the user config file.
    data_config_path : str
        The path to the data config file.
    model_config_path : str
        The path to the model config file.

    Returns
    -------
    dict
        Dotified dictionary of user_config, data_config, model_config.

    Raises
    ------
    ValueError
        If any of the config paths are None.
    """
    if user_config_path is None:
        raise ValueError("user_config path is a mandatory argument.")
    # load user config
    user_config = load_config(cfg_file=user_config_path)

    if data_config_path is None:
        raise ValueError("data_config path is a mandatory argument.")
    # load data config
    data_config = load_config(cfg_file=data_config_path)

    if model_config_path is None:
        raise ValueError("model_config path is a mandatory argument.")
    # load model config
    model_config = load_config(cfg_file=model_config_path)
    
    return user_config, data_config, model_config


def load_config(cfg_file, fs=None):
    """Create the Context from a config file location path.

    Parameters
    ----------
    path : str
        Location path of the .yaml config file.
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    Dotified dictionary with all the config parameters.
    """
    fs = fs or fsspec.filesystem("file")
    if fs.exists(cfg_file):

        def _dotted_access_getter(key, dct):
            for k in key.split("."):
                dct = dct[k]
            return dct

        def _repl_fn(match_obj, getter):
            return getter(match_obj.groups()[0])

        def _interpolate(val, repl_fn):
            if isinstance(val, dict):
                return {k: _interpolate(v, repl_fn) for k, v in val.items()}
            elif isinstance(val, list):
                return [_interpolate(v, repl_fn) for v in val]
            elif isinstance(val, str):
                # We shouldn't replace slashes in url's/ links. In out case api_base link is a HTTP and we shouldn't replace `/` with os specific slash in it
                # if not validators.url(val):
                #     val = val.replace(pp.sep, os.path.sep)
                return re.sub(r"\$\{([\w|.]+)\}", repl_fn, val)
            else:
                return val

        cfg = load_yml(cfg_file, fs=fs)

        cfg = _interpolate(
            cfg,
            partial(_repl_fn, getter=partial(_dotted_access_getter, dct=cfg)),
        )

        return cfg
    else:
        raise ValueError(f"{cfg_file} is not a valid config file.")


def clean_text(documents):
    cleaned_texts = []

    for document in documents:
        cleaned_text = document.page_content.replace('\n', ' ').strip()
        # Remove URLs
        cleaned_text = re.sub(r'http\S+', '', cleaned_text)

        # # Remove special characters and symbols
        # cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

        # Remove extra whitespaces
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_texts.append(cleaned_text)

    return cleaned_texts

def check_input_path(input_path):
    """
    Check if the input path is a folder and contains PDF files, or if it's a PDF file.

    Parameters:
    - input_path (str): Path to the folder or PDF file.

    Raises:
    - ValueError: If the input is not a folder and does not end with '.pdf'.
    """
    if os.path.isdir(input_path):
        pdf_files = [file for file in os.listdir(input_path) if file.lower().endswith('.pdf')]
        if not pdf_files:
            raise ValueError(f"{input_path} is a folder but does not contain any PDF files.")
    elif not input_path.lower().endswith('.pdf'):
        raise ValueError(f"{input_path} is not a valid folder path and does not end with '.pdf'.")

