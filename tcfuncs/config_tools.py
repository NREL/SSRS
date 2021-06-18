import errno
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from typing import Optional


def parse_config_from_args(
        args: [str],
        default_config: {},
) -> {}:
    """
    Makes a configuration map given a list of (command line) override args and a default configuration
    """
    config = deepcopy(default_config)

    arg = ''.join(args).strip()
    # print('c1: ', arg)

    # add enclosing brackets if they are missing
    if not arg.startswith('{'):
        arg = '{' + arg + '}'

    # print('c2: ', arg)

    # convert bare true, false, and null's to lowercase
    arg = re.sub(r'(?i)(?<=[\t :{},])["\']?(true|false|null)["\']?(?=[\t :{},])',
                 lambda match: match.group(0).lower(),
                 arg)
    # print('c4: ', arg)

    # replace bare or single quoted strings with quoted ones
    arg = re.sub(
        r'(?<=[\t :{},])["\']?(((?<=")((?!(?<!\\)").)*(?="))|(?<=\')((?!\').)*(?=\')|(?!(true|false|null).)(['
        r'a-zA-Z_][a-zA-Z_0-9]*))["\']?(?=[\t :{},])',
        r'"\1"',
        arg)

    # print('c5: ', arg)

    overrides = json.loads(arg)
    config = merge_configs(config, overrides)
    return config


def merge_configs(target, overrides):
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in target:
                target[key] = merge_configs(target[key], value)
            else:
                target[key] = value
        return target
    else:
        return overrides


def makedir_if_not_exists(filename: str) -> None:
    try:
        os.makedirs(filename)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_run(
        default_config: {},
        output_path: Optional[str] = None,
        run_suffix: Optional[str] = None,
        place_in_subdir: bool = True,
        append_date_to_run_path=False,
) -> ({}, str):
    '''
    Sets up a config and logging prefix for this run. Writes the config to a log file.
    :param default_config: the base configuration that the command line params override
    :param output_path: path to place output files including logs and configuration record. If None, './output' is used.
    :param run_suffix: appended to the run name. If None, then the ISO 8601 datetime is used with '.' instead of ':'.
    :param place_in_subdir: True to output into a subdir of output_path, False to directly into output_path
    :param parse_command_line whether or not to parse the command line as a json config and merge it into the config


    + one function that just does everything
    + one function that does one or the other thing

    parse command line as json?
    parse config from command line?
    both?


    :return: config, output_path, run_name
    '''

    config = parse_and_merge_command_line_config(default_config)

    run_name = config['run_name']
    if append_date_to_run_path:
        run_suffix = '_' + datetime.now().isoformat().replace(':',
                                                              '.') if run_suffix is None else run_suffix
        run_name += run_suffix

    output_path = os.path.join(
        os.path.curdir, 'output') if output_path is None else output_path
    output_path = output_path if place_in_subdir and run_name is None else os.path.join(
        output_path, run_name)
    makedir_if_not_exists(output_path)

    print('setup_run() run_name: "' + run_name + '"')
    print('setup_run() output_path: "' + output_path + '"')
    # print('setup_run() config:')
    # pprint(config)

    write_config_log(config, output_path, run_name)
    print('setup_run() complete.')
    return config, output_path, run_name


def parse_and_merge_command_line_config(config: {}):
    args = sys.argv
    num_args = len(args)
    if num_args > 2:
        offset = 1
        if args[offset] == 'file_name':
            if num_args <= 2:
                raise Exception('Unrecognized Command!')
            filename = args[offset + 1]
            offset += 2
            file_config = {}
            try:
                with open(filename) as json_file:
                    file_config = json.load(json_file)
            except:
                raise Exception('Unrecognized filename!')
            config = merge_configs(config, file_config)
        elif args[offset] == 'run_name':
            if num_args <= 2:
                raise Exception('Unrecognized Command!')
            run_name = args[offset + 1]
            filename = os.path.join(os.path.curdir, 'output/') + run_name + '/' + \
                       run_name + '.json'
            offset += 2
            file_config = {}
            try:
                with open(filename) as json_file:
                    file_config = json.load(json_file)
            except:
                raise Exception('Unrecognized run_name!')
            config = merge_configs(config, file_config)
        else:
            raise Exception(
                'Unrecognized argument!\nOptions: file_name, run_name')
        if num_args > offset:
            config = parse_config_from_args(args[offset, :], config)

    return config


# def parse_and_merge_command_line_config(config: {}, offset: int = 1):
#     args = sys.argv
#     if len(args) <= offset:  # nothing to parse
#         return config
#     return parse_config_from_args(args[offset, :], config)


def write_config_log(
        config: {},
        output_path: str,
        config_name: str = 'config') -> None:
    '''
    Writes a json log file containing the configuration of this run
    :param output_path: where to write the file
    :param run_name: prefix of the filename
    :param config_name: suffix of the filename
    '''
    config_filename = os.path.join(output_path, config_name + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_config_log_old(
        config: {},
        output_path: str,
        run_name: str,
        config_name: str = '_config') -> None:
    '''
    Writes a json log file containing the configuration of this run
    :param output_path: where to write the file
    :param run_name: prefix of the filename
    :param config_name: suffix of the filename
    '''
    config_filename = os.path.join(
        output_path, run_name + config_name + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)


# Set up the config structure
def setup_env(base_config: Optional[dict] = None) -> SimpleNamespace:
    # config = config_tools.parse_config_from_args(sys.argv[1:], base_config)
    config, output_path, run_name = setup_run(base_config)
    config['run_name'] = run_name
    config['output_dir'] = output_path
    config['data_dir'] = os.path.join(output_path, 'data/')
    config['fig_dir'] = os.path.join(output_path, 'figs/')
    # pprint(config)
    # make the dirs if they don't exist
    for key in ['output_dir', 'data_dir', 'fig_dir']:
        makedir_if_not_exists(config[key])
    return SimpleNamespace(**config)


# initalizing config for scripts
def initialize_code(fname: str, in_config):
    print('\n--- Initializing {0:s}'.format(fname))
    try:
        config: SimpleNamespace = setup_env(in_config)
    except:
        raise Exception('Unrecognized config name!\nSee config.py')
    return config
