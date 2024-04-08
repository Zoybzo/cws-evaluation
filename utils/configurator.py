# This file is used to load the configuration file and command line parameters.
# Refered to Recbole repo: https://recbole.io/

import sys
import re
import yaml

from loguru import logger


class Config(object):
    def __init__(self, config_file_list):
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.cmd_config_dict = self._load_cmd_line()
        self.config_dict = dict()
        self._merge_external_config_dict()

    def get_config(self):
        return self.config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, "r", encoding="utf-8") as f:
                    file_config_dict.update(
                        yaml.load(f.read(), Loader=self.yaml_loader)
                    )
        return file_config_dict

    def _load_cmd_line(self):
        r"""Read parameters from command line and convert it to str."""
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if (
                    cmd_arg_name in cmd_config_dict
                    and cmd_arg_value != cmd_config_dict[cmd_arg_name]
                ):
                    raise SyntaxError(
                        "There are duplicate commend arg '%s' with different value."
                        % arg
                    )
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger.warning(
                "command line args [{}] will not be used in RecBole".format(
                    " ".join(unrecognized_args)
                )
            )
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type."""
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(
                    value, (str, int, float, list, tuple, dict, bool, Enum)
                ):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.config_dict = external_config_dict

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.config_dict[key] = value

    def __getattr__(self, item):
        if "config_dict" not in self.__dict__:
            raise AttributeError(
                f"'Config' object has no attribute 'final_config_dict'"
            )
        if item in self.config_dict:
            return self.config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.config_dict

    def __str__(self):
        args_info = "\n"
        for arg, value in self.config_dict.items():
            args_info += f"{arg} : {value}\n"
        return args_info

    def __repr__(self):
        return self.__str__()
