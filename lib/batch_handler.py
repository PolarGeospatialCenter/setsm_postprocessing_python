#!/usr/bin/env python2

# Version 0.9; Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import math
import numpy as np
import os
from datetime import datetime
from glob import glob


SCHED_PBS = 'pbs'
SCHED_SLURM = 'slurm'
SCHED_SUPPORTED = [
    SCHED_PBS,
    SCHED_SLURM
]


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


class ArgumentPasser:

    def __init__(self, parser, executable_path, script_path):
        self.parser = parser
        self.exe = executable_path
        self.script = script_path
        self.vars = parser.parse_args()
        self.vars_dict = vars(self.vars)
        self.argstr2varstr = self.get_argstr2varstr_dict()
        self.varstr2argstr = self.get_varstr2argstr_dict()
        self.varstr2acttype = self.get_varstr2acttype_dict()
        self.argstr_pos = self.get_pos_args()
        self.cmd_optarg_base = None
        self.cmd = None
        self.update_cmd_base()

    def get(self, *argstrs):
        if len(argstrs) < 1:
            raise InvalidArgumentError("One or more argument strings must be provided")
        argstrs_invalid = set(argstrs).difference(set(self.argstr2varstr))
        if argstrs_invalid:
            raise InvalidArgumentError("This {} object does not have the following "
                                       "argument strings: {}".format(type(self).__name__, list(argstrs_invalid)))
        values = [self.vars_dict[self.argstr2varstr[argstr]] for argstr in argstrs]
        if len(values) == 1:
            values = values[0]
        return values

    def set(self, argstr, newval):
        if argstr not in self.argstr2varstr:
            raise InvalidArgumentError("This {} object has no '{}' argument string".format(type(self).__name__, argstr))
        self.vars_dict[self.argstr2varstr[argstr]] = newval
        if argstr in self.argstr_pos:
            self.update_cmd()
        else:
            self.update_cmd_base()

    def remove_args(self, *argstrs):
        for argstr in argstrs:
            self.vars_dict[self.argstr2varstr[argstr]] = None
        if set(argstrs).issubset(set(self.argstr_pos)):
            self.update_cmd()
        else:
            self.update_cmd_base()

    def get_pos_args(self):
        return [act.dest.replace('_', '-') for act in self.parser._actions if len(act.option_strings) == 0]

    def get_argstr2varstr_dict(self):
        argstr2varstr = {}
        for act in self.parser._actions:
            if len(act.option_strings) == 0:
                argstr2varstr[act.dest.replace('_', '-')] = act.dest
            else:
                for os in act.option_strings:
                    argstr2varstr[os] = act.dest
        return argstr2varstr

    def get_varstr2argstr_dict(self):
        varstr2argstr = {}
        for act in self.parser._actions:
            if len(act.option_strings) == 0:
                varstr2argstr[act.dest] = act.dest.replace('_', '-')
            else:
                varstr2argstr[act.dest] = sorted(act.option_strings)[0]
        return varstr2argstr

    def get_varstr2acttype_dict(self):
        return {act.dest: type(act) for act in self.parser._actions}

    def argval2str(self, item):
        return '"{}"'.format(item) if type(item) is str else '{}'.format(item)

    def update_cmd_base(self):
        arg_list = []
        for varstr, val in self.vars_dict.items():
            argstr = self.varstr2argstr[varstr]
            if argstr not in self.argstr_pos and val is not None:
                if isinstance(val, bool):
                    acttype = self.varstr2acttype[varstr]
                    if (   (acttype is argparse._StoreTrueAction and val is True)
                        or (acttype is argparse._StoreFalseAction and val is False)):
                        arg_list.append(argstr)
                elif isinstance(val, list) or isinstance(val, tuple):
                    arg_list.append('{} {}'.format(argstr, ' '.join([self.argval2str(item) for item in val])))
                else:
                    arg_list.append('{} {}'.format(argstr, self.argval2str(val)))
        self.cmd_optarg_base = ' '.join(arg_list)
        self.update_cmd()

    def update_cmd(self):
        posarg_list = []
        for argstr in self.argstr_pos:
            varstr = self.argstr2varstr[argstr]
            val = self.vars_dict[varstr]
            if val is not None:
                if isinstance(val, list) or isinstance(val, tuple):
                    posarg_list.append(' '.join([self.argval2str(item) for item in val]))
                else:
                    posarg_list.append(self.argval2str(val))
        self.cmd = '{} {} {} {}'.format(self.exe, self.script, self.cmd_optarg_base, " ".join(posarg_list))

    def get_cmd(self):
        return self.cmd


def write_task_bundles(task_list, tasks_per_bundle, dstdir, descr, task_fmt='%s', task_delim=' '):
    bundle_prefix = os.path.join(dstdir, '{}_{}'.format(descr, datetime.now().strftime("%Y%m%d%H%M%S")))
    jobnum_total = int(math.ceil(len(task_list) / float(tasks_per_bundle)))
    jobnum_fmt = '{:0>'+str(len(str(jobnum_total)))+'}'
    for jobnum, tasknum in enumerate(range(0, len(task_list), tasks_per_bundle)):
        bundle_file = '{}_{}.txt'.format(bundle_prefix, jobnum_fmt.format(jobnum+1))
        np.savetxt(bundle_file, task_list[tasknum:tasknum+tasks_per_bundle], fmt=task_fmt, delimiter=task_delim)
    bundle_files = glob(bundle_prefix+'*')
    return bundle_files


def read_task_bundle(bundle_file, task_dtype=np.dtype(str), task_delim=' '):
    return np.loadtxt(bundle_file, dtype=task_dtype, delimiter=task_delim)


def get_jobnum_fmtstr(processing_list, min_digits=3):
    return '{:0>'+str(max(min_digits, len(str(len(processing_list)))))+'}'


def get_jobsubmit_cmd(scheduler, job_script, job_name, *job_script_args):
    cmd = None

    if scheduler == SCHED_PBS:
        cmd = 'qsub'
        if job_script_args is not None:
            cmd_scriptargs = ','.join(['p{}="{}"'.format(i+1, a) for i, a in enumerate(job_script_args)])
            cmd = r'{} -v {}'.format(cmd, cmd_scriptargs)
        cmd = r'{} -N {} "{}"'.format(cmd, job_name, job_script)

    elif scheduler == SCHED_SLURM:
        cmd = 'sbatch'
        if job_script_args is not None:
            cmd_scriptargs = ' '.join(['--export=p{}="{}"'.format(i+1, a) for i, a in enumerate(job_script_args)])
            cmd = r'{} {}'.format(cmd, cmd_scriptargs)
        cmd = r'{} -J {} "{}"'.format(cmd, job_name, job_script)

    return cmd
