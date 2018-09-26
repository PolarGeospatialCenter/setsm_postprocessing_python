#!/usr/bin/env python2

# Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import math
import numpy as np
import os
from datetime import datetime
from glob import glob


SCHED_QSUB = 'qsub'
SCHED_SLURM = 'slurm'
SCHED_SUPPORTED = [
    SCHED_QSUB,
    SCHED_SLURM
]


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

    def get(self, argstr):
        return self.vars_dict[self.argstr2varstr[argstr]]

    def set(self, argstr, newval):
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
        return '"{}"'.format(item) if type(item) is str else item

    def update_cmd_base(self):
        arg_list = []
        for varstr, val in self.vars_dict.iteritems():
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
        arg_list = []
        for argstr in self.argstr_pos:
            varstr = self.argstr2varstr[argstr]
            val = self.vars_dict[varstr]
            if val is not None:
                if isinstance(val, list) or isinstance(val, tuple):
                    arg_list.append(' '.join([self.argval2str(item) for item in val]))
                else:
                    arg_list.append(self.argval2str(val))
        self.cmd = '{} {} {}'.format(self.exe, self.script, self.cmd_optarg_base, " ".join(arg_list))

    def get_cmd(self):
        return self.cmd


def write_task_bundles(task_list, tasks_per_bundle, dstdir, descr, task_fmt='%s', task_delim=' '):
    bundle_prefix = os.path.join(dstdir, '{}_{}_'.format(descr, datetime.now().strftime("%Y%m%d%H%M%S")))
    jobnum_total = int(math.ceil(len(task_list) / float(tasks_per_bundle)))
    jobnum_fmt = '{:0>'+str(len(str(jobnum_total)))+'}'
    for i in range(0, len(task_list), tasks_per_bundle):
        bundle_file = '{}_{}.txt'.format(bundle_prefix, jobnum_fmt.format(i+1))
        np.savetxt(bundle_file, task_list[i:i+tasks_per_bundle], fmt=task_fmt, delimiter=task_delim)
    bundle_files = glob(bundle_prefix+'*')
    return bundle_files


def read_task_bundle(bundle_file, task_dtype=np.dtype(str), task_delim=' '):
    return np.loadtxt(bundle_file, dtype=task_dtype, delimiter=task_delim)


def get_jobsubmit_cmd(scheduler, job_script, job_abbrev, *script_args):
    cmd = None

    if scheduler == SCHED_QSUB:
        cmd = 'qsub "{}" -N {}'.format(job_script, job_abbrev)
        if script_args is not None:
            cmd_scriptargs = ','.join(['p{}="{}"'.format(i+1, a) for i, a in enumerate(script_args)])
            cmd = '{} -v {}'.format(cmd, cmd_scriptargs)

    elif scheduler == SCHED_SLURM:
        cmd = 'sbatch "{}" -J {}'.format(job_script, job_abbrev)
        if script_args is not None:
            cmd_scriptargs = ' '.join(['--export=p{}="{}"'.format(i+1, a) for i, a in enumerate(script_args)])
            cmd = '{} {}'.format(cmd, cmd_scriptargs)

    return cmd
