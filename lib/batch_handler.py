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

    def get_jobsubmit_cmd(self, scheduler, jobscript, jobname, *jobscript_subs):
        if not os.path.isfile(jobscript):
            raise InvalidArgumentError('`jobscript` file does not exist: {}'.format(jobscript))

        cmd = None
        jobscript_optkey = None

        if scheduler == SCHED_PBS:
            cmd = 'qsub'
            jobscript_optkey = '#PBS'
            if jobscript_subs is not None:
                cmd_subs = ','.join(['p{}="{}"'.format(i+1, a) for i, a in enumerate(jobscript_subs)])
                cmd = r'{} -v {}'.format(cmd, cmd_subs)
            cmd = r'{} -N {}'.format(cmd, jobname)

        elif scheduler == SCHED_SLURM:
            cmd = 'sbatch'
            jobscript_optkey = '#SBATCH'
            if jobscript_subs is not None:
                cmd_subs = ','.join(['p{}="{}"'.format(i+1, a) for i, a in enumerate(jobscript_subs)])
                cmd = r'{} --export={}'.format(cmd, cmd_subs)
            cmd = r'{} -J {}'.format(cmd, jobname)

        if jobscript_optkey is not None:
            jobscript_condoptkey = jobscript_optkey.replace('#', '#CONDOPT_')

            jobscript_condopts = []
            with open(jobscript) as job_script_fp:
                for i, line in enumerate(job_script_fp.readlines()):
                    line_num = i + 1
                    if line.lstrip().startswith(jobscript_condoptkey):

                        cond_ifval = None
                        cond_cond = None
                        cond_elseval = None

                        cond_remain = line.replace(jobscript_condoptkey, '').strip()
                        cond_parts = [s.strip() for s in cond_remain.split(' ELSE ')]
                        if len(cond_parts) == 2:
                            cond_remain, cond_elseval = cond_parts
                        cond_parts = [s.strip() for s in cond_remain.split(' IF ')]
                        if len(cond_parts) == 2:
                            cond_ifval, cond_cond = cond_parts

                        try:
                            condopt_add = None

                            if cond_ifval is not None and cond_cond is not None:
                                if self.jobscript_condopt_eval(cond_cond, eval):
                                    condopt_add = self.jobscript_condopt_eval(cond_ifval, str)
                                elif cond_elseval is not None:
                                    condopt_add = self.jobscript_condopt_eval(cond_elseval, str)
                            elif cond_elseval is not None:
                                raise SyntaxError
                            else:
                                condopt_add = self.jobscript_condopt_eval(cond_remain, str)

                            if condopt_add is not None:
                                jobscript_condopts.append(condopt_add)

                        except SyntaxError:
                            raise InvalidArgumentError(' '.join([
                                "Invalid syntax in jobscript conditional option:",
                                "\n  File '{}', line {}: '{}'".format(jobscript, line_num, line.rstrip()),
                                "\nProper conditional option syntax is as follows:",
                                "'{} <options> [IF <conditional> [ELSE <options>]]'".format(jobscript_condoptkey)
                            ]))

            if jobscript_condopts:
                cmd = r'{} {}'.format(cmd, ' '.join(jobscript_condopts))

        cmd = r'{} "{}"'.format(cmd, jobscript)

        return cmd

    def jobscript_condopt_eval(self, condopt_expr, out_type):
        if out_type not in (str, eval):
            raise InvalidArgumentError("`out_type` must be either str or eval")
        vars_dict = self.vars_dict
        for varstr in sorted(vars_dict.keys(), key=len, reverse=True):
            possible_substr = {'%'+s for s in [varstr, self.varstr2argstr[varstr], self.varstr2argstr[varstr].lstrip('-')]}
            possible_substr = possible_substr.union({s.lower() for s in possible_substr}, {s.upper() for s in possible_substr})
            for substr in possible_substr:
                if substr in condopt_expr:
                    replstr = str(vars_dict[varstr]) if out_type is str else "vars_dict['{}']".format(varstr)
                    condopt_expr = condopt_expr.replace(substr, replstr)
                    break
        return out_type(condopt_expr)


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
