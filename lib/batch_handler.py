#!/usr/bin/env python2

# Version 0.9; Erik Husby; Polar Geospatial Center, University of Minnesota; 2019


import argparse
import copy
import math
import numpy as np
import os
import subprocess
from datetime import datetime
from glob import glob


SCHED_SUPPORTED = []
SCHED_PBS = 'pbs'
SCHED_SLURM = 'slurm'
SCHED_NAME_TESTCMD_DICT = {
    SCHED_PBS: 'pbsnodes',
    SCHED_SLURM: 'sinfo'
}
for sched_name in sorted(SCHED_NAME_TESTCMD_DICT.keys()):
    try:
        child = subprocess.Popen(SCHED_NAME_TESTCMD_DICT[sched_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdoutdata, stderrdata = child.communicate()
        if child.returncode == 0:
            SCHED_SUPPORTED.append(sched_name)
    except OSError:
        pass
if len(SCHED_SUPPORTED) == 0:
    SCHED_SUPPORTED.append(None)


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


class ArgumentPasser:

    def __init__(self, executable_path, script_path, parser, sys_argv=[''], parse=True):
        self.exe = executable_path
        self.script = script_path
        self.parser = parser
        self.sys_argv = list(sys_argv)
        self.script_run_cmd = ' '.join(self.sys_argv)
        self.parsed = parse

        self.argstr2varstr = self._make_argstr2varstr_dict()
        self.varstr2argstr = self._make_varstr2argstr_dict()
        self.varstr2action = self._make_varstr2action_dict()
        self.argstr_pos = self._find_pos_args()
        self.provided_opt_args = self._find_provided_opt_args()

        if parse:
            self.vars = self.parser.parse_args()
            self.vars_dict = vars(self.vars)
        else:
            self.vars = None
            self.vars_dict = {varstr: None for varstr in self.varstr2argstr}

        self._fix_bool_plus_args()

        self.cmd_optarg_base = None
        self.cmd = None
        self._update_cmd_base()

    def __deepcopy__(self, memodict={}):
        args = ArgumentPasser(self.exe, self.script, self.parser, self.sys_argv, self.parsed)
        args.vars_dict = copy.deepcopy(self.vars_dict)
        return args

    def get_as_list(self, *argstrs):
        if len(argstrs) < 1:
            raise InvalidArgumentError("One or more argument strings must be provided")
        elif len(argstrs) == 1 and type(argstrs[0]) in (list, tuple):
            argstrs = argstrs[0]
        argstrs_invalid = set(argstrs).difference(set(self.argstr2varstr))
        if argstrs_invalid:
            raise InvalidArgumentError("This {} object does not have the following "
                                       "argument strings: {}".format(type(self).__name__, list(argstrs_invalid)))
        values = [self.vars_dict[self.argstr2varstr[argstr]] for argstr in argstrs]
        return values

    def get(self, *argstrs):
        values = self.get_as_list(*argstrs)
        if len(values) == 1:
            values = values[0]
        return values

    def set(self, argstrs, newval=None):
        argstr_list = argstrs if type(argstrs) in (tuple, list) else [argstrs]
        for argstr in argstr_list:
            if argstr not in self.argstr2varstr:
                raise InvalidArgumentError("This {} object has no '{}' argument string".format(type(self).__name__, argstr))
            if newval is None:
                action = self.varstr2action[self.argstr2varstr[argstr]]
                acttype = type(action)
                if acttype is argparse._StoreAction and 'function argtype_bool_plus' in str(action.type):
                    newval = True
                elif acttype in (argparse._StoreTrueAction, argparse._StoreFalseAction):
                    newval = (acttype is argparse._StoreTrueAction)
                # else:
                #     raise InvalidArgumentError("Setting non-boolean argument string '{}' requires "
                #                                "a provided `newval` value".format(argstr))
            self.vars_dict[self.argstr2varstr[argstr]] = newval
        if set(argstr_list).issubset(set(self.argstr_pos)):
            self._update_cmd()
        else:
            self._update_cmd_base()

    def unset(self, *argstrs):
        if len(argstrs) < 1:
            raise InvalidArgumentError("One or more argument strings must be provided")
        elif len(argstrs) == 1 and type(argstrs[0]) in (list, tuple):
            argstrs = argstrs[0]
        for argstr in argstrs:
            action = self.varstr2action[self.argstr2varstr[argstr]]
            acttype = type(action)
            if acttype is argparse._StoreAction and 'function argtype_bool_plus' in str(action.type):
                newval = False
            elif acttype in (argparse._StoreTrueAction, argparse._StoreFalseAction):
                newval = (acttype is argparse._StoreFalseAction)
            else:
                newval = None
            self.vars_dict[self.argstr2varstr[argstr]] = newval
        if set(argstrs).issubset(set(self.argstr_pos)):
            self._update_cmd()
        else:
            self._update_cmd_base()

    def provided(self, argstr):
        return argstr in self.provided_opt_args

    def _make_argstr2varstr_dict(self):
        argstr2varstr = {}
        for act in self.parser._actions:
            if len(act.option_strings) == 0:
                argstr2varstr[act.dest.replace('_', '-')] = act.dest
            else:
                for os in act.option_strings:
                    argstr2varstr[os] = act.dest
        return argstr2varstr

    def _make_varstr2argstr_dict(self):
        varstr2argstr = {}
        for act in self.parser._actions:
            if len(act.option_strings) == 0:
                varstr2argstr[act.dest] = act.dest.replace('_', '-')
            else:
                varstr2argstr[act.dest] = sorted(act.option_strings)[0]
        return varstr2argstr

    def _make_varstr2action_dict(self):
        return {act.dest: act for act in self.parser._actions}

    def _find_pos_args(self):
        return [act.dest.replace('_', '-') for act in self.parser._actions if len(act.option_strings) == 0]

    def _find_provided_opt_args(self):
        provided_opt_args = []
        for token in self.sys_argv:
            potential_argstr = token.split('=')[0]
            if potential_argstr in self.argstr2varstr:
                provided_opt_args.append(self.varstr2argstr[self.argstr2varstr[potential_argstr]])
        return provided_opt_args

    def _fix_bool_plus_args(self):
        for varstr in self.vars_dict:
            argstr = self.varstr2argstr[varstr]
            action = self.varstr2action[varstr]
            if 'function argtype_bool_plus' in str(action.type) and self.get(argstr) is None:
                self.set(argstr, (argstr in self.provided_opt_args))

    def _argval2str(self, item):
        if type(item) is str:
            if item.startswith('"') and item.endswith('"'):
                item_str = item
            elif item.startswith("'") and item.endswith("'"):
                item_str = item
            else:
                item_str = '"{}"'.format(item)
        else:
            item_str = '{}'.format(item)
        return item_str

    def _update_cmd_base(self):
        arg_list = []
        for varstr, val in self.vars_dict.items():
            argstr = self.varstr2argstr[varstr]
            if argstr not in self.argstr_pos and val is not None:
                if isinstance(val, bool):
                    action = self.varstr2action[varstr]
                    acttype = type(action)
                    if acttype is argparse._StoreAction:
                        if 'function argtype_bool_plus' in str(action.type) and val is True:
                            arg_list.append(argstr)
                    elif (   (acttype is argparse._StoreTrueAction and val is True)
                          or (acttype is argparse._StoreFalseAction and val is False)):
                        arg_list.append(argstr)
                elif isinstance(val, list) or isinstance(val, tuple):
                    arg_list.append('{} {}'.format(argstr, ' '.join([self._argval2str(item) for item in val])))
                else:
                    arg_list.append('{} {}'.format(argstr, self._argval2str(val)))
        self.cmd_optarg_base = ' '.join(arg_list)
        self._update_cmd()

    def _update_cmd(self):
        posarg_list = []
        for argstr in self.argstr_pos:
            varstr = self.argstr2varstr[argstr]
            val = self.vars_dict[varstr]
            if val is not None:
                if isinstance(val, list) or isinstance(val, tuple):
                    posarg_list.append(' '.join([self._argval2str(item) for item in val]))
                else:
                    posarg_list.append(self._argval2str(val))
        self.cmd = '{} {} {} {}'.format(self.exe, self.script, " ".join(posarg_list), self.cmd_optarg_base)

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
                                if self._jobscript_condopt_eval(cond_cond, eval):
                                    condopt_add = self._jobscript_condopt_eval(cond_ifval, str)
                                elif cond_elseval is not None:
                                    condopt_add = self._jobscript_condopt_eval(cond_elseval, str)
                            elif cond_elseval is not None:
                                raise SyntaxError
                            elif cond_remain.startswith('import') or cond_remain.startswith('from'):
                                exec(cond_remain)
                            else:
                                condopt_add = self._jobscript_condopt_eval(cond_remain, str)

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

    def _jobscript_condopt_eval(self, condopt_expr, out_type):
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


def argtype_bool_plus(value, parse_fn=None):
    if parse_fn is not None:
        return parse_fn(value)
    return value


def write_task_bundles(task_list, tasks_per_bundle, dstdir, descr, task_fmt='%s', task_delim=' '):
    bundle_prefix = os.path.join(dstdir, '{}_{}'.format(descr, datetime.now().strftime("%Y%m%d%H%M%S")))
    jobnum_total = int(math.ceil(len(task_list) / float(tasks_per_bundle)))
    jobnum_fmt = '{:0>'+str(len(str(jobnum_total)))+'}'
    bundle_file_list = []
    print("Writing task bundle text files in directory: {}".format(dstdir))
    for jobnum, tasknum in enumerate(range(0, len(task_list), tasks_per_bundle)):
        bundle_file = '{}_{}.txt'.format(bundle_prefix, jobnum_fmt.format(jobnum+1))
        np.savetxt(bundle_file, task_list[tasknum:tasknum+tasks_per_bundle], fmt=task_fmt, delimiter=task_delim)
        bundle_file_list.append(bundle_file)
    return bundle_file_list


def read_task_bundle(bundle_file, args_dtype=np.dtype(str), args_delim=' '):
    task_list = np.loadtxt(bundle_file, dtype=args_dtype, delimiter=args_delim).tolist()
    if type(task_list) is not list:
        task_list = [task_list]
    return task_list


def get_jobnum_fmtstr(processing_list, min_digits=3):
    return '{:0>'+str(max(min_digits, len(str(len(processing_list)))))+'}'


def exec_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    (so, se) = p.communicate()
    rc = p.wait()
    print("RETURN CODE: {}".format(rc))
    if so != '':
        print("STDOUT:\n{}".format(so.rstrip()))
    if se != '':
        print("STDERR:\n{}".format(se.rstrip()))
    return rc
