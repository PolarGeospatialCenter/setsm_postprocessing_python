
# Version 1.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2019


from __future__ import print_function
import argparse
import contextlib
import copy
import functools
import math
import numpy as np
import operator
import os
import platform
import smtplib
import subprocess
import sys
from datetime import datetime
from email.mime.text import MIMEText


PYTHON_VERSION_REQUIRED_MIN = "2.7"  # supports multiple dot notation


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class VersionError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class DeveloperError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class ScriptArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class ExternalError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


class VersionString:
    def __init__(self, ver_str_or_num):
        self.ver_str = str(ver_str_or_num)
        self.ver_num_list = [int(n) for n in self.ver_str.split('.')]
    def get_comparable_lists(self, other):
        this_list = list(self.ver_num_list)
        other_list = list(other.ver_num_list)
        if len(this_list) < len(other_list):
            this_list.extend([0]*(len(other_list)-len(this_list)))
        elif len(this_list) > len(other_list):
            other_list.extend([0]*(len(this_list)-len(other_list)))
        return this_list, other_list
    def __str__(self):
        return self.ver_str
    def __repr__(self):
        return self.ver_str
    def __compare_absolute(self, other, inequality=False):
        this_ver_num_list, other_ver_num_list = self.get_comparable_lists(other)
        for i in range(len(this_ver_num_list)):
            if this_ver_num_list[i] != other_ver_num_list[i]:
                return inequality
        return (not inequality)
    def __compare_relative(self, other, op, allow_equal=False):
        this_ver_num_list, other_ver_num_list = self.get_comparable_lists(other)
        for i in range(len(this_ver_num_list)):
            if this_ver_num_list[i] != other_ver_num_list[i]:
                return op(this_ver_num_list[i], other_ver_num_list[i])
        return allow_equal
    def __eq__(self, other):
        return self.__compare_absolute(other, inequality=False)
    def __ne__(self, other):
        return self.__compare_absolute(other, inequality=True)
    def __gt__(self, other):
        return self.__compare_relative(other, operator.gt, allow_equal=False)
    def __ge__(self, other):
        return self.__compare_relative(other, operator.gt, allow_equal=True)
    def __lt__(self, other):
        return self.__compare_relative(other, operator.lt, allow_equal=False)
    def __le__(self, other):
        return self.__compare_relative(other, operator.le, allow_equal=True)

PYTHON_VERSION = VersionString(platform.python_version())
if PYTHON_VERSION < VersionString(PYTHON_VERSION_REQUIRED_MIN):
    raise VersionError("Python version ({}) is below required minimum ({})".format(
        PYTHON_VERSION, PYTHON_VERSION_REQUIRED_MIN))

if PYTHON_VERSION < VersionString(3):
    from StringIO import StringIO
else:
    from io import StringIO


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


@contextlib.contextmanager
def capture_stdout_stderr():
    oldout, olderr = sys.stdout, sys.stderr
    out = [StringIO(), StringIO()]
    try:
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


def exec_cmd(cmd, strip_returned_stdout=False, suppress_stdout_in_success=False):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    (so, se) = p.communicate()
    rc = p.wait()
    if rc != 0:
        print("RETURN CODE: {}".format(rc))
    if so != '' and (not suppress_stdout_in_success or rc != 0):
        print("STDOUT:\n{}".format(so.rstrip()))
    if se != '':
        print("STDERR:\n{}".format(se.rstrip()))
    return rc, so.strip() if strip_returned_stdout else so, se


def send_email(to_addr, subject, body, from_addr=None):
    if from_addr is None:
        platform_node = platform.node()
        from_addr = platform_node if platform_node is not None else 'your-computer'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    s = smtplib.SMTP('localhost')
    s.sendmail(to_addr, [to_addr], msg.as_string())
    s.quit()


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter): pass

def argtype_path_handler(path, argstr,
                         abspath_fn=os.path.realpath,
                         existcheck_fn=None, existcheck_reqval=None,
                         accesscheck_reqtrue=None, accesscheck_reqfalse=None):
    path = os.path.expanduser(path)
    existcheck_fn_desc_dict = {
        os.path.isfile: 'file',
        os.path.isdir: 'directory',
        os.path.exists: 'file/directory'
    }
    accesscheck_perm_desc_list = [
        [os.F_OK, 'existing'],
        [os.R_OK, 'readable'],
        [os.W_OK, 'writeable'],
        [os.X_OK, 'executable']
    ]
    pathtype_desc = existcheck_fn_desc_dict[existcheck_fn] if existcheck_fn else 'path'
    if accesscheck_reqtrue is None:
        accesscheck_reqtrue = []
    if accesscheck_reqfalse is None:
        accesscheck_reqfalse = []
    if type(accesscheck_reqtrue) not in (set, tuple, list):
        accesscheck_reqtrue = [accesscheck_reqtrue]
    if type(accesscheck_reqfalse) not in (set, tuple, list):
        accesscheck_reqfalse = [accesscheck_reqfalse]
    accesscheck_reqtrue = set(accesscheck_reqtrue)
    accesscheck_reqfalse = set(accesscheck_reqfalse)
    perms_overlap = set(accesscheck_reqtrue).intersection(accesscheck_reqfalse)
    if len(perms_overlap) > 0:
        raise DeveloperError("The following permission settings (`os.access` modes)"
                             " appear in both required True and False lists: {}".format(perms_overlap))
    if existcheck_fn is not None and existcheck_fn(path) != existcheck_reqval:
        existresult_desc = 'does not exist' if existcheck_reqval is True else 'already exists'
        raise ScriptArgumentError("argument {}: {} {}".format(argstr, pathtype_desc, existresult_desc))
    access_desc_reqtrue_list = [perm_descr for perm, perm_descr in accesscheck_perm_desc_list if perm in accesscheck_reqtrue]
    access_desc_reqfalse_list = [perm_descr for perm, perm_descr in accesscheck_perm_desc_list if perm in accesscheck_reqfalse]
    access_desc_reqtrue_err_list = [perm_descr for perm, perm_descr in accesscheck_perm_desc_list if perm in accesscheck_reqtrue and os.access(path, perm) is not True]
    access_desc_reqfalse_err_list = [perm_descr for perm, perm_descr in accesscheck_perm_desc_list if perm in accesscheck_reqfalse and os.access(path, perm) is not False]
    if len(access_desc_reqtrue_err_list) > 0 or len(access_desc_reqfalse_err_list) > 0:
        errmsg = ' '.join([
            "{} must".format(pathtype_desc),
            (len(access_desc_reqtrue_list) > 0)*"be ({})".format(' & '.join(access_desc_reqtrue_list)),
            "and" if (len(access_desc_reqtrue_list) > 0 and len(access_desc_reqfalse_list) > 0) else '',
            (len(access_desc_reqfalse_list) > 0)*"not be ({})".format(', '.join(access_desc_reqfalse_list)),
            ", but it",
            (len(access_desc_reqtrue_err_list) > 0)*"is not ({})".format(', '.join(access_desc_reqtrue_err_list)),
            "and" if (len(access_desc_reqtrue_err_list) > 0 and len(access_desc_reqfalse_err_list) > 0) else '',
            (len(access_desc_reqfalse_err_list) > 0)*"is ({})".format(', '.join(access_desc_reqfalse_err_list)),
        ])
        errmsg = ' '.join(errmsg.split())
        errmsg = errmsg.replace(' ,', ',')
        raise ScriptArgumentError("argument {}: {}".format(argstr, errmsg))
    return abspath_fn(path) if abspath_fn is not None else path
ARGTYPE_PATH = functools.partial(functools.partial, argtype_path_handler)

def argtype_num_encode(num):
    num_str = str(num)
    if num_str.startswith('-') or num_str.startswith('+'):
        num_str = "'({})' ".format(num_str)
    return num_str
def argtype_num_decode(num_str):
    num_str = ''.join(num_str.split())
    return num_str.strip("'").strip('"').lstrip('(').rstrip(')')
def argtype_num_handler(num_str, argstr,
                        numeric_type=float,
                        allow_pos=True, allow_neg=True, allow_zero=True,
                        allow_inf=False, allow_nan=False,
                        allowed_min=None, allowed_max=None,
                        allowed_min_incl=True, allowed_max_incl=True):
    num_str = argtype_num_decode(num_str)
    if (   (allowed_min is not None and ((allowed_min < 0 and not allow_neg) or (allowed_min == 0 and not allow_zero) or (allowed_min > 0 and not allow_pos)))
        or (allowed_max is not None and ((allowed_max < 0 and not allow_neg) or (allowed_max == 0 and not allow_zero) or (allowed_max > 0 and not allow_pos)))):
        raise DeveloperError("Allowed min/max value does not align with allowed pos/neg/zero settings")
    dtype_name_dict = {
        int: 'integer',
        float: 'decimal'
    }
    lt_min_op = operator.lt if allowed_min_incl else operator.le
    gt_max_op = operator.gt if allowed_max_incl else operator.ge
    errmsg = None
    try:
        number_float = float(num_str)
    except ValueError:
        errmsg = "input could not be parsed as a valid (floating point) number"
    if errmsg is None:
        if number_float != number_float:  # assume number is NaN
            number_true = number_float
            if not allow_nan:
                errmsg = "NaN is not allowed"
        else:
            if number_float in (float('inf'), float('-inf')):
                number_true = number_float
                if not allow_inf:
                    errmsg = "+/-infinity is not allowed"
            else:
                try:
                    number_true = numeric_type(number_float)
                    if number_true != number_float:
                        errmsg = "number must be of {}type {}".format(
                            dtype_name_dict[numeric_type]+' ' if numeric_type in dtype_name_dict else '', numeric_type)
                except ValueError:
                    errmsg = "input could not be parsed as a designated {} number".format(numeric_type)
            if errmsg is None:
                if (   (not allow_pos and number_true > 0) or (not allow_neg and number_true < 0) or (not allow_zero and number_true == 0)
                    or (allowed_min is not None and lt_min_op(number_true, allowed_min)) or (allowed_max is not None and gt_max_op(number_true, allowed_max))):
                    input_cond = ' '.join([
                        "input must be a",
                        'positive'*allow_pos, 'or'*(allow_pos&allow_neg), 'negative'*allow_neg, 'non-zero'*(not allow_zero),
                        '{} number'.format(dtype_name_dict[numeric_type]) if numeric_type in dtype_name_dict else 'number of type {}'.format(numeric_type),
                        (allow_inf|allow_nan)*' ({} allowed)'.format(' '.join([
                            '{}infinity'.format('/'.join(['+'*allow_pos, '-'*allow_neg]).strip('/'))*allow_inf, 'and'*(allow_inf&allow_nan), 'NaN'*allow_nan]))
                    ])
                    if allowed_min is not None or allowed_max is not None:
                        if allowed_min is not None and allowed_max is not None:
                            input_cond_range = "in the range {}{}, {}{}".format(
                                '[' if allowed_min_incl else '(', allowed_min, allowed_max, ']' if allowed_max_incl else ']')
                        else:
                            if allowed_min is not None:
                                cond_comp = 'greater'
                                cond_value = allowed_min
                                cond_bound_incl = allowed_min_incl
                            elif allowed_max is not None:
                                cond_comp = 'less'
                                cond_value = allowed_max
                                cond_bound_incl = allowed_max_incl
                            input_cond_range = '{} than {} ({})'.format(
                                cond_comp, cond_value, 'inclusive' if cond_bound_incl else 'exclusive')
                        input_cond = ' '.join([input_cond, input_cond_range])
                    input_cond = ' '.join(input_cond.split())
                    input_cond = input_cond.replace(' ,', ',')
                    errmsg = input_cond
    if errmsg is not None:
        raise ScriptArgumentError("argument {}: {}".format(argstr, errmsg))
    else:
        return number_true
ARGTYPE_NUM = functools.partial(functools.partial, argtype_num_handler)
ARGNUM_POS_INF = argtype_num_encode(float('inf'))
ARGNUM_NEG_INF = argtype_num_encode(float('-inf'))
ARGNUM_NAN = argtype_num_encode(float('nan'))

def argtype_bool_plus(value, parse_fn=None):
    if parse_fn is not None:
        return parse_fn(value)
    return value
ARGTYPE_BOOL_PLUS = functools.partial(functools.partial, argtype_bool_plus)


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
                cmd_subs = ','.join(['p{}="{}"'.format(i+1, a) for i, a in enumerate(jobscript_subs) if a is not None])
                cmd = r'{} -v {}'.format(cmd, cmd_subs)
            cmd = r'{} -N {}'.format(cmd, jobname)

        elif scheduler == SCHED_SLURM:
            cmd = 'sbatch'
            jobscript_optkey = '#SBATCH'
            if jobscript_subs is not None:
                cmd_subs = ','.join(['p{}="{}"'.format(i+1, a) for i, a in enumerate(jobscript_subs) if a is not None])
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
