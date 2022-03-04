
import math
import operator
import re
from collections import OrderedDict


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)
class RegexConstructionFailure(Exception):
    def __init__(self, regrp, string, restr):
        msg = "`{}` argument value '{}' failed regex fullmatch '{}'".format(
            regrp, string, restr
        )
        super(Exception, self).__init__(msg)


RE_PATTERN_TYPE = type(re.compile(""))
RE_MATCH_TYPE = type(re.match("", ""))

def re_fullmatch_using_match(pattern, string, **kwargs):
    pattern_type = type(pattern)
    if pattern_type is str:
        pattern_str = pattern
    elif pattern_type is RE_PATTERN_TYPE:
        pattern_str = pattern.pattern
    else:
        raise TypeError("first argument must be string or compiled pattern")
    if not pattern_str.endswith('\Z'):
        if pattern is pattern_str:
            pattern += '\Z'
        else:
            pattern = re.compile(pattern_str+'\Z')
    return re.match(pattern, string, **kwargs)

try:
    RE_FULLMATCH_FN = re.fullmatch
except AttributeError:
    RE_FULLMATCH_FN = re_fullmatch_using_match


class VersionString(object):
    def __init__(self, version, nelements=None, allow_alpha=False):
        self.is_numeric = (not allow_alpha)
        if type(version) in (list, tuple):
            nums_input = version
            nums_internal = []
            for n in nums_input:
                if type(n) is float:
                    n_float = n
                    n_int = int(n)
                    if n_float != n_int:
                        raise InvalidArgumentError(
                            "Non-integer element '{}' in version number list: {}".format(n, nums_input)
                        )
                    n = n_int
                if self.is_numeric:
                    try:
                        n = int(n)
                    except ValueError:
                        raise InvalidArgumentError(
                            "Non-numeric element '{}' in version number list: {}".format(n, nums_input)
                        )
                else:
                    n = str(n).strip()
                    if '.' in n:
                        raise InvalidArgumentError(
                            "Invalid element '{}' in version number list: {}".format(n, nums_input)
                        )
                nums_internal.append(n)
            self.nums = nums_internal
            self.string = '.'.join([str(n) for n in self.nums])
        else:
            version_string = version
            self.string = str(version_string).strip()
            if not self.is_numeric:
                self.nums = [n.strip() for n in self.string.split('.')]
            else:
                nums_internal = []
                for n in self.string.split('.'):
                    try:
                        n = int(n)
                    except ValueError:
                        raise InvalidArgumentError(
                            "Non-numeric element '{}' in version string: '{}'".format(n, version_string)
                        )
                    nums_internal.append(n)
                self.nums = nums_internal
        if nelements is not None:
            numel_diff = nelements - len(self.nums)
            if numel_diff < 0:
                raise InvalidArgumentError(
                    "Provided version string '{}' has more elements ({}) than `nelements` ({})".format(
                        self.string, len(self.nums), nelements
                    )
                )
            elif numel_diff > 0:
                self.nums.extend([0 if self.is_numeric else '0'] * numel_diff)
                self.string = '.'.join([str(n) for n in self.nums])
    def __str__(self):
        return self.string
    def __repr__(self):
        return self.string
    def _get_comparable_nums(self, other):
        if self.is_numeric and other.is_numeric:
            zero_num = 0
            this_nums = list(self.nums)
            other_nums = list(other.nums)
        else:
            zero_num = '0'
            this_nums = [str(n) for n in self.nums] if self.is_numeric else list(self.nums)
            other_nums = [str(n) for n in other.nums] if other.is_numeric else list(other.nums)
        numel_diff = len(other_nums) - len(this_nums)
        if numel_diff > 0:
            this_nums.extend([zero_num] * numel_diff)
        elif numel_diff < 0:
            other_nums.extend([zero_num] * (-numel_diff))
        if type(zero_num) is str:
            for i in range(len(this_nums)):
                ellen_diff = len(other_nums[i]) - len(this_nums[i])
                if ellen_diff > 0:
                    this_nums[i] = zero_num*ellen_diff + this_nums[i]
                elif ellen_diff < 0:
                    other_nums[i] = zero_num*(-ellen_diff) + other_nums[i]
        return this_nums, other_nums
    def _compare_absolute(self, other, inequality=False):
        this_nums, other_nums = self._get_comparable_nums(other)
        for i in range(len(this_nums)):
            if this_nums[i] != other_nums[i]:
                return inequality
        return (not inequality)
    def _compare_relative(self, other, op, allow_equal=False):
        this_nums, other_nums = self._get_comparable_nums(other)
        for i in range(len(this_nums)):
            if this_nums[i] != other_nums[i]:
                return op(this_nums[i], other_nums[i])
        return allow_equal
    def __eq__(self, other):
        return self._compare_absolute(other, inequality=False)
    def __ne__(self, other):
        return self._compare_absolute(other, inequality=True)
    def __gt__(self, other):
        return self._compare_relative(other, operator.gt, allow_equal=False)
    def __ge__(self, other):
        return self._compare_relative(other, operator.gt, allow_equal=True)
    def __lt__(self, other):
        return self._compare_relative(other, operator.lt, allow_equal=False)
    def __le__(self, other):
        return self._compare_relative(other, operator.le, allow_equal=True)


class Regex(object):
    recmp = None
    restr = None
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        string_or_match_type = type(string_or_match)
        if string_or_match is None:
            self.string = None
            self._populate_match(None)
        elif string_or_match_type is str:
            self.string = string_or_match
            self._re_function(re_function, in_place=True, **re_function_kwargs)
        elif string_or_match_type is RE_MATCH_TYPE:
            re_match = string_or_match
            self.string = re_match.string
            self._populate_match(re_match)
        else:
            raise InvalidArgumentError(
                "First argument to Regex class constructor can be None, "
                "a string to parse, or a `re` match object of type {} "
                "but was '{}' of type {}".format(RE_MATCH_TYPE, string_or_match, string_or_match_type))
    def _populate_match(self, re_match):
        if re_match is not None:
            self.matched = True
            self.re_match = re_match
            self.match_str = re_match.group(0)
            self.groupdict = re_match.groupdict()
        else:
            self._reset_match_attributes()
    def _reset_match_attributes(self):
        self.matched = False
        self.re_match = None
        self.match_str = None
        self.groupdict = None
    def _re_function(self, function, string=None, in_place=True, **kwargs):
        if string is not None:
            use_string = string
            if in_place:
                self.string = string
        elif self.string is not None:
            use_string = self.string
        else:
            raise InvalidArgumentError(
                "Argument `string` must be provided when calling `re` function "
                "on Regex object that has not been provided a search string prior"
            )
        match_results = function(self.restr if kwargs else self.recmp, use_string, **kwargs)
        if function is re.findall:
            return match_results
        elif function is re.finditer:
            return self._yield_match_results(match_results, in_place)
        else:
            re_match = match_results
            if in_place:
                result_obj = self
                result_obj._populate_match(re_match)
            else:
                result_obj = type(self)(string_or_match=re_match)
            return result_obj
    def _yield_match_results(self, match_results, in_place):
        for re_match in match_results:
            if in_place:
                result_obj = self
                result_obj._populate_match(re_match)
            else:
                result_obj = type(self)(string_or_match=re_match)
            yield result_obj
    def clear(self):
        self.string = None
        self._reset_match_attributes()
    def search(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.search, string=string, in_place=(not return_new), **kwargs)
    def match(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.match, string=string, in_place=(not return_new), **kwargs)
    def fullmatch(self, string=None, return_new=False, **kwargs):
        return self._re_function(RE_FULLMATCH_FN, string=string, in_place=(not return_new), **kwargs)
    def findall(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.findall, string=string, in_place=(not return_new), **kwargs)
    def finditer(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.finditer, string=string, in_place=(not return_new), **kwargs)


class SetsmVersionKey(Regex):
    regrp_major = 'major'
    regrp_minor = 'minor'
    regrp_patch = 'patch'
    restr = r"v(?P<%s>\d{2})(?P<%s>\d{2})(?P<%s>\d{2})" % (regrp_major, regrp_minor, regrp_patch)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(SetsmVersionKey, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(SetsmVersionKey, self)._populate_match(re_match)
        if self.matched:
            self.SetsmVersionKey = self
            self.verkey          = self.match_str
            self.major           = int(self.groupdict[SetsmVersionKey.regrp_major])
            self.minor           = int(self.groupdict[SetsmVersionKey.regrp_minor])
            self.patch           = int(self.groupdict[SetsmVersionKey.regrp_patch])
            self.version         = VersionString('.'.join([str(num) for num in [self.major, self.minor, self.patch]]))
    def _reset_match_attributes(self):
        super(SetsmVersionKey, self)._reset_match_attributes()
        self.SetsmVersionKey = None
        self.verkey          = None
        self.major           = None
        self.minor           = None
        self.patch           = None
        self.version         = None


def SetsmVersion(version):
    accepted_version_types = (str, float, int, SetsmVersionKey, VersionString)
    version_key = None
    version_string = None
    version_type = type(version)
    if version_type in (str, float, int):
        if version_type is str and version.startswith('v'):
            verkey_str = version
            version_key = SetsmVersionKey(verkey_str, RE_FULLMATCH_FN)
            if not version_key.matched:
                raise InvalidArgumentError(
                    "Failed to parse `version` argument ('{}') with {} regex ('{}')".format(
                        version, SetsmVersionKey, SetsmVersionKey.restr
                    )
                )
        else:
            version_string = VersionString(version, nelements=3)
    elif isinstance(version, SetsmVersionKey):
        version_key = version
    elif isinstance(version, VersionString):
        version_string = VersionString(version.string, nelements=3)
    else:
        raise InvalidArgumentError("`version` type must be one of {}, but was {}".format(
            accepted_version_types, version
        ))
    if version_key is None:
        assert version_string is not None
        if len(version_string.nums) != 3:
            raise InvalidArgumentError(
                "`version` {} argument must have three elements (major, minor, patch), "
                "but has {} elements: '{}'".format(
                    VersionString, len(version_string.nums), version_string.string
                )
            )
        if not all([0 <= n <= 99 for n in version_string.nums]):
            raise InvalidArgumentError(
                "`version` {} argument element values (major, minor, patch) "
                "must be in the range [0, 99]: '{}'".format(
                    VersionString, version_string.string
                )
            )
        verkey_str = 'v{}'.format(''.join(['{:0>2}'.format(n) for n in version_string.nums]))
        version_key = SetsmVersionKey(verkey_str, RE_FULLMATCH_FN)
        if not version_key.matched:
            raise InvalidArgumentError(
                "Failed to parse {} '{}' into {} object".format(
                    VersionString, version, SetsmVersionKey
                )
            )
    return version_key


class DemResName(Regex):
    regrp_value = 'value'
    regrp_unit = 'unit'
    unit_m = 'm'
    unit_cm = 'cm'
    restr = r"(?P<%s>\d+)(?P<%s>%s|%s)" % (regrp_value, regrp_unit, unit_m, unit_cm)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(DemResName, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(DemResName, self)._populate_match(re_match)
        if self.matched:
            self.DemResName = self
            self.res_name   = self.match_str
            self.value      = self.groupdict[DemResName.regrp_value]
            self.unit       = self.groupdict[DemResName.regrp_unit]
    def _reset_match_attributes(self):
        super(DemResName, self)._reset_match_attributes()
        self.DemResName = None
        self.res_name   = None
        self.value      = None
        self.unit       = None


class DemRes(object):

    accepted_res_types = (str, float, int, DemResName)
    accepted_res_name_types = (str, DemResName)
    standard_res_meters = (0.5, 1, 2, 8)

    def __init__(self, res, allow_nonstandard_res=False):
        res_name = None
        res_meters = None
        res_whole_meters = None
        res_type = type(res)
        if res_type is str:
            try:
                res_str_as_float = float(res)
                try:
                    res_str_as_int = int(res)
                    res_whole_meters = res_str_as_int
                except ValueError:
                    res_meters = res_str_as_float
            except ValueError:
                res_name = res
        elif res_type is float:
            res_meters = res
        elif res_type is int:
            res_whole_meters = res
        elif isinstance(res, DemResName):
            res_name = res
        else:
            raise InvalidArgumentError("`res` type must be one of {}, but was {}".format(
                DemRes.accepted_res_types, res_type
            ))
        self.name, self.meters, self.whole_meters = self.get_res_forms(
            res_name, res_meters, res_whole_meters, allow_nonstandard_res
        )

    @staticmethod
    def get_res_forms(res_name=None, res_meters=None, res_whole_meters=None,
                      allow_nonstandard_res=False):
        if not any([arg is not None for arg in [res_name, res_meters, res_whole_meters]]):
            raise InvalidArgumentError("At least one resolution argument must be provided")

        if res_meters is not None:
            res_meters = float(res_meters)
        converted_res_meters = None

        if res_name is not None:
            res_name_type = type(res_name)
            if res_name_type is str:
                dem_res_name = DemResName(res_name, RE_FULLMATCH_FN)
                if not dem_res_name.matched:
                    raise InvalidArgumentError(
                        "Failed to parse `res_name` argument ('{}') with {} regex ('{}')".format(
                        res_name, DemResName, DemResName.restr
                    ))
            elif isinstance(res_name, DemResName):
                dem_res_name = res_name
            else:
                raise InvalidArgumentError("`res_name` type must be one of {}, but was {}".format(
                    DemRes.accepted_res_name_types, res_name_type
                ))
            res_name = dem_res_name.string
            if dem_res_name.unit == DemResName.unit_m:
                converted_res_meters = float(dem_res_name.value)
            elif dem_res_name.unit == DemResName.unit_cm:
                converted_res_meters = float(dem_res_name.value) / 100
            if res_meters is None:
                res_meters = converted_res_meters
            elif res_meters != converted_res_meters:
                raise InvalidArgumentError(
                    "Mismatch between converted res meters ({}) from `res_name` argument ('{}') "
                    "and `res_meters` argument ({}) ".format(
                        converted_res_meters, res_name, res_meters,
                    )
                )

        if res_whole_meters is not None:
            res_whole_meters_as_int = int(res_whole_meters)
            if res_whole_meters_as_int != res_whole_meters:
                raise InvalidArgumentError(
                    "`res_whole_meters_as_int` argument must be convertible to an integer, "
                    "but was {}".format(res_whole_meters)
                )
            res_whole_meters = res_whole_meters_as_int
            converted_res_meters = 0.5 if res_whole_meters == 0 else float(res_whole_meters)
            if res_meters is None:
                res_meters = converted_res_meters
            elif res_meters != converted_res_meters:
                raise InvalidArgumentError(
                    "Mismatch between converted res meters ({}) from `res_whole_meters` argument ('{}') "
                    "and other resolution argument(s)".format(
                        converted_res_meters, res_whole_meters,
                    )
                )

        if res_meters <= 0:
            raise InvalidArgumentError("Resolution in meters must be > 0, but was {}".format(res_meters))
        if not allow_nonstandard_res and res_meters not in DemRes.standard_res_meters:
            raise InvalidArgumentError("Resolution in meters must be one of standard set {}, "
                                       "but was {}".format(DemRes.standard_res_meters, res_meters))

        if res_whole_meters is None:
            res_whole_meters = int(math.floor(res_meters))
        if res_name is None:
            if res_meters == res_whole_meters:
                res_name = '{}m'.format(int(res_meters))
            else:
                res_name = '{}cm'.format(int(math.floor(res_meters * 100)))

        return res_name, res_meters, res_whole_meters


RECMP_CATALOGID = re.compile("[0-9A-F]{16}")
RECMP_ORDERNUM = re.compile("\d{12}_\d{2}")
RECMP_TILENUM = re.compile("R\d+C\d+")
RECMP_PARTNUM = re.compile("P\d{3}")


class Pairname(Regex):
    regrp_sensor, recmp_sensor = 'sensor', re.compile("[A-Z][A-Z0-9]{2}[0-9]")
    regrp_timestamp, recmp_timestamp = 'timestamp', re.compile("\d{8}")
    regrp_catid1, recmp_catid1 = 'catid1', RECMP_CATALOGID
    regrp_catid2, recmp_catid2 = 'catid2', RECMP_CATALOGID
    @staticmethod
    def construct(sensor=None, timestamp=None, catid1=None, catid2=None,
                  validate=True, return_regex=False):
        skip_inspection = False
        if all([sensor, timestamp, catid1, catid2]):
            if not validate:
                skip_inspection = True
        elif not return_regex:
            raise InvalidArgumentError(
                "All regex group values must be provided when `return_regex=False`"
            )
        regrp_setting_dict = OrderedDict([
            (Pairname.regrp_sensor,      [sensor,    Pairname.recmp_sensor]),
            (Pairname.regrp_timestamp,   [timestamp, Pairname.recmp_timestamp]),
            (Pairname.regrp_catid1,      [catid1,    Pairname.recmp_catid1]),
            (Pairname.regrp_catid2,      [catid2,    Pairname.recmp_catid2]),
        ])
        if not skip_inspection:
            try:
                for regrp, setting in regrp_setting_dict.items():
                    arg_string, default_recmp = setting
                    if arg_string is None:
                        setting[0] = default_recmp.pattern
                    elif validate and not RE_FULLMATCH_FN(default_recmp, arg_string):
                        raise RegexConstructionFailure(regrp, arg_string, default_recmp.pattern)
            except RegexConstructionFailure:
                if not return_regex:
                    return None
                else:
                    raise
        if return_regex:
            full_restr = '_'.join([
                "(?P<{}>{})".format(regrp, setting[0])
                for regrp, setting in regrp_setting_dict.items()
            ])
            return full_restr
        else:
            full_string = '_'.join([setting[0] for setting in regrp_setting_dict.values()])
            return full_string
    @staticmethod
    def get_regex(sensor=None, timestamp=None, catid1=None, catid2=None, validate=True):
        return Pairname.construct(sensor, timestamp, catid1, catid2, validate, True)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN,
                 sensor=None,
                 timestamp=None,
                 catid1=None,
                 catid2=None,
                 **re_function_kwargs):
        if any([sensor, timestamp, catid1, catid2]):
            self.restr = self.get_regex(sensor, timestamp, catid1, catid2)
            self.recmp = re.compile(self.restr)
        super(Pairname, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(Pairname, self)._populate_match(re_match)
        if self.matched:
            self.Pairname   = self
            self.pairname   = self.match_str
            self.sensor     = self.groupdict[Pairname.regrp_sensor]
            self.timestamp  = self.groupdict[Pairname.regrp_timestamp]
            self.catid1     = self.groupdict[Pairname.regrp_catid1]
            self.catid2     = self.groupdict[Pairname.regrp_catid2]
            self.catids     = [self.catid1, self.catid2]
    def _reset_match_attributes(self):
        super(Pairname, self)._reset_match_attributes()
        self.Pairname   = None
        self.pairname   = None
        self.sensor     = None
        self.timestamp  = None
        self.catid1     = None
        self.catid2     = None
        self.catids     = None
Pairname.restr = Pairname.get_regex()
Pairname.recmp = re.compile(Pairname.restr)


class StripDemID(Regex):
    regrp_pairname = 'pairname'
    regrp_res = 'res'
    regrp_verkey = 'verkey'
    restr = r"(?P<%s>{0})_(?P<%s>{1})_(?P<%s>{2})" % (
        regrp_pairname, regrp_res, regrp_verkey
    )
    restr = restr.format(Pairname.recmp.pattern, DemResName.recmp.pattern, SetsmVersionKey.recmp.pattern)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(StripDemID, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(StripDemID, self)._populate_match(re_match)
        if self.matched:
            self.stripDemID = self.match_str
            self.pairname   = self.groupdict[StripDemID.regrp_pairname]
            self.Pairname   = Pairname(self.re_match)
            self.res        = self.groupdict[StripDemID.regrp_res]
            self.DemRes     = DemRes(self.res)
            self.verkey     = self.groupdict[StripDemFolder.regrp_verkey]
            self.Verkey     = SetsmVersionKey(self.re_match)
            self.version    = self.Verkey.version
    def _reset_match_attributes(self):
        super(StripDemID, self)._reset_match_attributes()
        self.stripDemID = None
        self.pairname   = None
        self.Pairname   = None
        self.res        = None
        self.DemRes     = None
        self.version    = None
        self.Verkey     = None
        self.version    = None


class StripDemFolder(Regex):
    regrp_pairname = 'pairname'
    regrp_res = 'res'
    regrp_lsf = 'lsf'
    regrp_verkey = 'verkey'
    restr = r"(?P<%s>{0})_(?P<%s>{1})(?:_(?P<%s>lsf))?_(?P<%s>{2})" % (
        regrp_pairname, regrp_res, regrp_lsf, regrp_verkey
    )
    restr = restr.format(Pairname.recmp.pattern, DemResName.recmp.pattern, SetsmVersionKey.recmp.pattern)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(StripDemFolder, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(StripDemFolder, self)._populate_match(re_match)
        if self.matched:
            self.StripDemFolder = self
            self.stripDemFolder = self.match_str
            self.StripDemID     = StripDemID.construct(
                pairname=self.groupdict[StripDemFolder.regrp_pairname],
                res     =self.groupdict[StripDemFolder.regrp_res],
                version =self.groupdict[StripDemFolder.regrp_verkey]
            )
            self.lsf            = self.groupdict[StripDemFolder.regrp_lsf]

    def _reset_match_attributes(self):
        super(StripDemFolder, self)._reset_match_attributes()
        self.StripDemFolder = None
        self.stripDemFolder = None
        self.StripDemID     = None
        self.lsf            = None


class SceneDemOverlapID(Regex):
    regrp_pairname = 'pairname'
    regrp_tile1 = 'tile1'
    regrp_tile2 = 'tile2'
    regrp_orderid1 = 'orderid1'
    regrp_orderid2 = 'orderid2'
    regrp_ordernum1 = 'ordernum1'
    regrp_ordernum2 = 'ordernum2'
    regrp_partnum1 = 'partnum1'
    regrp_partnum2 = 'partnum2'
    restr = ''.join([
        r"(?P<%s>{0})_" % regrp_pairname,
        r"(?:(?P<%s>{1})-)?(?P<%s>(?P<%s>{2})_(?P<%s>{3}))_" % (regrp_tile1, regrp_orderid1, regrp_ordernum1, regrp_partnum1),
        r"(?:(?P<%s>{1})-)?(?P<%s>(?P<%s>{2})_(?P<%s>{3}))"  % (regrp_tile2, regrp_orderid2, regrp_ordernum2, regrp_partnum2),
    ])
    restr = restr.format(Pairname.recmp.pattern, RECMP_TILENUM.pattern, RECMP_ORDERNUM.pattern, RECMP_PARTNUM.pattern)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(SceneDemOverlapID, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(SceneDemOverlapID, self)._populate_match(re_match)
        if self.matched:
            self.SceneDemOverlapID = self
            self.sceneDemOverlapID = self.match_str
            self.pairname          = self.groupdict[SceneDemOverlapID.regrp_pairname]
            self.Pairname          = Pairname(self.re_match)
            self.tile1             = self.groupdict[SceneDemOverlapID.regrp_tile1]
            self.orderid1          = self.groupdict[SceneDemOverlapID.regrp_orderid1]
            self.ordernum1         = self.groupdict[SceneDemOverlapID.regrp_ordernum1]
            self.partnum1          = self.groupdict[SceneDemOverlapID.regrp_partnum1]
            self.tile2             = self.groupdict[SceneDemOverlapID.regrp_tile2]
            self.orderid2          = self.groupdict[SceneDemOverlapID.regrp_orderid2]
            self.ordernum2         = self.groupdict[SceneDemOverlapID.regrp_ordernum2]
            self.partnum2          = self.groupdict[SceneDemOverlapID.regrp_partnum2]
            self.orderids          = [self.orderid1,  self.orderid2]
            self.ordernums         = [self.ordernum1, self.ordernum2]
    def _reset_match_attributes(self):
        super(SceneDemOverlapID, self)._reset_match_attributes()
        self.SceneDemOverlapID = None
        self.sceneDemOverlapID = None
        self.pairname          = None
        self.Pairname          = None
        self.tile1             = None
        self.orderid1          = None
        self.ordernum1         = None
        self.partnum1          = None
        self.tile2             = None
        self.orderid2          = None
        self.ordernum2         = None
        self.partnum2          = None
        self.orderids          = None
        self.ordernums         = None


class SceneDemID(Regex):
    regrp_sceneDemOverlapID = 'sceneDemOverlapID'
    regrp_res = 'res'
    regrp_subtile = 'subtile'
    restr = ''.join([
        r"(?P<%s>{0})_" % regrp_sceneDemOverlapID,
        r"(?P<%s>\d{{1}})(?P<%s>-\d{{2}})?" % (regrp_res, regrp_subtile)
    ])
    restr = restr.format(SceneDemOverlapID.recmp.pattern)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(SceneDemID, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(SceneDemID, self)._populate_match(re_match)
        if self.matched:
            self.SceneDemID        = self
            self.sceneDemID        = self.match_str
            self.sceneDemOverlapID = self.groupdict[SceneDemID.regrp_sceneDemOverlapID]
            self.SceneDemOverlapID = SceneDemOverlapID(self.re_match)
            self.pairname          = self.groupdict[SceneDemOverlapID.regrp_pairname]
            self.Pairname          = Pairname(self.re_match)
            self.tile1             = self.groupdict[SceneDemOverlapID.regrp_tile1]
            self.orderid1          = self.groupdict[SceneDemOverlapID.regrp_orderid1]
            self.ordernum1         = self.groupdict[SceneDemOverlapID.regrp_ordernum1]
            self.partnum1          = self.groupdict[SceneDemOverlapID.regrp_partnum1]
            self.tile2             = self.groupdict[SceneDemOverlapID.regrp_tile2]
            self.orderid2          = self.groupdict[SceneDemOverlapID.regrp_orderid2]
            self.ordernum2         = self.groupdict[SceneDemOverlapID.regrp_ordernum2]
            self.partnum2          = self.groupdict[SceneDemOverlapID.regrp_partnum2]
            self.res               = self.groupdict[SceneDemID.regrp_res]
            self.DemRes            = DemRes(self.res)
            self.subtile           = self.groupdict[SceneDemID.regrp_subtile]
            self.orderids          = [self.orderid1,  self.orderid2]
            self.ordernums         = [self.ordernum1, self.ordernum2]
    def _reset_match_attributes(self):
        super(SceneDemID, self)._reset_match_attributes()
        self.SceneDemID        = None
        self.sceneDemID        = None
        self.SceneDemOverlapID = None
        self.pairname          = None
        self.Pairname          = None
        self.tile1             = None
        self.orderid1          = None
        self.ordernum1         = None
        self.partnum1          = None
        self.tile2             = None
        self.orderid2          = None
        self.ordernum2         = None
        self.partnum2          = None
        self.res               = None
        self.DemRes            = None
        self.subtile           = None
        self.orderids          = None
        self.ordernums         = None


class StripSegmentID(Regex):
    regrp_algorithm = 'algorithm'
    regrp_s2s_ver = 's2s_ver'
    regrp_pairname = 'pairname'
    regrp_res = 'res'
    regrp_lsf = 'lsf'
    regrp_segnum = 'segnum'
    restr = ''.join([
        r"(?P<%s>SETSM)_(?P<%s>s2s\d{{3}})_" % (regrp_algorithm, regrp_s2s_ver),
        r"(?P<%s>{0})_(?P<%s>{1})(?:_(?P<%s>lsf))?_seg(?P<%s>\d+)" % (
            regrp_pairname, regrp_res, regrp_lsf, regrp_segnum
        )
    ])
    restr = restr.format(Pairname.recmp.pattern, DemResName.recmp.pattern)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN, **re_function_kwargs):
        super(StripSegmentID, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(StripSegmentID, self)._populate_match(re_match)
        if self.matched:
            self.StripSegmentID = self
            self.stripSegmentID = self.match_str
            self.algorithm      = self.groupdict[StripSegmentID.regrp_algorithm]
            self.s2s_ver        = self.groupdict[StripSegmentID.regrp_s2s_ver]
            self.pairname       = self.groupdict[StripSegmentID.regrp_pairname]
            self.Pairname       = Pairname(self.re_match)
            self.res            = self.groupdict[StripSegmentID.regrp_res]
            self.DemRes         = DemRes(self.res)
            self.lsf            = self.groupdict[StripSegmentID.regrp_lsf]
            self.segnum         = self.groupdict[StripSegmentID.regrp_segnum]
    def _reset_match_attributes(self):
        super(StripSegmentID, self)._reset_match_attributes()
        self.StripSegmentID = None
        self.stripSegmentID = None
        self.algorithm      = None
        self.s2s_ver        = None
        self.pairname       = None
        self.Pairname       = None
        self.res            = None
        self.DemRes         = None
        self.lsf            = None
        self.segnum         = None


class StripSegmentFile(Regex):
    regrp_stripsegmentid = 'stripsegmentid'
    regrp_suffix, recmp_suffix = 'suffix', re.compile("_.+")
    @staticmethod
    def construct(stripsegmentid=None, suffix=None,
                  validate=True, return_regex=False):
        skip_inspection = False
        if all([stripsegmentid, suffix]):
            if not validate:
                skip_inspection = True
        elif not return_regex:
            raise InvalidArgumentError(
                "All regex group values must be provided when `return_regex=False`"
            )
        regrp_setting_dict = OrderedDict([
            (StripSegmentFile.regrp_stripsegmentid, [stripsegmentid,    StripSegmentID.recmp]),
            (StripSegmentFile.regrp_suffix,         [suffix, StripSegmentFile.recmp_suffix]),
        ])
        if not skip_inspection:
            try:
                for regrp, setting in regrp_setting_dict.items():
                    arg_string, default_recmp = setting
                    if arg_string is None:
                        setting[0] = default_recmp.pattern
                    elif validate and not RE_FULLMATCH_FN(default_recmp, arg_string):
                        raise RegexConstructionFailure(regrp, arg_string, default_recmp.pattern)
            except RegexConstructionFailure:
                if not return_regex:
                    return None
                else:
                    raise
        if return_regex:
            full_restr = ''.join([
                "(?P<{}>{})".format(regrp, setting[0])
                for regrp, setting in regrp_setting_dict.items()
            ])
            return full_restr
        else:
            full_string = ''.join([setting[0] for setting in regrp_setting_dict.values()])
            return full_string
    @staticmethod
    def get_regex(stripsegmentid=None, suffix=None, validate=True):
        return StripSegmentFile.construct(stripsegmentid, suffix, validate, True)
    def __init__(self, string_or_match=None, re_function=RE_FULLMATCH_FN,
                 stripsegmentid=None,
                 suffix=None,
                 **re_function_kwargs):
        if any([stripsegmentid, suffix]):
            self.restr = self.get_regex(stripsegmentid, suffix)
            self.recmp = re.compile(self.restr)
        super(StripSegmentFile, self).__init__(string_or_match, re_function, **re_function_kwargs)
    def _populate_match(self, re_match):
        super(StripSegmentFile, self)._populate_match(re_match)
        if self.matched:
            self.StripSegmentFile = self
            self.stripsegmentfile = self.match_str
            self.stripsegmentid   = self.groupdict[StripSegmentFile.regrp_stripsegmentid]
            self.StripSegmentID   = StripSegmentID(self.re_match)
            self.suffix           = self.groupdict[StripSegmentFile.regrp_suffix]
    def _reset_match_attributes(self):
        super(StripSegmentFile, self)._reset_match_attributes()
        self.StripSegmentFile = None
        self.stripsegmentfile = None
        self.stripsegmentid   = None
        self.StripSegmentID   = None
        self.suffix           = None
StripSegmentFile.restr = StripSegmentFile.get_regex()
StripSegmentFile.recmp = re.compile(StripSegmentFile.restr)
