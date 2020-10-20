
import re


RE_CATALOGID = re.compile("[0-9A-F]{16}")
RE_ORDERNUM = re.compile("\d{12}_\d{2}")

RE_TILENUM = re.compile("R\d+C\d+-")
RE_PARTNUM = re.compile("P\d{3}")


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


class Regex(object):
    recmp = None
    restr = None
    def __init__(self, string_or_match=None):
        re_string = None
        re_match = None
        if string_or_match is None:
            pass
        elif type(string_or_match) is str:
            re_string = string_or_match
        else:
            re_match = string_or_match
            try:
                re_match.groups
            except AttributeError:
                raise InvalidArgumentError("Argument to Regex class constructor must be "
                                           "None, the string to parse, or a re.Match object")
        self.string = re_string
        self._populate_match(re_match)
    def _populate_match(self, re_match):
        self.re_match = re_match
        self.matched = (self.re_match is not None)
        if self.matched:
            self.match_str = self.re_match.group(0)
            self.groupdict = self.re_match.groupdict()
        else:
            self.match_str = None
            self.groupdict = None
    def _re_function(self, function, string=None, in_place=True, **kwargs):
        if in_place and string is not None:
            self.string = string
        return_obj = self if in_place else type(self)(string_or_match=string)
        return_obj._populate_match(function(return_obj.recmp, return_obj.string, **kwargs))
        return return_obj
    def search(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.search, string=string, in_place=(not return_new), **kwargs)
    def match(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.match, string=string, in_place=(not return_new), **kwargs)
    def fullmatch(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.fullmatch, string=string, in_place=(not return_new), **kwargs)
    def findall(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.findall, string=string, in_place=(not return_new), **kwargs)
    def finditer(self, string=None, return_new=False, **kwargs):
        return self._re_function(re.finditer, string=string, in_place=(not return_new), **kwargs)


class Pairname(Regex):
    regrp_sensor = 'sensor'
    regrp_timestamp = 'timestamp'
    regrp_catid1 = 'catid1'
    regrp_catid2 = 'catid2'
    restr = "(?P<%s>[A-Z0-9]{{4}})_(?P<%s>\d{{8}})_(?P<%s>{0})_(?P<%s>{0})".format(RE_CATALOGID.pattern)
    restr = restr % (regrp_sensor, regrp_timestamp, regrp_catid1, regrp_catid2)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None):
        super(Pairname, self).__init__(string_or_match)
    def _populate_match(self, re_match):
        super(Pairname, self)._populate_match(re_match)
        if not self.matched:
            return
        self.pairname = self.match_str
        self.sensor = self.groupdict[Pairname.regrp_sensor]
        self.timestamp = self.groupdict[Pairname.regrp_timestamp]
        self.catid1 = self.groupdict[Pairname.regrp_catid1]
        self.catid2 = self.groupdict[Pairname.regrp_catid2]
        self.catids = [self.catid1, self.catid2]


class DemGroupID(Regex):
    regrp_pairname = 'pairname'
    regrp_res = 'res'
    regrp_lsf = 'lsf'
    regrp_version = 'version'
    restr_with_lsf = ''.join([
        "(?P<%s>{})_".format(Pairname.recmp.pattern) % regrp_pairname,
        "(?P<%s>\d+c?m)" % regrp_res,
        "(_(?P<%s>lsf))?" % regrp_lsf,
        "_(?P<%s>v\d{6})" % regrp_version,
    ])
    restr_without_lsf = ''.join([
        "(?P<%s>{})_".format(Pairname.recmp.pattern) % regrp_pairname,
        "(?P<%s>\d+c?m)" % regrp_res,
        "_(?P<%s>v\d{6})" % regrp_version,
    ])
    recmp = re.compile(restr_with_lsf)
    def __init__(self, string_or_match=None, can_have_lsf=False):
        self.can_have_lsf = can_have_lsf
        super(DemGroupID, self).__init__(string_or_match)
    def _populate_match(self, re_match):
        super(DemGroupID, self)._populate_match(re_match)
        if not self.matched:
            return
        self.stripdemid = self.match_str
        self.pairname = self.groupdict[DemGroupID.regrp_pairname]
        self.Pairname = Pairname(self.re_match)
        self.res = self.groupdict[DemGroupID.regrp_res]
        if self.can_have_lsf:
            self.lsf = self.groupdict[DemGroupID.regrp_lsf]
        self.version = self.groupdict[DemGroupID.regrp_version]

class StripDemGroupID(DemGroupID):
    recmp = re.compile(DemGroupID.restr_with_lsf)
    def __init__(self, string_or_match=None):
        super(StripDemGroupID, self).__init__(string_or_match, can_have_lsf=True)

class SceneDemGroupID(DemGroupID):
    recmp = re.compile(DemGroupID.restr_without_lsf)
    def __init__(self, string_or_match=None):
        super(SceneDemGroupID, self).__init__(string_or_match, can_have_lsf=False)


class StripDemID(Regex):
    regrp_pairname = 'pairname'
    regrp_res = 'res'
    regrp_lsf = 'lsf'
    regrp_segnum = 'segnum'
    restr = ''.join([
        "(?P<%s>{})_".format(Pairname.recmp.pattern) % regrp_pairname,
        "(?P<%s>\d+c?m)" % regrp_res,
        "(_(?P<%s>lsf))?" % regrp_lsf,
        "_seg(?P<%s>\d+)" % regrp_segnum,
    ])
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None):
        super(StripDemID, self).__init__(string_or_match)
    def _populate_match(self, re_match):
        super(StripDemID, self)._populate_match(re_match)
        if not self.matched:
            return
        self.stripdemid = self.match_str
        self.pairname = self.groupdict[StripDemID.regrp_pairname]
        self.Pairname = Pairname(self.re_match)
        self.res = self.groupdict[StripDemID.regrp_res]
        self.lsf = self.groupdict[StripDemID.regrp_lsf]
        self.segnum = self.groupdict[StripDemID.regrp_segnum]

class StripDemFile(Regex):
    regrp_stripdemid = 'stripdemid'
    regrp_suffix = 'suffix'
    restr = ''.join([
        "(?P<%s>{})_".format(StripDemID.recmp.pattern) % regrp_stripdemid,
        "(?P<%s>.+)" % regrp_suffix,
    ])
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None):
        super(StripDemFile, self).__init__(string_or_match)
    def _populate_match(self, re_match):
        super(StripDemFile, self)._populate_match(re_match)
        if not self.matched:
            return
        self.stripdemfile = self.match_str
        self.stripdemid = self.groupdict[StripDemFile.regrp_stripdemid]
        self.StripDemID = StripDemID(self.re_match)
        self.suffix = self.groupdict[StripDemFile.regrp_suffix]


class SceneDemID(Regex):
    regrp_pairname = 'pairname'
    regrp_tile1 = 'tile1'
    regrp_tile2 = 'tile2'
    regrp_order1 = 'order1'
    regrp_order2 = 'order2'
    regrp_part1 = 'part1'
    regrp_part2 = 'part2'
    regrp_res = 'res'
    regrp_subtile = 'subtile'
    restr = ''.join([
        "(?P<%s>{0})_" % regrp_pairname,
        "(?P<%s>{1})?(?P<%s>{2})_(?P<%s>{3})_" % (regrp_tile1, regrp_order1, regrp_part1),
        "(?P<%s>{1})?(?P<%s>{2})_(?P<%s>{3})_" % (regrp_tile2, regrp_order2, regrp_part2),
        "(?P<%s>\d{{1}})(?P<%s>-\d{{2}})?" % (regrp_res, regrp_subtile),
    ])
    restr = restr.format(Pairname.recmp.pattern, RE_TILENUM.pattern, RE_ORDERNUM.pattern, RE_PARTNUM.pattern)
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None):
        super(SceneDemID, self).__init__(string_or_match)
    def _populate_match(self, re_match):
        super(SceneDemID, self)._populate_match(re_match)
        if not self.matched:
            return
        self.scenedemid = self.match_str
        self.pairname = self.groupdict[SceneDemID.regrp_pairname]
        self.Pairname = Pairname(self.re_match)
        self.tile1 = self.groupdict[SceneDemID.regrp_tile1]
        self.order1 = self.groupdict[SceneDemID.regrp_order1]
        self.part1 = self.groupdict[SceneDemID.regrp_part1]
        self.tile2 = self.groupdict[SceneDemID.regrp_tile2]
        self.order2 = self.groupdict[SceneDemID.regrp_order2]
        self.part2 = self.groupdict[SceneDemID.regrp_part2]
        self.res = self.groupdict[SceneDemID.regrp_res]
        self.subtile = self.groupdict[SceneDemID.regrp_subtile]
        self.orders = [self.order1, self.order2]

class SceneDemFile(Regex):
    regrp_scenedemid = 'scenedemid'
    regrp_suffix = 'suffix'
    restr = ''.join([
        "(?P<%s>{})_".format(SceneDemID.recmp.pattern) % regrp_scenedemid,
        "(?P<%s>.+)" % regrp_suffix,
    ])
    recmp = re.compile(restr)
    def __init__(self, string_or_match=None):
        super(SceneDemFile, self).__init__(string_or_match)
    def _populate_match(self, re_match):
        super(SceneDemFile, self)._populate_match(re_match)
        if not self.matched:
            return
        self.scenedemfile = self.match_str
        self.sscenedemid = self.groupdict[SceneDemFile.regrp_scenedemid]
        self.SceneDemID = SceneDemID(self.re_match)
        self.suffix = self.groupdict[SceneDemFile.regrp_suffix]
