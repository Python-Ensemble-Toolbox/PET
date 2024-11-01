"""Read Schlumberger Eclipse output files.
"""
from __future__ import division
import argparse
import codecs
import collections
import datetime
import fnmatch
import logging
import numpy
import numpy.ma
import os
import os.path as path
import struct
import sys


# import these from the common package, since we may need to use the same
# enumeration across file formats
from .ecl_common import Phase, Prop


# add a valid log in case we are not run through the main program which
# sets up one for us
log = logging.getLogger(__name__)  # pylint: disable=invalid-name
log.addHandler(logging.NullHandler())


# each keyword record in the file is associated with a name and an index
# (since a keyword may occur more than once)
_DataKey = collections.namedtuple('_DataKey', [
    'kwd',  # keyword present at this position
    'seq',  # sequence number (occurrence in file)
])


# for every (indexed) keyword, we want to know where to read the data
# in the file, as well as how much data there is
_DataDescr = collections.namedtuple('_DataDescr', [
    'pos',  # file position of this record
    'num',  # number of elements
    'typ',  # type of data
    'rcs',  # number of records containing data
])


def _read_descr(fileobj):
    """Read the the opened file and build a descriptor record of the
    following data array.

    :param fileobj: File object opened in binary reading mode.
    :returns:       Tuple of keyword, file position, number of items,
                    and their type.
    :rtype:         (str, int, int, str)
    """
    # read the length of this record, as a signed quad; if we encounter
    # end of file, then return None as the sentinel descriptor
    quad = bytes(fileobj.read(4))
    if len(quad) == 0:
        return None, None, None, None
    rec_len, = struct.unpack('>i', quad)

    # sub-records should not occur
    assert (rec_len == 16)

    # read the next 16 bytes as descriptor
    kwd, = struct.unpack('8s', fileobj.read(8))
    num, = struct.unpack('>i', fileobj.read(4))
    typ, = struct.unpack('4s', fileobj.read(4))

    # for Python 3 compatibility
    kwd = codecs.ascii_decode(kwd)[0]
    typ = codecs.ascii_decode(typ)[0]

    # seek forward to the end of the record
    fileobj.seek(abs(rec_len) - 16, 1)

    # read the trailing length as well
    trail, = struct.unpack('>i', fileobj.read(4))

    # verify that the trailing length is the same as the
    assert (trail == rec_len)

    # get the position of the next record in the file; this is where
    # we want to seek to, to start reading the data array
    pos = fileobj.tell()

    # return the descriptor of this array
    return (kwd, pos, num, typ)


# for each type of data, we need to know how much to read
_DataType = collections.namedtuple('_DataType', [
    'siz',  # number of bytes
    'fmt',  # struct formatting string
    'nch',  # not character type, which needs special processing
    'dsk',  # NumPy data type when stored on disk (big-endian)
    'mem',  # corresponding NumPy data type in memory
])


# Python 3 uses Unicode for characters
_STR_TYPE = 'S' if sys.hexversion < 0x03000000 else 'U'


# actual data types found in Eclipse files
_TYPES = {
    'INTE': _DataType(siz=4, fmt='i', nch=True,
                      dsk=numpy.dtype('>i4'), mem=numpy.int32),
    'REAL': _DataType(siz=4, fmt='f', nch=True,
                      dsk=numpy.dtype('>f4'), mem=numpy.float32),
    'LOGI': _DataType(siz=4, fmt='i', nch=True,
                      dsk=numpy.dtype('>i4'), mem=numpy.int32),
    'DOUB': _DataType(siz=8, fmt='d', nch=True,
                      dsk=numpy.dtype('>f8'), mem=numpy.float64),
    'CHAR': _DataType(siz=8, fmt='c', nch=False,
                      dsk=numpy.dtype('|S8'), mem=(
                          numpy.dtype('<' + _STR_TYPE + '8'))),
    'MESS': _DataType(siz=0, fmt='x', nch=True,
                      dsk=numpy.void, mem=numpy.void),
}


def _skip_rec(fileobj, num, typ):
    """Skip one or more records for a certain number of entries.

    :param fileobj: File object opened in binary read mode.
    :param num:     Number of items that should be skipped.
    :param typ:     Data type of the entries.
    :returns:       Number of data records that were skipped.
    """
    # total number of bytes to be skipped
    remaining = num * _TYPES[typ].siz

    # number of records these bytes were stored in
    rcs = 0

    # read until we have processed the entire data array
    while remaining > 0:
        # read the number of bytes in this record
        rec_len, = struct.unpack('>i', fileobj.read(4))

        # this part of the data array is now read
        remaining -= rec_len
        rcs += 1

        # actually skip those bytes, plus the trailing record length
        fileobj.seek(rec_len + 4, os.SEEK_CUR)

    # we should not end up in the middle of a record!
    assert (remaining == 0)

    # store the number of records so that we know exactly how many
    # records to read next time
    return rcs


def _build_cat(fileobj):
    """Build a catalog over where in the data file each record is.

    :param fileobj:  File object opened in binary read mode.
    :returns:        Tuple of dictionaries with the number of occurrences
    :rtype:          (dict [str] -> int, dict [_DataKey] -> _DataDescr)
    """
    cnt = {}  # number of types we have seen each keyword
    cat = {}  # where is the file each keyword is located
    while True:
        # read the record containing the data descriptor first
        kwd, pos, num, typ = _read_descr(fileobj)

        # stop once we have reached the end of the file (no more records)
        if kwd is None:
            break

        # skip over the data records for this key; we are only interested
        # in building the index at this stage
        rcs = _skip_rec(fileobj, num, typ)

        # update the number of occurrences of this keyword
        cnt[kwd] = cnt[kwd] + 1 if kwd in cnt else 1

        # create a descriptor for this record
        cat[_DataKey(kwd=kwd, seq=cnt[kwd] - 1)] = (
            _DataDescr(pos=pos, num=num, typ=typ, rcs=rcs))

    return (cnt, cat)


def _read_rec(fileobj, descr):
    """Read a set of records for a descriptor.

    :param fileobj:  File object opened in binary read mode.
    :param descr:    Descriptor for where the property is stored.
    :type descr:     _DataDescr
    """
    # seek to the appropriate place in the file
    fileobj.seek(descr.pos, os.SEEK_SET)

    # get a description of the type of data
    rec_typ = _TYPES[descr.typ]

    # pre-allocate an array so we can just read straight into it
    data = numpy.empty(descr.num, dtype=rec_typ.dsk)

    # start reading into the front of the array
    fst = 0

    # read as many records as needed from here on
    for rec_ndx in range(descr.rcs):  # pylint: disable=unused-variable
        # how many bytes are there in this particular record
        rec_siz, = struct.unpack('>i', fileobj.read(4))

        # an item cannot be split over two physical records
        assert (rec_siz % rec_typ.siz == 0)

        # how many *items* are there in this record
        rec_num = int(abs(rec_siz) / rec_typ.siz)

        # now we know which range we are going to read into;
        # bulk load it directly from file. character records need special
        # handling.
        if rec_typ.nch:
            data[fst:(fst + rec_num)] = (
                numpy.fromfile(fileobj, dtype=rec_typ.dsk, count=rec_num))
        else:
            # load the characters as uint8 raw bytes from the file
            raw = numpy.fromfile(fileobj, dtype=numpy.uint8,
                                 count=rec_num * rec_typ.siz)
            for item_ndx in range(rec_num):
                # start and end iterator for where in the raw array the bytes
                # for this particular item in the array is; each of the strings
                # are considered to be *exactly* 8 bytes by Eclipse
                beg = (item_ndx + 0) * rec_typ.siz
                end = (item_ndx + 1) * rec_typ.siz

                # convert these raw characters into a sensible string, and
                # store them in the output array
                str_val = codecs.ascii_decode(bytearray(raw[beg: end]))[0]
                data[fst + item_ndx] = str_val.rstrip()

        # skip the trailing record size
        fileobj.seek(4, os.SEEK_CUR)

        # move to the next window
        fst += rec_num

    # if we loaded big-endian data on a little-endian machine,
    # swap the entire array in-place now, and then cast it
    # (with a view) to the native type so we see the bits as
    # the right values again
    if rec_typ.nch:
        if sys.byteorder != 'big':
            data.byteswap(True)
        data = data.view(rec_typ.mem)
    else:
        # character data cannot be casted, but must be converted into the
        # correct number of bytes per character
        data = data.astype(rec_typ.mem)

    return data


class EclipseFile (object):
    """Low-level class to read records from binary files.

    Access to this object must be within a monitor (`with`-statement).
    """

    def __init__(self, root, ext):
        """Initialize file object from a path in the filesystem.

        :param root:  Stem of the file name (including directory)
        :param ext:   Extension of the file to read from.
        """
        # keep the file open (and locked) to read from it
        self.filename = '{0}.{1}'.format(root, ext.upper())
        self.fileobj = open(self.filename, 'rb')

        # index the file so that we can find properties easily
        log.debug("Indexing data file \"%s\"", self.filename)
        self.cnt, self.cat = _build_cat(self.fileobj)

    def __enter__(self):
        self.fileobj.__enter__()
        return self

    def __exit__(self, typ, val, traceback):
        """Close the underlaying file object when we go out of scope."""
        self.fileobj.__exit__(typ, val, traceback)

    def get(self, kwd, seq=0):
        """Read a (potentially indexed) keyword from file.

        :param kwd:  Keyword.
        :type kwd:   str
        :param seq:  Sequence number, starting from zero.
        :type seq:   int

        >>> with EclipseFile ('foo', 'EGRID') as grid:
        >>>     zcorn = grid.get ('ZCORN')
        """
        # make sure that the keyword is exactly 8 chars
        kwd = "{0:<8s}".format(kwd[:8].upper())

        # locate the data in the file and read them there, returning
        # None if we didn't find any matching record in the catalog
        key = _DataKey(kwd=kwd, seq=seq)
        if key in self.cat:
            descr = self.cat[key]
            data = _read_rec(self.fileobj, descr)
            return data
        else:
            return None

    def dump(self, positional=False, fileobj=sys.stdout):
        """Dump catalog contents of records in the datafile.

        :param positional:  True if the keywords should be sorted
                            on position in the file.
        :type positional:   bool

        >>> f = EclipseFile ("foo", 'INIT')
        >>> f.dump ()
        """
        if positional:
            # build a dictionary of positions and they record that
            # are at each of those positions. by using the position
            # as the key of the dictionary, we can sort them and then
            # retrive the keys in positionally sorted order
            by_pos = {}
            for key in self.cat.keys():
                descr = self.cat[key]
                by_pos[descr.pos] = key
            key_list = [by_pos[pos] for pos in sorted(by_pos.keys())]
        else:
            # sort the list of keywords alphabetically and then
            # by sequence number
            key_list = sorted(self.cat.keys())

        # write each entry in the catalog to the console
        for key in key_list:
            kwd, seq = key
            descr = self.cat[key]
            total = self.cnt[kwd]

            # print different text if there is only one of this keyword
            if total == 1:
                fmt = '{0:<8s}  {5:7s}  {2:>8d} * {3:4s}\n'
            else:
                fmt = '{0:<8s}  {1:03d}/{4:03d}  {2:>8d} * {3:4s}\n'

            # write information for this record
            fileobj.write(fmt.format(
                kwd, seq + 1, descr.num, descr.typ, total, ''))


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class EclipseGrid (object):
    """Corner-point geometry data from an Eclipse Extensive Grid file."""

    def __init__(self, root):
        # save the root variable for later
        self.root = root

        # read the extent of the grid right away since we need that for
        # later (when reading properties).
        with EclipseFile(root, "EGRID") as egrid:
            # read the grid header
            grid_head = egrid.get('GRIDHEAD')

            # only corner-point grids are supported
            assert (grid_head[0] == 1)

            # get dimensions of the grid
            self.ni = grid_head[1]  # pylint: disable=invalid-name
            self.nj = grid_head[2]  # pylint: disable=invalid-name
            self.nk = grid_head[3]  # pylint: disable=invalid-name
            log.info("Grid dimension is %d x %d x %d",
                     self.ni, self.nj, self.nk)

            # also store a shape tuple which describes the grid cube
            self.shape = (self.nk, self.nj, self.ni)

            # read the structure of the grid
            log.debug("Reading pillar coordinates")
            coord = egrid.get('COORD')
            self.coord = numpy.reshape(coord,
                                       (self.nj + 1, self.ni + 1, 2, 3))

            log.debug("Reading corner depths")
            zcorn = egrid.get('ZCORN')
            self.zcorn = numpy.reshape(zcorn,
                                       (self.nk, 2, self.nj, 2, self.ni, 2))

            log.debug("Reading active flag")
            actnum = egrid.get('ACTNUM')

            # make it a true flag, so we can use it for binary indexing
            actnum = actnum.astype(bool)

            # now we keep the actnum array in its proper shape,
            # corresponding to the grid cube
            self.actnum = numpy.reshape(actnum, self.shape)

            # create a common mask in the grid that we can use in
            # masked arrays for other properties.
            self.mask = numpy.logical_not(self.actnum)

            # restart properties are only saved for the active elements,
            # so we can cache this number to compare
            self.num_active = numpy.sum(self.actnum)
            log.info("Grid has %d active cells", self.num_active)

    def grid(self):
        """\
        Create a grid structure from the information read from file.

        :return:  Grid structure
        :rtype:   dict

        >>> # convert from .EGRID to .grdecl:
        >>> import pyresito.io.ecl as ecl
        >>> import pyresito.io.grdecl as grdecl
        >>> grdecl.write('foo.grdecl', ecl.EclipseGrid('FOO').grid())
        """
        # we have all the information that is required in the correct
        # format already as soon as we read it from file
        dimens = numpy.array([self.shape[2],
                              self.shape[1],
                              self.shape[0]], dtype=numpy.int32)

        # return grid in the same format as if it was read directly from
        # the original corner-point grid definition (.grdecl)
        return {'DIMENS': dimens,
                'COORD':  self.coord.copy(),
                'ZCORN':  self.zcorn.copy(),
                'ACTNUM': self.actnum.copy(),
                }


# Eclipse 300 on Windows initialize the intehead array to this value
# and doesn't bother to write all values
_UNINITIALIZED = -2345


def _intehead_date(intehead):
    """Get the date coded in the file header."""
    year = intehead[66]
    month = intehead[65]
    day = intehead[64]

    # libecl used in OPM seems to write only date information, not time,
    # unless it is necessary; the header will then be only 95 integers long
    # see function ecl_init_file_alloc_INTEHEAD at lines 56-59 in file
    # lib/ecl/ecl_init_file.cpp in libecl release 2018.10 commit 77ca7d5
    if (len(intehead) > 208):
        hour = intehead[206]
        minute = intehead[207]

        # take into account special values that indicate uninitialized field
        hour = 0 if hour == _UNINITIALIZED else hour
        minute = 0 if minute == _UNINITIALIZED else minute
    else:
        hour = 0
        minute = 0

    # init files from the Windows version doesn't have this field (?)
    if (len(intehead) >= 411):
        msecs = intehead[410]
    else:
        msecs = 0

    # split microseconds into seconds and remaining microseconds
    secs = int(msecs / 1e6)

    # spurious values found in this field which makes it unaligned with
    # the start date; better just drop the microsecond field
    msecs = 0   # msecs -= int (secs * 1e6)

    return datetime.datetime(year, month, day,
                             hour, minute, secs, msecs)


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class EclipseData (object):
    """Base class for both static and recurrent data."""

    def __init__(self, grid, root, ext):
        self.grid = grid  # grid extent information
        self.root = root  # underlaying file name
        self.ext = ext    # connected file with extent of grid

        # list of components isn't loaded yet (and may never be)
        self.comp = None

    def components(self):
        """Components that exist in the restart file. Components used in the
        simulation are stored in all the restart files instead of once in the
        init file, since it is the restart file that hold component fields.
        """
        # delay-load the list of components; if it already is loaded,
        # then just return the stored one. we cache component names since
        # they are needed to lookup property names.
        if self.comp is None:
            # load that datafile
            with EclipseFile(self.root, self.ext) as store:
                raw_comp = store.get('ZCOMPS')

            # create a dictionary over all the components, in the order they
            # were specified in the file, since component-based properties have
            # this index as part of their name.
            self.comp = collections.OrderedDict(
                [(str(nom), ndx + 1) for ndx, nom in enumerate(raw_comp)])

        return self.comp

    def _comp_phase(self, selector):  # pylint: disable=no-self-use
        """Code indicating the phase, when we are asking for a component-based
        property. This code is different than when asking for a phase-based
        property.
        """
        # second argument is the component, (optional) third argument is
        # the phase. when it comes to listing mole fractions, simulators (for
        # some unknown reason) uses other mnemonics than O, W, G to designate
        # phases.
        if len(selector) == 2:
            mph = 'Z'
        elif selector[2] == Phase.oil:
            mph = 'X'
        elif selector[2] == Phase.wat:
            mph = 'A'
        elif selector[2] == Phase.gas:
            mph = 'Y'
        else:
            assert False

        return mph

    def _get_prop_name(self, selector):  # pylint: disable=no-self-use
        """Compose an internal name for the property as specified by a
        high-level selector.

        :param selector:  Selector tuple, e.g. (Prop.mole, 'CO2', Phase.gas)
        :param names:     Dictionary of defined names in the case. There must
                          be an entry "components" containing the names of the
                          components in the case files.
        :returns:         Name of the property to be queried, e.g. "ZMF2"
        """
        # if the selector is a tuple, then use the first member to get a
        # function that can parse the others
        if isinstance(selector, tuple):
            # take the full selector (and the name dictionary to supplement)
            # and compose the final name. the selector can therefore be thought
            # of as an s-expr which is evaluated here
            if selector[0] == Prop.pres:
                name = 'PRESSURE'
            elif selector[0] == Prop.sat:
                # second argument is the phase
                name = 'S' + selector[1]
            elif selector[1] == Prop.mole:
                # get the phase scope of the component property
                scope = self._comp_phase(selector)

                # use dictionary to find the component index number
                ndx = ('' if selector[1] is None
                       else str(self.components()[selector[1]]))

                name = scope + 'MF' + ndx
            else:
                assert False

        else:
            # if it is not a selector, then it is (probably) a string, and
            # we forward it without any formatting, for compatibility
            name = selector

        return name

    def cell_data(self, selector):
        """Get a field property for every cell at this restart step.

        :param selector: Specification of the property to be loaded. This is a
                         tuple starting with a Prop, and then some context-
                         dependent items.
        :returns:        Array of the data with inactive cells masked off.

        >>> pres = case.cell_data ((Prop.pres,))
        >>> swat = case.cell_data ((Prop.sat, Phase.wat))
        >>> xco2 = case.cell_data ((Prop.mole, 'CO2', Phase.gas))
        """
        # get the internal name of the property requested
        propname = self._get_prop_name(selector)

        # load the data itself
        with EclipseFile(self.root, self.ext) as store:
            active_data = store.get(propname)

        # if there is a fully specified array, then just use the data
        if active_data.shape[0] == numpy.prod(self.grid.shape):
            data = numpy.reshape(active_data, self.grid.shape)

        # if the data is "compressed", i.e. only the active elements are
        # stored, then uncompress it before storing it in a masked array
        elif active_data.shape[0] == self.grid.num_active:
            # allocate an array to hold the entire array. fill with zeros
            # since this value exists for all datatypes (nan doesn't)
            data = numpy.zeros(self.grid.shape, dtype=active_data.dtype)

            # assign only the values that are actually active
            data[self.grid.actnum] = active_data
        else:
            assert (False)

        # return an array where the non-active elements are masked away
        return numpy.ma.array(data=data, dtype=data.dtype,
                              mask=self.grid.mask)

    def field_data(self, propname):
        """Get a property for the entire field at this restart step.

        :param propname: Name of the property to be loaded.
        :returns:        Array of the data.
        """
        # load the data itself
        with EclipseFile(self.root, self.ext) as store:
            return store.get(propname)

    def summary_data(self, propname):
        """Get a property from the summary file at this restart step.

        :param propname: Name of the property to be loaded. This is on the form
                         'mnemonic well', e.g. 'WWIR I05'. Alternatively, one
                         propname can be either only well or only mnemonic.
                         Then the value for all mnememonics or all wells are
                         given, e.g., propname='WWIR' returns WWIR for all
                         wells.
        :returns:        Array of the data.
        """
        # need to find the mneumonics and well info from the specification file
        with EclipseFile(self.root, 'SMSPEC') as store:
            mnemonic = store.get('KEYWORDS')
            well = store.get('WGNAMES')

        # split the propname
        prop_elem = propname.upper().split()

        assert len(prop_elem) < 3

        if len(prop_elem) == 1:  # only well or mnemonic is given
            if prop_elem in mnemonic:
                ind = numpy.where(mnemonic == prop_elem)
            elif prop_elem in well:
                ind = numpy.where(well == prop_elem)

        elif len(prop_elem) == 2:
            # find all indexes for the mnemonic
            tmp_ind = numpy.where(mnemonic == prop_elem[0])

            # fine all indexes for the well
            tmp_ind2 = numpy.where(well == prop_elem[1])

            # only one element is equal
            ind = tmp_ind[0][numpy.array(
                [elem in tmp_ind2[0] for elem in tmp_ind[0]])]

        # The challenge with the summary files is that it contains vectors of
        # values for all "ministeps" controlled by the internal time stepping
        # procedure. We are only interested in values at the report time, e.g.,
        # the final "ministep". It it, however, possible to collect all values,
        # e.g., for number of newton iterations.
        with EclipseFile(self.root, self.ext) as store:
            # Since we do not know the number of timesteps we must find how
            # many times the PARAM keyword has been given
            total = store.cnt['PARAMS  ']  # Gives number of ministeps

            # Get the final instance, need to remove one since python counts
            # from zero
            vec_dat = store.get(kwd='PARAMS  ', seq=total-1)

        return vec_dat[ind]


def _intehead_phases(intehead):
    """Get list of faces from the integer header"""
    # no phases discovered yet
    phs = []

    # build list of phases from the flags in the header
    if intehead[14] & 1 == 1:  # bit 0 set if oil present
        phs.append(Phase.oil)
    if intehead[14] & 2 == 2:  # bit 1 set if water present
        phs.append(Phase.wat)
    if intehead[14] & 4 == 4:  # bit 2 set if gas present
        phs.append(Phase.gas)

    return phs


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class EclipseInit (EclipseData):
    """Read information from static data (init) file."""

    def __init__(self, root):
        # bootstrap ourself; if the base class is reading a property
        # that require reformatting, then we have all the information
        # needed right here!
        super(EclipseInit, self).__init__(self, root, 'INIT')

        # read properties from file (notice that file is closed afterwards)
        with EclipseFile(self.root, self.ext) as store:
            intehead = store.get('INTEHEAD')
            porv = store.get('PORV')

        # get the dimensions from the grid
        self.ni = intehead[8]  # pylint: disable=invalid-name
        self.nj = intehead[9]  # pylint: disable=invalid-name
        self.nk = intehead[10]  # pylint: disable=invalid-name

        # notice that we store the grid in C order, i.e. the dimensions
        # are swapped compared with what they are in Eclipse (Fortran)
        self.shape = (self.nk, self.nj, self.ni)

        # get the starting date of the simulation
        self.start_date = _intehead_date(intehead)

        # get the phases in the simulation
        self.phases = _intehead_phases(intehead)

        # number of active elements is explicitly listed
        self.num_active = intehead[11]

        # get the pore volume and reformat it into the shape of the grid
        self.pore_vol = numpy.reshape(porv, self.shape)

        # create a mask and actnum array based on the pore volume
        # (active cells must have strictly positive volume)
        self.actnum = numpy.greater(self.pore_vol, 0.)
        self.mask = numpy.logical_not(self.actnum)
        assert (numpy.sum(self.actnum.ravel()) == self.num_active)


class EclipseRestart (EclipseData):
    """Read information from a recurrent data (restart) file."""

    def __init__(self, grid, seq):
        """
        :param grid:  Initialization file which contains grid dimensions.
        :type grid:   EclipseInit (or EclipseGrid)
        :param seq:   Run number.
        :type seq:    int
        """
        super(EclipseRestart, self).__init__(grid, grid.root,
                                             "X{0:04d}".format(seq))

    def date(self):
        """Simulation date the restart file is created for."""
        # dates are stored in the header
        with EclipseFile(self.root, self.ext) as store:
            intehead = store.get('INTEHEAD')

        # convert Eclipse date field to a Python date object
        return _intehead_date(intehead)
    
    def arrays(self):
        ecl_file = EclipseFile(self.root, self.ext)
        return [list(ecl_file.cat.keys())[i][0] for i, _ in enumerate(ecl_file.cat)]


class EclipseSummary (EclipseData):
    """Read information from a recurrent data (summary) file."""

    def __init__(self, grid, seq):
        """
        :param grid:  Initialization file which contains grid dimensions.
        :type grid:   EclipseInit (or EclipseGrid)
        :param seq:   Run number.
        :type seq:    int
        """
        super(EclipseSummary, self).__init__(grid, grid.root,
                                             "S{0:04d}".format(seq))

    def date(self):
        """Simulation date the restart file is created for."""
        # dates are stored in the header
        with EclipseFile(self.root, self.ext) as store:
            intehead = store.get('INTEHEAD')

        # convert Eclipse date field to a Python date object
        return _intehead_date(intehead)


def _quick_date(fileobj):
    """Quick scan of a restart file to determine its date.

    :param fileobj:  File object opened in binary read mode.
    """
    # INTEHEAD record is always placed first in the file; we have to read
    # four sectors; first sector contains the signature, second sector
    # (around 24 + 64*4 = 280) contains the day and (around 24 + 206*4 = 848)
    # time and fourth sector (around 24 + 410*4 = 1664) contains the seconds
    # we thus read four sectors continuously from the start of the file to
    # get maximum efficient I/O (remember we are trying to scan a lot of
    # files very quickly to build an index)
    fileobj.seek(0, os.SEEK_SET)
    block = fileobj.read(512 * 4)

    # verify the signature with 4 bytes leading length
    signature = codecs.ascii_decode(block[4:12])[0]
    assert (signature == 'INTEHEAD')

    # first record with the descriptor is 4 + 16 + 4 = 24 bytes long;
    # integer array is prefixed with another 4 bytes length, so the data
    # itself starts at 24 + 4 = 28 bytes (skipping 28 / 4 = 7 items --
    # they really didn't have aligned access in mind).
    # we don't care about the actual length of the block, as long as we
    # at least cover the number of items needed to extract the date
    intehead = numpy.frombuffer(block[28:],
                                dtype=numpy.dtype('>i4'),
                                count=512 - 7).copy()

    # the data we read was big-endian; convert it into a native format
    # for this architecture
    if sys.byteorder != 'big':
        intehead.byteswap(True)
    intehead = intehead.view(numpy.int32)

    # get the date for this header
    return _intehead_date(intehead)


class EclipseCase (object):
    """Read data for an Eclipse simulation case."""

    def __init__(self, casename):
        """
        :param casename:  Path to the case, with or without extension.
        :type casename:   str
        """
        # keys of this dictionary will be dates, the values are the sequence
        # numbers that are associated with those dates
        self.by_date = {}

        # get the directory of the file, using dot for current one if
        # no directory has been specified
        dir_name, file_name = os.path.split(casename)
        dir_name = '.' if dir_name == '' else path.expanduser(dir_name)

        # further remove the extension from the filename
        root, _ = os.path.splitext(file_name)

        # get a list of all restart files that are in this directory
        log.debug("Indexing restart files")
        for fname in os.listdir(dir_name):
            if fnmatch.fnmatch(fname,
                               '{0}.X[0-9][0-9][0-9][0-9]'.format(root)):
                # last four characters of the filename is the sequence number;
                # using int does not cause it to be interpreted octal even if
                # it starts with zero, fortunately
                seq = int(fname[-4:])

                # quickly scan this file to determine the correlation between
                with open(os.path.join(dir_name, fname), 'rb') as fileobj:
                    this_date = _quick_date(fileobj)
                self.by_date[this_date] = seq
        log.debug("Found %d restart files", len(self.by_date))

        # read the initial file to get the size of the grid and the
        # active flags, to load other properties
        self.root = os.path.join(dir_name, root)
        self.init = EclipseInit(self.root)

        # we haven't loaded any grid or recurrent data yet
        self._grid = None
        self.recur = {}
        self.sum = {}
        self.comp = None

    def shape(self):
        """\
        Get shape of returned field data.

        :return:  (num_k, num_j, num_i)
        :rtype:   (int, int, int)
        """
        return self.init.shape

    def start_date(self):
        """Starting date of the simulation"""
        return self.init.start_date

    def components(self):
        """Components that exist in the restart file.

        >>> case = EclipseCase (filename)
        >>> comps = case.components ()
        >>> print ("Number of components is %d" % (len (comps)))
        >>> for num, name in enumerate (comps):
        >>>     print ("%d : %s" % (num, name))
        """
        # get the last report date, and the index of that. the last
        # is selected because it is the most likely timestep to be
        # loaded by the user code (end results), and the number of
        # components shouldn't change during the run.
        lst_dat = max(self.by_date.keys())
        lst_seq = self.by_date[lst_dat]
        log.debug("Loading component list at date %s (step %04d)",
                  lst_dat.strftime('%Y-%m-%d'), lst_seq)
        restart = self._delay_load(lst_seq)
        return restart.components()

    def phases(self):
        """Phases that exist in the restart file.

        >>> case = EclipseCase (filename)
        >>> phs = case.phases ()
        >>> print ("Number of phases is %d" % (len (phs)))
        >>> print ("Keyword for first phase is %s" % ("S" + phs [0]))
        >>> if Phase.oil in phs:
        >>>     print ("Oil is present")
        >>> else:
        >>>     print ("Oil is not present")
        """
        return self.init.phases

    def report_dates(self):
        """List of all the report dates that are available in this
        case. Items from this list can be used as a parameter to `at`
        to get the case for that particular report step.

        :returns: List of available report dates in sequence
        :rtype:   list [datetime.datetime]

        >>> for rstp in case.report_dates ():
                print (case.at (rstp).this_date)

        See also
        --------
        [`ecl.EclipseCase.at`][]
        """
        return sorted(self.by_date.keys())

    def _delay_load(self, seq):
        """Make sure that recurrent data for a sequence step is loaded
        into memory.

        :param seq:   Sequence number of the restart
        :type seq:    int
        """
        # check if we have loaded the data file yet. since we need
        # to index the file while reading from it, we maintain a
        # cache of all the files that has been loaded.
        if seq in self.recur:
            restart = self.recur[seq]
        else:
            restart = EclipseRestart(self.init, seq)
            self.recur[seq] = restart

        return restart

    def at(self, when):  # pylint: disable=invalid-name
        """Recurrent data for a certain timestep. The result of this
        method is usually passed to function that need to calculate
        something using several properties.

        :param when:  Date of the property.
        :type when:   :class:`datetime.datetime` or int
        :returns:     Object containing properties for this timestep
        :rtype:       EclipseRestart
        """
        # get the sequence number of the report step; this is either
        # specified directly as an int, or given as a date
        if isinstance(when, datetime.datetime):
            assert (when in self.by_date)
            seq = self.by_date[when]
        else:
            seq = when

        restart = self._delay_load(seq)
        return restart

    def atsm(self, when):  # pylint: disable=invalid-name
        """Recurrent data from summary file for a certain timestep. The result
        of this method is usually passed to function that need to calculate
        something using several properties.

        :param when:  Date of the property.
        :type when:   :class:`datetime.datetime` or int
        :returns:     Object containing summary properties for this timestep
        :rtype:       EclipseSummary
        """
        # get the sequence number of the report step; this is either
        # specified directly as an int, or given as a date
        if isinstance(when, datetime.datetime):
            assert (when in self.by_date)
            seq = self.by_date[when]
        else:
            seq = when

        # check if we have loaded the data file yet. since we need
        # to index the file while reading from it, we maintain a
        # cache of all the files that has been loaded.
        if seq in self.sum:
            summary = self.sum[seq]
        else:
            summary = EclipseSummary(self.init, seq)
            self.sum[seq] = summary

        return summary

    def cell_data(self, prop, when=None):
        """Read cell-wise data from case. This can be either static information
        or recurrent information for a certain date.

        :param prop:  Name of the property, e.g. 'SWAT'
        :type prop:   str
        :param when:  Date of the property, or None if static
        :type when:   :class:`datetime.datetime` or int
        :returns:     Loaded array for the property.
        :rtype:       :class:`numpy.ndarray`

        Examples
        --------
        >>> case = EclipseCase (cmd_args.filename)
        >>> zmf2 = case.cell_data ('ZMF2', datetime.datetime (2054, 7, 1))
        """
        # if it is static property, it is found in the initial file
        if when is None:
            return self.init.cell_data(prop)
        else:
            # ask the restart file for the data
            return self.at(when).cell_data(prop)

    def field_data(self, prop, when=None):
        """Read field-wise data from case. This can be either static information
        or recurrent information for a certain date.

        :param prop:  Name of the property, e.g. 'ZPHASE'
        :type prop:   str
        :param when:  Date of the property, or None if static
        :type when:   :class:`datetime.datetime` or int
        :returns:     Loaded array for the property.
        :rtype:       :class:`numpy.ndarray`

        :example:
        >>> case = EclipseCase (cmd_args.filename)
        >>> zphase = case.cell_data ('ZPHASE', datetime.datetime (2054, 7, 1))
        """
        # if it is static property, it is found in the initial file
        if when is None:
            return self.init.field_data(prop)
        else:
            # ask the restart file for the data
            return self.at(when).field_data(prop)

    def summary_data(self, prop, when):
        """Read summary data from case. This is typically well data, but can
        also be e.g., newton iterations.

        :param prop: Name of the property, e.g., 'WWPR PRO1'.
        :type prop:  str
        :param when: Date of the property
        :type when:  :class:`datetime.datetime` or int
        :returns:     Loaded array for the property.
        :rtype:       :class:`numpy.ndarray`
        :example:
        >>> case = EclipseCase (cmd_args.filename)
        >>> data = case.cell_data ('WWIR INJ-5',
                                   datetime.datetime (2054, 7, 1))
        """
        # Summary data is a bit more tricky than restart data. Mostly because
        # all specifications are given in the SMSPEC file. We must therefore
        # start by loading the mnemonic and the well associated with each data
        # vector
        return self.atsm(when).summary_data(prop)

    def grid(self):
        """Grid structure for simulation case."""
        # on-demand load the grid file
        if self._grid is None:
            self._grid = EclipseGrid(self.init.root)

        # offload this routine to the grid object
        return self._grid.grid()

    def arrays(self, when):
        return self.at(when).arrays()


class EclipseRFT (object):
    """Read data from an Eclipse RFT file."""

    def __init__(self, casename):
        """
        :param casename:  Path to the case, with or without extension.
        :type casename:   str
        """
        # get the directory of the file, using dot for current one if
        # no directory has been specified
        dir_name, file_name = os.path.split(casename)
        dir_name = '.' if dir_name == '' else dir_name

        # further remove the extension from the filename
        root, _ = os.path.splitext(file_name)

        # read the initial file to get the size of the grid and the
        # active flags, to load other properties
        self.root = os.path.join(dir_name, root)

        self.recur = {}
        self.sum = {}

    def rft_data(self, well, prop):
        """
        Read the rft data for the requested well
        :param well: Name of well, e.g. 'PRO-1'
        :param prop: Type of property (depth, pressure, swat, or sgas)
        :type well: str
        :type prop: str
        :return: Loaded array for the property.
        :rtype: :class:`numpy.ndarray`
        >>> case = EclipseRFT (cmd_args.filename)
        >>> data = case.rft_data (well='INJ-5', prop='PRESSURE')
        """
        with EclipseFile(self.root, 'RFT') as eclf:
            vec_dat = None
            # Find the array from the current well
            for index in range(eclf.cnt['WELLETC ']):
                # well name for current index
                if eclf.get(kwd='WELLETC', seq=index)[1] == well.upper():
                    # check that this well has RFT data
                    assert 'R' in eclf.get(kwd='WELLETC', seq=index)[5]
                    vec_dat = eclf.get(kwd=prop, seq=index)
                    break

        return vec_dat


def main(*args):
    """Read a data file to see if it parses OK."""
    # setup simple logging where we prefix each line with a letter code
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname).1s: %(message).76s")

    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    cmd_args = parser.parse_args(*args)

    # process file
    root, ext = os.path.splitext(cmd_args.filename)
    with EclipseFile(root, ext[1:]) as eclf:
        eclf.dump(True)


# executable library
if __name__ == "__main__":
    main(sys.argv[1:])
