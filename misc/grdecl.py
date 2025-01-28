# pylint: disable=too-many-lines
"""Read Schlumberger Eclipse grid input files.

Examples
--------
>>> import pyresito.io.grdecl as grdecl
>>> grid = grdecl.read ("FOO")
"""
from __future__ import division
import argparse
import codecs  # pylint: disable=unused-import
import collections as col
import contextlib as ctx
import logging
import math
import mmap
import numpy
import numpy.ma
import os
import os.path
import re
from six.moves import range  # pylint: disable=redefined-builtin, import-error
import six
import sys


# even though the memoryview api exists in Python 2, the regular expression
# engine cannot use it, so we must keep the old buffer interface around, too
if sys.version_info[0] < 3:
    def membuf(mem, bgn, end):  # pylint: disable=missing-docstring
        return buffer(mem, bgn, end - bgn)

else:
    def membuf(mem, bgn, end):  # pylint: disable=missing-docstring
        return memoryview(mem)[bgn: end]


# there is a bug in the interaction between int() and memoryview in
# Python 3: it attempts to interpret the entire buffer instead of just
# the subset are indexed. this is solved by casting it to a bytebuffer
# before conversion, so no string decoding needs to take place
if sys.version_info[0] < 3:
    def conv_atoi(buf):  # pylint: disable=missing-docstring
        return int(buf)

    def conv_atof(buf):  # pylint: disable=missing-docstring
        return float(buf)

else:
    def conv_atoi(buf):  # pylint: disable=missing-docstring
        return int(bytes(buf))

    def conv_atof(buf):  # pylint: disable=missing-docstring
        return float(bytes(buf))


# file on disk is read as single bytes, whereas text in memory is of the string
# type. these helper functions convert between the two, in Python 2 there is no
# difference between the types
if sys.version_info[0] < 3:
    def enc(text):  # pylint: disable=missing-docstring
        return text

    def dec(buf):  # pylint: disable=missing-docstring
        return buf

else:
    def enc(text):  # pylint: disable=missing-docstring
        return codecs.utf_8_encode(text)[0]

    def dec(buf):  # pylint: disable=missing-docstring
        return codecs.utf_8_decode(buf)[0]


# add a valid log in case we are not run through the main program which
# sets up one for us
log = logging.getLogger(__name__)  # pylint: disable=invalid-name
log.addHandler(logging.NullHandler())


class GrdEclError (Exception):
    """Exception thrown if invalid syntax is encountered."""

    def __init__(self, path, pos, message):
        line, column = pos
        # include the filename if any was specified. only print the name,
        # not the entire path, since this may be a very long string
        fmt = ("in \"{0}\" " if path is not None else "") + "at ({1},{2}): {3}"
        basename = (os.path.basename(path) if path else "")
        super(GrdEclError, self).__init__(
            fmt.format(basename, line, column, message))


# pylint: disable=invalid-name
# tokens that exists in Eclipse grid files:

# whitespace is everything that is between other tokens; I've put comments
# into this class so that they can appear inside the data tables (e.g. for
# headings and dividers); it shouldn't add much to the processing time since
# they can be decided with very little look-ahead
whitespace = re.compile(enc(r'[\ \t]+|--[^\r]*'))

# newlines need to be their own tokens since we are going to update the
# position counter at each such occurrence (source files are better viewed
# as (line, column) rather than byte offset)
newline = re.compile(enc(r'\r?\n'))

# cardinals are integer numbers that describe sizes, e.g. the number of
# rows in a spatial direction
cardinal = re.compile(enc(r'[0-9]+'))

# multiple cardinals prefixed with an optional repeat count
multint = re.compile(enc(r'([0-9]+\*)?([0-9]+)'))

# single floating-point number without any count. coordinates are described
# using this type, since there is no meaning in defining a repeat count that
# goes across both x and y-coordinates.
_digs = r'([0-9]?\.[0-9]+|[0-9]+(\.[0-9]+)?)([eEdD][-+]?[0-9]+)?'

# properties that are imported from other sources may also have NaN written
# in cells where the value doesn't apply; Petrel will understand this, but
# Eclipse won't.
_nan = r'[nN][aA][nN]'

# regular expression that accepts both NaNs and single numbers
_anumb = r'-?(' + _nan + r'|' + _digs + r')'
single = re.compile(enc(_anumb))

# floating-point numbers. the bulk of the data will be tokens of this type.
# first group is the repeat count, the sign is optional, then comes two
# subgroups: either the integer part or the fractional part of the number
# can be omitted, but not both. third group is the exponent, allowing for
# Fortran-style single precision indicator
number = re.compile(enc(r'([0-9]+\*)?(' + _anumb + r')'))

# most records ends with a slash; this is used as a guard token to ensure
# that we are parsing the file correctly
endrec = re.compile(enc(r'/'))

# a flag is encoded as zero or one, with an optional repeat count. this
# type of lexeme is used to parse the ACTNUM property.
flag = re.compile(enc(r'([0-9]+\*)?([01])'))

# keywords are section headers. they are really only supposed to be 8
# characters but Petrel will write longer names for custom attributes
keyword = re.compile(enc(r'[a-zA-Z_]+'))

# filenames are used to include; either we specify more or less anything
# as long as it is between quotes (notably including forward slashes!), or
# it can be something that resembles a sensible MS-DOS name (including
# colons and back-slashes!)
filename = re.compile(enc(r'(\'[^\']*\')|([a-zA-Z0-9_\-\.\:\\\ ]+)'))


class _Lexer (object):
    """View the input file as a stream of tokens"""

    # pylint: disable=unused-argument
    def __init__(self, filename, mem):
        """Lex a given input file with an opened memory mapping."""

        super(_Lexer, self).__init__()

        # line number is one-based, and we start at the first line
        self.lineno = 1

        # this is the file offset of the first character on the current
        # line. we use this to determine the column number (it is not
        # updated whenever we read something, but rather generated on demand)
        self.startofs = 0

        # this is the position of where we last found a token; we store the
        # offset into the file, and can find the column from that
        self.last_pos = 0

        # use the memory-mapping of the file as if it was a string buffer
        self.mem = mem

        # when viewed as a byte-buffer/string memory mapped files always
        # "start" at the beginning of the file, regardless of the cursor.
        # we therefore maintain a separate buffer object that points into
        # the current position in the file. the reason that we bother to
        # use the position at all is in case it signals to the os where
        # to put the memory window of the file
        self.buf = membuf(self.mem, self.mem.tell(), self.mem.size())

        # save a reference to the filename for use in error messages
        self.path = filename

    def close(self):
        """Clean up internal memory buffer usage"""
        self.buf = None
        self.mem = None

    def _advance(self, length):
        """Advance the current position of the file by length bytes. length
        is commonly the length of the last read token.
        """
        self.mem.seek(length, os.SEEK_CUR)
        self.buf = membuf(self.mem, self.mem.tell(), self.mem.size())

    def skip_blanks(self):
        """Skip one or more blank tokens, updating the counter as we go.
        """
        done = False
        while not done:
            match = re.match(whitespace, self.buf)
            if match:
                # advance the stream to skip over the whitespace
                self._advance(match.end() - match.start())
            else:
                # maybe it wasn't blank whitespace, but rather a newline?
                match = re.match(newline, self.buf)

                if match:
                    # still skip this token
                    self._advance(match.end() - match.start())

                    # but now also update the line counter
                    self.lineno = self.lineno + 1
                    self.startofs = self.mem.tell()
                else:
                    # it is a legitimate token, return for further processing;
                    # the memory stream will not have advanced, so the token
                    # is still at the beginning of the stream buffer
                    done = True

    def pos(self):
        """Return the current position in the file as (line, column)"""
        # we have the line counter available; the column counter is
        # available from looking how far we have progressed since the
        # last linebreak. we return the position of the last token,
        # which may not be visible to the parser (can be converted to number)
        return (self.lineno, self.last_pos - self.startofs + 1)

    def expect(self, lexeme):
        """Expect a certain type of lexeme, e.g. keyword, to appear as the
        next non-whitespace token in the stream.
        """
        # skip any blanks in front of the wanted token
        self.skip_blanks()

        # next token is going to be at this position
        self.last_pos = self.mem.tell()

        # attempt to match the lexeme in question
        match = re.match(lexeme, self.buf)

        if match:
            # read off the token that was matched into a separate string
            token = match.group(0)

            # update the buffer pointer to the new position
            self._advance(match.end() - match.start())

            return token
        else:
            return None

    def expect_count(self, lexeme):
        """Expect a production where the first group is an optional repeat
        count. The function returns a tuple where the first item is the value
        and the second one is the repeat count.
        """
        # skip any blanks in front of the wanted token
        self.skip_blanks()

        # next token is going to be at this position
        self.last_pos = self.mem.tell()

        # attempt to match the lexeme
        match = re.match(lexeme, self.buf)

        if match:
            # if the count group is not present, then set count to be 1
            if match.group(1) is None:
                count = 1
            else:
                count = conv_atoi(match.group(1)[:-1])

            # the value is the second group, capturing the rest of the token
            value = match.group(2)

            # update the buffer pointer to the new position
            self._advance(match.end() - match.start())

            return (count, value)
        else:
            return (0, None)

    def expect_str(self, lexeme):
        """Expect a text token."""
        txt = self.expect(lexeme)
        return (None if txt is None else dec(txt))

    def expect_filename(self):
        """Expect a filename."""
        name = self.expect(filename)
        # specification of filenames allow spaces at the beginning and the
        # end. since the slash is not part of the filename but rather and
        # end record, it is natural to separate them with a space. to avoid
        # an overly complex regular expression, we do the trimming here
        return (None if name is None else dec(name).strip().strip("\'"))

    def expect_enum(self, keywords, kind):
        """Expect a keyword in a given set, or throw error if not found"""
        token = self.expect_str(keyword)
        if not token:
            raise GrdEclError(self.path, self.pos(),
                              "Expected keyword for {0}".format(kind))
        if token not in keywords:
            raise GrdEclError(self.path, self.pos(),
                              "Unsupported {0} \"{1}\"".format(kind, token))
        return token

    def expect_single(self):
        """Expect a single floating point number"""
        token = self.expect(single)
        if not token:
            raise GrdEclError(self.path, self.pos(),
                              "Expected floating-point number")
        return float(token)

    def expect_cardinal(self):
        """Expect a single cardinal number"""
        token = self.expect(cardinal)
        if not token:
            raise GrdEclError(self.path, self.pos(),
                              "Expected positive, integer number")
        return conv_atoi(token)

    def expect_multi_card(self):
        """Expect multiple cardinal numbers"""
        # use scanning routine which takes care of the optional count with star
        # in front of our own lexeme
        count, value = self.expect_count(multint)

        # convert the string itself to our integral type afterwards
        return (count, numpy.int32(value))

    def expect_numbers(self):
        """Expect a floating-point number with an optional repeat count"""
        count, value = self.expect_count(number)
        # convert to byte buffer to be able to search and modify it
        value = bytearray(value)

        # check if we are using the Fortran-style exponent, and it that
        # case replace it with something our runtime understands
        ndx = value.find(b'd')
        if ndx != -1:
            value[ndx] = b'e'
        else:
            ndx = value.find(b'D')
            if ndx != -1:
                value[ndx] = b'e'

        return (count, numpy.float64(value))

    def expect_bools(self):
        """Expect boolean flags with an optional repeat count"""
        count, value = self.expect_count(flag)
        # if we don't convert to int first and then to bool, the conversion
        # will always yield a value of True!
        return (count, bool)

    def at_eof(self):
        """Check if the lexer has progressed to the end of the stream"""
        return self.mem.tell() == self.mem.size()


class _FileMapPair (object):  # pylint: disable=too-few-public-methods
    """An open file with an associated memory-mapping"""

    def __init__(self, path):
        """Open file and mapping from a path specification"""

        # must open files as binary on Windows NT, or they are automagically
        # converted into carriage-return/newline! also, by declaring that we
        # are reading the file sequentially, the fs cache can optimize.
        # pylint: disable=no-member
        flags = os.O_BINARY | os.O_SEQUENTIAL if os.name == 'nt' else 0
        self.file_handle = os.open(path, os.O_RDONLY | flags)

        # tell the operating system that we intend to read the file in one go
        if (sys.platform.startswith('linux') and
                sys.version_info[0:2] >= (3, 3)):
            os.posix_fadvise(self.file_handle, 0, 0, os.POSIX_FADV_SEQUENTIAL)

        # attempt to create a memory-mapping of the file; if this doesn't
        # succeed, then close the underlaying file as well
        try:
            self.map_obj = mmap.mmap(self.file_handle, 0,
                                     access=mmap.ACCESS_READ)
        except Exception:
            os.close(self.file_handle)
            raise

    def close(self):
        """Close both the file and the memory mapping"""
        try:
            self.map_obj.close()
        finally:
            os.close(self.file_handle)


class _OwningLexer (_Lexer):
    """Lexer that owns exclusively the underlaying memory map"""

    def __init__(self, path):
        # first attempt to open the file and map it; if this is successful,
        # then create a lexer using the memory-map (and keep the file object
        # in ourself). if the lexer could not be created, then make sure that
        # the file doesn't leak
        self.fmp = _FileMapPair(path)
        try:
            super(_OwningLexer, self).__init__(path, self.fmp.map_obj)
        except Exception:
            self.fmp.close()
            raise

    def close(self):
        """Clean up all resources associated with the lexer"""

        # let the superclass clear its internal references first, and then
        # make sure that the underlaying OS handles are closed afterwards
        try:
            super(_OwningLexer, self).close()
        finally:
            self.fmp.close()


# pylint: disable=too-many-arguments, too-many-locals, invalid-name
def _axes_rot(x1, y1, x2, y2, x3, y3):
    """
    Rotation matrix and offset vector for axes mapping.

    Returns
    -------
    tuple
        A tuple (A, b) where A is the rotation matrix and b is the offset vector.
        Each point x in the coordinate array should be rotated with A(x - b) + b.
    """
    # y axis vector
    v1_x = x1 - x2
    v1_y = y1 - y2
    v1_len = math.sqrt(v1_x * v1_x + v1_y * v1_y)
    ey_x = v1_x / v1_len
    ey_y = v1_y / v1_len

    # x axis vector
    v3_x = x3 - x2
    v3_y = y3 - y2
    v3_len = math.sqrt(v3_x * v3_x + v3_y * v3_y)
    ex_x = v3_x / v3_len
    ex_y = v3_y / v3_len

    # define transformation; each point x in COORD array is rotated A(x-b)+b
    A = numpy.array([[ex_x, ey_x, 0.],
                     [ex_y, ey_y, 0.],
                     [0.,   0.,   1.]], dtype=numpy.float64)
    b = numpy.array([x2, y2, 0], dtype=numpy.float64)

    return (A, b)


# pylint: disable=invalid-name
def _no_rot():
    """Rotation matrix and offset vector if there is no axes mapping."""
    A = numpy.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]], dtype=numpy.float64)
    b = numpy.array([0., 0., 0.], dtype=numpy.float64)
    return (A, b)


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class _Parser (object):
    """Read sections usually exported by Petrel from an Eclipse cornerpoint grid
    file.
    """

    def __init__(self, filename, mem):
        super(_Parser, self).__init__()

        # these sections have their own productions, because they need special
        # treatment or give side-effects
        self.prod = {}
        self.prod['GRID'] = self._ignore
        self.prod['PINCH'] = self._pinch
        self.prod['NOECHO'] = self._ignore
        self.prod['MAPUNITS'] = self._mapunits
        self.prod['MAPAXES'] = self._mapaxes
        self.prod['GRIDUNIT'] = self._gridunit
        self.prod['SPECGRID'] = self._specgrid
        self.prod['COORDSYS'] = self._coordsys
        self.prod['COORD'] = self._coord
        self.prod['ZCORN'] = self._zcorn
        self.prod['ACTNUM'] = self._actnum
        self.prod['ECHO'] = self._ignore
        self.prod['INCLUDE'] = self._include
        self.prod['DIMENS'] = self._dimens

        # this list contains the keywords that are all readable in the same way
        self.cell_sections = [x for x in _SECTIONS.keys()
                              if _SECTIONS[x]['cell']]

        # if no rotation matrix is defined, then use identity
        self.rot_A, self.rot_b = _no_rot()

        # this class does the interpretation of the bytes on disk
        self.lex = _Lexer(filename, mem)

        # each section is read into a dictionary object
        self.section = dict()

        # before we have read anything, we don't know the dimensions of
        # the grid
        self.ni = None
        self.nj = None
        self.nk = None

        # if we have inactive cells, this attribute tells us where they are
        # so that we can create a masked array
        self.mask = None

        # if partial files are opened, then the old lexers are pushed onto
        # this stack while the new file is parsed. when the new file is
        # finished, then parsing resumes from the top of this stack
        self.lexer_stack = []

    def close(self):
        """Clean up internal memory buffer usage"""
        # if we were interrupted while we were in the middle of a parse,
        # then it might be that there are several opened sub-files that must
        # be closed. normally this loop should be empty.
        while len(self.lexer_stack):
            self.lexer_stack.pop().close()

        self.lex.close()
        self.lex = None

    def _include(self):
        """Continue parsing a fragment file at this point"""
        # read the file name from the keyword stream
        name = self.lex.expect_filename()
        if not name:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              ("Include statements must be followed by "
                               "a file name"))
        self.lex.expect(endrec)

        # included files should be read relative to the file that refer to
        # them, not to any arbitrary current directory of the process.
        fullname = os.path.normpath(os.path.join(os.path.dirname(
            self.lex.path), name))

        log.info('Reading from file "%s"', name)
        # put the current lexer on hold, and put a new lexer in its place so
        # that parsing continues from the new file
        self.lexer_stack.append(self.lex)
        self.lex = _OwningLexer(fullname)

    def _ignore(self):
        """Ignore this single keyword"""
        pass

    def _pinch(self):
        """Production for PINCH keyword"""
        # this is just a processing flag that is set; nothing more to read
        self.lex.expect(endrec)

    def _mapunits(self):
        """Production for MAPUNITS keyword"""
        # read the keyword for units specified. currently, only specifying
        # metres directly is supported. alternatively, a different unit should
        # be looked up here in a dictionary and a scaling factor set
        self.lex.expect_enum(['METRES'], "unit")
        self.lex.expect(endrec)

    def _mapaxes(self):  # pylint: disable=too-many-locals
        """Production for MAPAXES keyword"""
        # a point on the Y axis
        x1 = self.lex.expect_single()
        y1 = self.lex.expect_single()

        # origin
        x2 = self.lex.expect_single()
        y2 = self.lex.expect_single()

        # a point on the X axis
        x3 = self.lex.expect_single()
        y3 = self.lex.expect_single()

        # even though data is fixed-length, it needs a record terminator
        self.lex.expect(endrec)

        self.rot_A, self.rot_b = _axes_rot(x1, y1, x2, y2, x3, y3)

    def _gridunit(self):
        """Production for GRIDUNIT keyword"""
        self.lex.expect_enum(["METRES"], "unit")

        # map flag is optional
        token = self.lex.expect_str(keyword)
        if token:
            if token != "MAP":
                raise GrdEclError(self.lex.path, self.lex.pos(),
                                  ("Expected flag MAP or nothing, "
                                   "not \"{0}\"").format(token))
        self.lex.expect(endrec)

    def _specgrid(self):
        """Production for the SPECGRID keyword"""
        # grid dimensions; we read this directly into the object fields
        # since we need these to allocate memory later
        self.ni = self.lex.expect_cardinal()
        self.nj = self.lex.expect_cardinal()
        self.nk = self.lex.expect_cardinal()

        # also add this as if we had read the DIMENS keyword
        self.section['DIMENS'] = numpy.array([self.ni, self.nj, self.nk],
                                             dtype=numpy.int32)

        # number of reservoirs are not used in Eclipse, but are in the API
        numres = self.lex.expect_cardinal()
        if numres != 1:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "Expected only one reservoir, but got {0:d}",
                              numres)

        # radial coordinates are not supported yet
        token = self.lex.expect_str(keyword)
        if token:
            if token != "F":
                raise GrdEclError(self.lex.path, self.lex.pos(),
                                  ("Expected F for Cartesian coordinates "
                                   "but got \"{0}\"").format(token))
        # end-of-record sentinel in case any tokens are defaulted
        self.lex.expect(endrec)

    def _dimens(self):
        """Production for the DIMENS keyword"""
        ni = self.lex.expect_cardinal()
        nj = self.lex.expect_cardinal()
        nk = self.lex.expect_cardinal()
        self.section['DIMENS'] = numpy.array([ni, nj, nk],
                                             dtype=numpy.int32)

        self.lex.expect(endrec)

    def _coordsys(self):
        """Production for the COORDSYS keyword"""
        # we are reading only one grid, so we don't really have any notion
        # about where this is going to end up in the total reservoir.
        self.lex.expect_cardinal()  # k_lower =
        self.lex.expect_cardinal()  # k_upper =
        # there can be more arguments to this keyword, but Petrel will only
        # write the first two
        self.lex.expect(endrec)

    def _coord(self):
        """Production for the COORD keyword"""
        # check that we actually know how many numbers to read
        if 'DIMENS' not in self.section:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "COORD section must come after SPECGRID")

        # each column/row of cells have a pillar on either side, meaning
        # that there is an extra one at each end, and there is a top and
        # a bottom coordinate for every pillar
        num_coord = (self.ni + 1) * (self.nj + 1) * 2

        # allocate memory
        coord = numpy.empty((num_coord, 3), dtype=numpy.float64)

        # read three and three numbers, making out a coordinate. we take
        # care not to allocate any extra memory in this loop
        vec = numpy.empty((3, ), dtype=numpy.float64)
        for i in range(num_coord):
            # read coordinate specified in file, into intermediate vector
            vec[0] = self.lex.expect_single()  # x
            vec[1] = self.lex.expect_single()  # y
            vec[2] = self.lex.expect_single()  # z

            # rotate according to map axes, and assign into global grid
            vec -= self.rot_b
            numpy.dot(self.rot_A, vec, out=coord[i, :])
            coord[i, :] += self.rot_b

        # after we're done reading, then rearrange the grid so that is
        # is no longer a list of coordinates, but rather a true grid
        self.section['COORD'] = numpy.reshape(
            coord, (self.nj + 1, self.ni + 1, 2, 3))

        # after all the coordinates are read, we expect the sentinel token
        self.lex.expect(endrec)

    def _zcorn(self):
        """Production for the ZCORN keyword"""
        # check that we actually know how many numbers to read
        if 'DIMENS' not in self.section:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "ZCORN section must come after SPECGRID")

        # allocate memory
        num_zcorn = self.nk * self.nj * self.ni * 2 * 2 * 2
        zcorn = numpy.empty((num_zcorn, ), dtype=numpy.float64)

        # read numbers from file; each read may potentially fill a different
        # length of the array
        num_read = 0
        while num_read < num_zcorn:
            count, value = self.lex.expect_numbers()
            zcorn[num_read: num_read+count] = value
            num_read += count

        # verify that we got exactly the number of corners we wanted
        if num_read > num_zcorn:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "Expected {0:d} numbers, but got {1:d}".format(
                num_zcorn, num_read))

        # reformat the data into a natural kji/bfr hypercube
        self.section['ZCORN'] = numpy.reshape(
            zcorn, (self.nk, 2, self.nj, 2, self.ni, 2))

        # after all the data, there should be a sentinel slash
        self.lex.expect(endrec)

    def _actnum(self):
        """Production for the ACTNUM keyword"""
        # check that we actually know how many numbers to read
        if 'DIMENS' not in self.section:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "ACTNUM section must come after SPECGRID")

        # allocate memory
        num_cells = self.nk * self.nj * self.ni
        actnum = numpy.empty((num_cells, ), dtype=bool)

        # read numbers from file; each read may potentially fill a different
        # length of the array
        num_read = 0
        while num_read < num_cells:
            count, value = self.lex.expect_bools()
            actnum[num_read: num_read+count] = value
            num_read += count

        # verify that we got exactly the number of corners we wanted
        if num_read > num_cells:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "Expected {0:d} numbers, but got {1:d}".format(
                num_cells, num_read))

        # if not all cells were specified, the rest should be marked as
        # inactive
        if num_read < num_cells:
            actnum[num_read:] = False

        # reformat the data into a natural kji/bfr hypercube
        self.section['ACTNUM'] = numpy.reshape(
            actnum, (self.nk, self.nj, self.ni))

        # mask is simply the opposite of the active flag
        self.mask = numpy.logical_not(self.section['ACTNUM'])

        # after all the data, there should be a sentinel slash
        self.lex.expect(endrec)

    def _celldata(self, kw):
        """Production for the keywords that load data per cell"""
        # check that we actually know how many numbers to read
        if 'DIMENS' not in self.section:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "{0} section must come after SPECGRID".format(
                kw))

        # check if this keyword is registered with a known type
        typ = _kw_dtype(kw)

        # allocate memory
        num_cells = self.nk * self.nj * self.ni
        data = numpy.empty((num_cells, ), dtype=typ)

        # read numbers from file; each read may potentially fill a different
        # length of the array. use an almost identical type of loop for
        # integers and floating-point, but avoid using test inside the loop
        # or polymorphism through function pointer, to ease optimization
        num_read = 0
        if numpy.issubdtype(typ, numpy.integer):
            while num_read < num_cells:
                count, value = self.lex.expect_multi_card()
                data[num_read: num_read+count] = value
                num_read += count
        else:
            while num_read < num_cells:
                count, value = self.lex.expect_numbers()
                data[num_read: num_read+count] = value
                num_read += count

        # verify that we got exactly the number of corners we wanted
        if num_read > num_cells:
            raise GrdEclError(self.lex.path, self.lex.pos(),
                              "Expected {0:d} numbers, but got {1:d}".format(
                num_cells, num_read))

        # reformat the data into a natural kji/bfr hypercube
        data = numpy.reshape(data, (self.nk, self.nj, self.ni))
        if self.mask is not None:
            # integral data is mostly category indices; -1 tends to be an
            # over-all good enough value to use for that purpose, but Petrel
            # won't understand it if we are reading the file back in.
            if numpy.issubdtype(data.dtype, numpy.integer):
                data[self.mask] = 0
            else:
                data[self.mask] = numpy.nan
            self.section[kw] = numpy.ma.array(data=data,
                                              dtype=data.dtype,
                                              mask=self.mask)
        else:
            self.section[kw] = data

        # after all the data, there should be a sentinel slash
        self.lex.expect(endrec)

    def parse(self):
        """Parse the entire file"""
        # read all sections of the file
        while True:
            # read a keyword first
            kw = self.lex.expect_str(keyword)

            # check here, after we have read the keyword, for end-of-file
            # since it may be that the function above skip some blanks
            # at the end without ever returning a keyword
            if self.lex.at_eof():
                # if there are any nested files open, then close this file
                # and continue reading from the previous one
                if len(self.lexer_stack):
                    self.lex.close()
                    self.lex = self.lexer_stack.pop()
                    log.info('Resume reading from "%s"', self.lex.path)
                    continue
                else:
                    break

            # if this is a recognizable keyword, then call its handler
            # function; otherwise raise an error about the keyword
            if kw:
                if kw in self.prod:
                    log.info("Reading keyword %s", kw)
                    self.prod[kw]()
                elif kw in self.cell_sections:
                    log.info("Reading keyword %s", kw)
                    self._celldata(kw)
                else:
                    raise GrdEclError(self.lex.path, self.lex.pos(),
                                      "Unknown keyword \"{0}\"".format(kw))
            else:
                raise GrdEclError(self.lex.path, self.lex.pos(),
                                  "Expected keyword here")

        return self.section


_GEN_BGN = b'-- Generated [\r\n'
_GEN_END = b'-- Generated ]\r\n'


def _parse_meta(mem):
    """
    Parse the file headers for properties set by the generator.

    These properties can provide hints for efficient file processing.

    Parameters
    ----------
    mem : mmap.mmap
        Opened memory-map for the main input file.

    Returns
    -------
    dict
        Properties set in the header of the data file.
    int
        Number of bytes that can be skipped to avoid processing this header again (it will be in comments).
    """
    # dictionary that will receive parsed headers, if any
    props = col.OrderedDict()

    # in case of an early exit by the below test, we don't want the client
    # to skip anything (there is no header)
    nxt = 0

    # if the very first line is a special header, then start reading
    # the properties from the beginning of the file
    if mem.find(_GEN_BGN, 0, len(_GEN_BGN)) != -1:
        # skip the initial indicator, go straight to the first property
        pos = len(_GEN_BGN)
        while True:
            # pos is the position where the current property starts, and
            # nxt is the position where the next property starts (everything
            # in between those two belongs to this particular property).
            # add one so that we get the byte that is after the newline.
            nxt = mem.find(b'\n', pos) + 1

            # if we cannot find the end of the property (file wrap-around),
            # then just claim the file as exhausted
            if nxt == 0:
                nxt = mem.size()
                break

            # if this property is the end marker, then stop processing
            if mem.find(_GEN_END, pos, nxt) != -1:
                break
            else:
                # two first three bytes are comment hyphens ("-- "), and the
                # last two bytes are the newline ("\r\n"). read the tokens
                # that are inside these brackets
                fst = pos + 3
                lst = nxt - 2

                # find the colon if this is a named property, and use that
                # colon to parse the line into a key and a value. if there
                # is no value, then interpret this as a kind of directive
                delim = mem.find(b':', fst, lst)
                if delim != -1:
                    key = dec(mem[fst: delim]).rstrip()
                    value = dec(mem[delim + 2: lst])
                else:
                    key = dec(mem[fst: lst])
                    value = None

                # assign the values found to the output set
                props[key] = value

                # move to processing the next line
                pos = nxt

    # return the properties that were found, or empty dictionary if not
    # a generated file
    return (props, nxt)


# known sections that can be in a datafile, and whether they have any
# associated data with them (should be read until a final slash or not)
# result refers to whether this section should be read as part of the
# final result or not. cell refers to whether this is a (regular) cell
# property (with format nk*nj*ni)
_SECTIONS = {
    'PINCH':      {'slashed': True,  'result': False, 'cell': False, },
    'ECHO':       {'slashed': False, 'result': False, 'cell': False, },
    'MAPUNITS':   {'slashed': True,  'result': False, 'cell': False, },
    'MAPAXES':    {'slashed': True,  'result': False, 'cell': False, },
    'GRIDUNIT':   {'slashed': True,  'result': False, 'cell': False, },
    'SPECGRID':   {'slashed': True,  'result': False, 'cell': False, },
    'COORDSYS':   {'slashed': True,  'result': False, 'cell': False, },
    'COORD':      {'slashed': True,  'result': False, 'cell': False, },
    'ZCORN':      {'slashed': True,  'result': False, 'cell': False, },
    'ACTNUM':     {'slashed': True,  'result': False, 'cell': False, },
    'PORO':       {'slashed': True,  'result': True,  'cell': True, },
    'PERMX':      {'slashed': True,  'result': True,  'cell': True, },
    'PERMY':      {'slashed': True,  'result': True,  'cell': True, },
    'PERMZ':      {'slashed': True,  'result': True,  'cell': True, },
    'TRANSX':     {'slashed': True,  'result': True,  'cell': True, },
    'TRANSY':     {'slashed': True,  'result': True,  'cell': True, },
    'NOECHO':     {'slashed': False, 'result': False, 'cell': False, },
    'GRID':       {'slashed': False, 'result': False, 'cell': False, },
    'INCLUDE':    {'slashed': True,  'result': False, 'cell': False, },
    'NTG':        {'slashed': True,  'result': True,  'cell': True, },
    'SWATINIT':   {'slashed': True,  'result': True,  'cell': True, },
    'SWCR':       {'slashed': True,  'result': True,  'cell': True, },
    'EQLNUM':     {'slashed': True,  'result': True,  'cell': True, },
    'FIPNUM':     {'slashed': True,  'result': True,  'cell': True, },
    # for properties that are not used by Eclipse, Petrel doesn't care
    # about the eight character limit, but just writes the full name
    'BULKVOLUME':          {'slashed': True,  'result': True,  'cell': True, },
    'ELEVATIONGENERAL':    {'slashed': True,  'result': True,  'cell': True, },
    'FACIES':              {'slashed': True,  'result': True,  'cell': True,
                            'dtype':   numpy.int32},
}


def _kw_dtype(kw):
    """Data type of a keyword, if specified in the section of known keywords"""
    # thanks to boolean short-circuiting, we can test for the presence of the
    # keyword in the section list and whether it has the dtype property in the
    # same expression. if it isn't there, then the default are floating point
    if kw in _SECTIONS and 'dtype' in _SECTIONS[kw]:
        return _SECTIONS[kw]['dtype']
    else:
        return numpy.float64


def _fast_index(path, mem, skip=0):
    # lookup table that will be returned
    index = {}

    # don't associate keywords in the main file with a particular file,
    # since we already have the memory-mapping open, but extract the
    # directory of the path so we can read included files from there
    base_dir = os.path.normpath(os.path.dirname(path))
    _fast_index_mem(base_dir, None, mem, skip, index)

    # return the lookup table to caller after the file is processed
    return index


def _fast_index_file(base_dir, fname, index):
    # full path and name of the file to be included
    fullfile = os.path.normpath(os.path.join(base_dir, fname))

    # it might be that the included file is in a sub-dir, in which case
    # further files included should be read from that directory too
    new_dir = os.path.dirname(fullfile)

    # open a new memory buffer for this file, and index it into the existing
    # section list, associating keywords with this file
    with ctx.closing(_FileMapPair(fullfile)) as fmp:
        _fast_index_mem(new_dir, fullfile, fmp.map_obj, 0, index)


def _fast_index_mem(base_dir, fname, mem, skip, index):
    """
    Parse and index the memory-mapped file.

    Parameters
    ----------
    base_dir : str
        Directory from which included files will be read.
    fname : str
        Filename associated with the memory-mapping.
    mem : mmap.mmap
        Memory-mapping of the file that should be indexed.
    skip : int
        Number of bytes that should be skipped at the beginning of the file.
    index : dict
        Lookup table of the recognized keywords in the file and their position.
        The position is a pair of the address of the first data byte (which may be a whitespace),
        and of the ending slash, which can be seen as the end-point of the data, exclusive.
        If the last item in the tuple is not None, then it is in another file than the main file.

    Returns
    -------
    None
    """
    # build an index of keywords found in the file
    cur_pos = skip

    # read the entire file to end
    while cur_pos < mem.size():

        # skip lines that are comments or empty lines
        if (mem.find(b'-', cur_pos, cur_pos + 1) == cur_pos or
                mem.find(b'\r', cur_pos, cur_pos + 1) == cur_pos):
            cur_pos = mem.find(b'\n', cur_pos) + 1
            continue

        # next bytes are a keyword; all keywords are written with a padding
        # up to a comments that says that this keyword is written by Petrel
        space_pos = mem.find(b' ', cur_pos)

        # read the keyword itself; this identifies the section
        kw = dec(mem[cur_pos: space_pos])

        # skip the rest of this line
        cur_pos = mem.find(b'\n', cur_pos) + 1

        # if the keyword has associated data, then figure out where in the file
        # that section is
        if _SECTIONS[kw]['slashed']:

            # sometimes the keyword has another name in Petrel, but an entry
            # is generated in the .grdecl based on its type. the name in
            # Petrel is then added in a comment right below the keyword. here
            # we simply skip those comments
            if (mem.find(b'-', cur_pos, cur_pos + 1) == cur_pos and
                    mem.find(b'-', cur_pos + 1, cur_pos + 2) == cur_pos + 1):
                cur_pos = mem.find(b'\n', cur_pos) + 1

            # include statements is a special-case: the associated data
            # is not in this file
            if kw == 'INCLUDE':
                # filename is always in single quotes at the first data line
                start_pos = cur_pos + 1
                end_pos = mem.find(b'\'', start_pos)
                fname = dec(mem[start_pos: end_pos])
                log.info('Indexing file: "%s"', fname)

                # index the included file recursively. we don't keep that
                # file open, because it just contains a single keyword which
                # is only read at most one time later
                _fast_index_file(base_dir, fname, index)

                # advance past the slash that ends the inclusion statement
                cur_pos = mem.find(b'/', end_pos) + 1

            else:
                # range of the data that is associated with the keyword
                start_pos = cur_pos
                end_pos = mem.find(b'/', cur_pos)
                cur_pos = end_pos + 1

        # no associated data, simply return a blank tuple
        else:
            start_pos = 0
            end_pos = 0

        # make an entry of this keyword in the lookup table
        if kw != 'INCLUDE':
            index[kw] = (start_pos, end_pos, fname)


# definition of special characters that can be compared directly to the
# contents of the memory-map. notice that this is the inverse of the
# six.byte2int function that is used further below when manipulating a
# bytearray copy.
if sys.version_info[0] < 3:
    _SP = b' '
    _NL = b'\n'
else:
    _SP = ord(b' ')
    _NL = ord(b'\n')


def _tokenize(mem, bgn, end):
    """
    Read tokens from the data part of a section.

    Parameters
    ----------
    mem : mmap.mmap
        Memory-mapping of the file the section should be read from.
    bgn : int
        Offset of the first byte that is part of the section.
    end : int
        Offset of the first next byte that is *not* part of the section (end range, exclusive).

    Returns
    -------
    generator
        Generator yielding tuples of (int, int) representing ranges for each token.
    """
    # current position we are looking at
    cur = bgn

    # tokenize while there is any tokens left in the stream
    while cur < end:

        # skip all leading whitespace
        while (cur < end) and ((mem[cur] == _SP) or (mem[cur] == _NL)):
            cur = cur + 1

        # the token begin at the first non-whitespace character
        fst = cur

        while (cur < end) and ((mem[cur] != _SP) and (mem[cur] != _NL)):
            cur = cur + 1

        # token ends at the next whitespace encountered
        lst = cur

        # generate this token, if it is non-empty
        if fst < lst:
            yield (fst, lst)


def _decode(mem, fst, lst, typ):
    """
    Decode a token into a value.

    Parameters
    ----------
    mem : mmap.mmap
        Memory-mapping of the file we are reading from.
    fst : int
        Offset into the file of the start of the token.
    lst : int
        Offset into the file of one past the last byte of the token.
    typ : str -> Any
        Conversion routine for the value.

    Returns
    -------
    tuple of (int, Any)
        The value that is decoded, and the repeat count of that value.
    """
    # is a repeat count specified?
    asterisk = mem.find(b'*', fst, lst)

    # no repeat count, just one single value
    if asterisk == -1:
        count = 1
        value = typ(mem[fst: lst])

    # repeat count specified, must decode separately
    else:
        count = int(mem[fst: asterisk])
        value = typ(mem[asterisk+1: lst])

    return (count, value)


def _read_array(mem, bgn, end, dims, typ):
    """
    Read a section as a typed matrix.

    Parameters
    ----------
    mem : mmap.mmap
        Memory-mapping of the file the section should be read from.
    bgn : int
        Offset of the first byte that is part of the section.
    end : int
        Offset of the first next byte that is *not* part of the section (end range, exclusive).
    dims : list of int
        Tuple consisting of final dimensions for the array.
    typ : str -> Any
        Conversion routine for the value.

    Returns
    -------
    numpy.ndarray
        Array of values read from the file with shape `dims` and dtype `typ`.
    """
    # allocate the memory of the array first
    total = int(numpy.product(dims))
    data = numpy.empty((total, ), dtype=typ)

    # index that we are currently going to assign the next token to
    ofs = 0

    # process all tokens found in the stream
    for fst, lst in _tokenize(mem, bgn, end):
        # look for asterisk and get the repeat count
        count, value = _decode(mem, fst, lst, typ)

        # assign a number of items in batch using the repeat count
        data[ofs: ofs+count] = value

        # increment the counter to where the next token should start
        ofs = ofs + count

    # if the latter part of the array is zero, then Petrel won't bother
    # to write it
    data[ofs: total] = 0

    # reshape into proper grid before returning
    return numpy.reshape(data, dims)


# pylint: disable=too-many-arguments
def _read_section(sec_name, dims, typ, mem, sec_tbl):
    """
    Read a section from file, if it is present.

    Parameters
    ----------
    sec_name : str
        Name of the section to be read.
    dims : list of int
        Tuple consisting of final dimensions for the array.
    typ : str -> Any
        Conversion routine for the value.
    mem : mmap.mmap
        Memory-mapping of the file the section should be read from.
    sec_tbl : dict of str to tuple of (int, int)
        Lookup table for each section in the file.

    Returns
    -------
    numpy.ndarray
        Array of values read from the file with shape `dims` and dtype `typ`.
    """
    log.info("Reading keyword %s", sec_name)

    # get the address of the subsection in memory
    bgn, end, fname = sec_tbl[sec_name]

    # if it is from another file, and not from the memory-map of the main
    # file, then it must be opened and data read from there. however, we
    # don't keep it open, because it probably only contain a single property
    # and will never be read again.
    if fname is not None:
        with ctx.closing(_FileMapPair(fname)) as fmp:
            data = _read_array(fmp.map_obj, bgn, end, dims, typ)
    else:
        data = _read_array(mem, bgn, end, dims, typ)
    return data


def _sec_mat(mem, sec_tbl, sec_name, dtype, usecols):
    """
    Read a data section matrix for a keyword.

    Parameters
    ----------
    mem : mmap.mmap
        Memory-mapping of the file the section should be read from.
    sec_tbl : dict of str to tuple of (int, int)
        Lookup table for each section in the file.
    sec_name : str
        Name of the section to be read.
    dtype : numpy.dtype
        Expected NumPy data type of the values in the section.
    usecols : list of int or None
        Columns that should be read. Use None to read a freely formatted section.

    Returns
    -------
    numpy.ndarray
        Array of values read from the file with shape inferred from the section and dtype `dtype`.
    """
    # get the address of the section within the file
    bgn, end, fname = sec_tbl[sec_name]

    # if it is from another file, and not from the memory-map of the main
    # file, then it must be opened and data read from there. however, we
    # don't keep it open, because it probably only contain a single property
    # and will never be read again.
    if fname is not None:
        with ctx.closing(_FileMapPair(fname)) as fmp:
            data = _sec_mat_mem(fmp.map_obj, bgn, end, dtype, usecols)
    else:
        data = _sec_mat_mem(mem, bgn, end, dtype, usecols)

    return data


def _sec_mat_mem(mem, bgn, end, dtype, usecols):
    # create view of the section that just includes the numbers
    buf = bytearray(membuf(mem, bgn, end))

    # if free format stream, we must copy it into memory and remove all the
    # newline, since Petrel is obviously not able to keep a consistent number
    # of column on each line.
    if usecols is None:
        _strip_newline(buf)

    # let the library do the heavy lifting of this section; it is just an
    # array without any special formatting (anymore)
    with ctx.closing(six.BytesIO(buf)) as src:
        data = numpy.loadtxt(src, dtype=dtype, usecols=usecols)

    return data


def _read_specgrid(mem, sec_tbl):
    """
    Read the grid dimensions.

    Parameters
    ----------
    mem : mmap.mmap
        Memory-mapping of the file the section should be read from.
    sec_tbl : dict of str to tuple of (int, int)
        Lookup table for each section in the file.

    Returns
    -------
    tuple of int
        The dimensions of the grid.
    """
    # read the first three numbers from the section
    log.info("Reading keyword SPECGRID")
    spec = _sec_mat(mem, sec_tbl, 'SPECGRID', numpy.int32, (0, 1, 2))

    # reverse the dimensions (since the i axis varies fastest), before
    # assigning it to the target dictionary
    return spec[::-1]


_CR = six.byte2int(b'\r')
_LF = six.byte2int(b'\n')
_WS = six.byte2int(b' ')


def _strip_newline(data):
    """Replace newlines with regular whitespace. This allows us to read an
    array with a known total number of arrays, but where we don't know how
    many column are formatted on each line.
    """
    # according to Guido van Rossum, in-place bytearray operations are "rare",
    # so it is OK for the bytearray to reuse string methods that return a copy.
    # hopefully, this implementation is straight-forward enough so that it can
    # be optimized into something that use less time than having to allocate a
    # string copy for every line.
    for i in range(len(data)):
        if (data[i] == _CR) or (data[i] == _LF):
            data[i] = _WS


def _axes_map(mem, section_index):
    """
    Rotation matrix and offset vector for axes mapping.

    Returns
    -------
    tuple of (ndarray, ndarray)
        A tuple (A, b) where A is the rotation matrix and b is the offset vector.
        Each point x in the coordinate array should be rotated with A(x - b) + b.
    """
    # if a mapping is specified in the file, then use this, otherwise
    # return a unit mapping, so that we can proceed along the same code path
    if 'MAPAXES' in section_index:
        vec = _read_section('MAPAXES', (3, 2),
                            numpy.float64, mem, section_index)
        A, b = _axes_rot(vec[0, 0], vec[0, 1],
                         vec[1, 0], vec[1, 1],
                         vec[2, 0], vec[2, 1])
    else:
        A, b = _no_rot()

    return (A, b)


def _read_coord(dims, mem, sec_tbl, A, b):
    """
    Read pillar coordinates.

    Parameters
    ----------
    mem : mmap.mmap
        Memory-mapping of the file the section should be read from.
    sec_tbl : dict of str to tuple of (int, int)
        Lookup table for each section in the file.
    A : ndarray
        Rotation matrix.
    b : ndarray
        Rotation offset.
    """
    # extract dimensions of the table into more sensible local names
    num_j = dims[1]
    num_i = dims[2]

    # pillar coordinates don't have any repeated values, so they are always
    # written with 'un-starred' floats.
    log.info("Reading keyword COORD")
    coord = _sec_mat(mem, sec_tbl, 'COORD', numpy.float64, None)

    # reformat the section into the correct format
    coord = numpy.reshape(coord, (num_j + 1, num_i + 1, 2, 3))

    # rotate the coordinate pillars
    coord -= b
    numpy.dot(coord, A)
    coord += b

    # pillar coordinates can now be assigned into output structure
    return coord


def _zcorn_dims(dims):
    """Dimensions for the ZCORN array, given grid dimensions"""
    return (dims[0], 2, dims[1], 2, dims[2], 2)


# include statements in a wrapper file written by us look like this:
_INCL_STMT = re.compile(enc(r'INCLUDE\ *\n\'(.*)\'\n/.*\n'), re.MULTILINE)


def read(filename):
    """Read an Eclipse input grid into a dictionary.
    """
    # get the canonical path of the file to read
    path = os.path.expanduser(filename)

    # open the file and create a memory-mapping so that we can read
    # the input file as if it was a string buffer
    with ctx.closing(_FileMapPair(path)) as fmp:
        mem = fmp.map_obj

        # determine first if there are a header section at the top
        # of the file. if there is, the start reading after it
        props, skip = _parse_meta(mem)
        mem.seek(skip, os.SEEK_SET)

        # notice that there is an optimization opportunity since
        # Petrel only uses a subset of the format
        if ('Exported by' in props and
                props['Exported by'].startswith('PyReSiTo (multi) v1.0')):
            log.info("File is PyReSiTo multi-file")

            grid = _read_multi(path, mem)

        elif ('Exported by' in props and
              props['Exported by'].startswith('Petrel')):
            log.info("File is exported by Petrel")
            grid = {}
            section_index = _fast_index(path, mem, skip)

            # specgrid is special because it must be read before everything
            # else, so that we know how to read other sections
            dims = _read_specgrid(mem, section_index)
            grid['DIMENS'] = dims[::-1]

            # coord is special because we need to possibly rotate it
            A, b = _axes_map(mem, section_index)
            grid['COORD'] = _read_coord(dims, mem, section_index, A, b)

            # actnum is special because we need it to set the mask
            grid['ACTNUM'] = _read_section('ACTNUM', dims, numpy.int32,
                                           mem, section_index).astype(
                numpy.bool)
            mask = numpy.logical_not(grid['ACTNUM'])

            # zcorn is special because it has a different format (and
            # cannot have inactive elements)
            grid['ZCORN'] = _read_section('ZCORN', _zcorn_dims(dims),
                                          numpy.float64, mem,
                                          section_index)

            # read general cell properties that are marked as result
            # properties, from the file
            for kw in section_index:
                props = _SECTIONS[kw]
                if props['result']:
                    # read the data that is stored in the file
                    data = _read_section(kw, dims, _kw_dtype(kw), mem,
                                         section_index)

                    # create an array where the inactive region is masked
                    # out (won't appear on plots, in statistics etc.)
                    grid[kw] = numpy.ma.array(data=data, mask=mask)

        # use the regular, but slower parser
        else:
            with ctx.closing(_Parser(path, mem)) as parser:
                grid = parser.parse()

    return grid


def _read_multi(wrapper_name, mem):
    """
    Parameters
    ----------
    wrapper_name : str
        Name of the file containing the inclusion wrapper. This file is only 
        interesting because the name of the dimensions file is constructed based on it.
    mem : mmap.mmap
        Handle to memory-mapping of the wrapper file.
    """
    log.info("Reading wrapper for include statements")
    # parse the entire buffer for inclusion clauses
    buf = membuf(mem, 0, mem.size())
    files = [dec(match.group(1))
             for match in re.finditer(_INCL_STMT, buf)]
    buf = None

    # strip off the extension of the filename. this stem is then
    # recombined to get the name of the dimension file, which is
    # not included by the wrapper but is part of the exported files
    stem, _ = os.path.splitext(wrapper_name)

    dim_name = '{0}_dimens.grdecl'.format(stem)
    log.info("Reading grid dimensions from \"%s\"", dim_name)
    ni, nj, nk = [int(x) for x in numpy.loadtxt(
        dim_name, dtype=numpy.int32, skiprows=1, comments='/')]
    log.info("Grid has dimensions %d x %d x %d", ni, nj, nk)

    # shape of the data properties; notice that the order of the dimensions
    # is reversed (in Python) compared to the way they are stored in the file
    dims = [nk, nj, ni]

    # read the active flag from file
    act_name = '{0}_actnum.grdecl'.format(stem)
    log.info("Reading cell activeness from \"%s\"", act_name)
    actnum = numpy.reshape(numpy.loadtxt(
        act_name, dtype=numpy.bool, skiprows=1, comments='/'),
        (nk, nj, ni))

    coord_name = '{0}_coord.grdecl'.format(stem)
    log.info("Reading pillar coordinates from \"%s\"", coord_name)
    coord = numpy.reshape(numpy.loadtxt(
        coord_name, dtype=numpy.float, skiprows=1, comments='/'),
        (nj + 1, ni + 1, 2, 3))

    zcorn_name = '{0}_zcorn.grdecl'.format(stem)
    log.info("Reading corner depths from \"%s\"", zcorn_name)
    zcorn = numpy.reshape(numpy.loadtxt(
        zcorn_name, dtype=numpy.float, skiprows=1, comments='/'),
        (nk, 2, nj, 2, ni, 2))

    # create the embryonic grid containing at least the format
    grid = {'DIMENS': [ni, nj, nk],
            'COORD':  coord,
            'ZCORN':  zcorn,
            'ACTNUM': actnum,
            }

    # remove the known properties from the list that is already loaded
    for prop in ['actnum', 'coord', 'zcorn']:
        filename = '{0}_{1}.grdecl'.format(os.path.basename(stem), prop)
        if filename in files:
            files.remove(filename)

    # read each extra property from its own file
    stem_dir = os.path.dirname(stem)
    for filename in files:
        _read_multi_prop(os.path.join(stem_dir, filename), dims, grid)

    return grid


def _read_multi_prop(filename, dims, grid):
    """
    Read a property from a file and store it in the grid structure.

    Parameters
    ----------
    filename : str
        Name of the file containing the property.
    dims : list of int
        Dimensions of the grid.
    grid : dict
        Structure that will receive the property.

    Returns
    -------
    None
    """
    with open(filename, 'rb') as fileobj:
        # read the name of the property from the first line of the file
        propname = codecs.ascii_decode(fileobj.readline())[0][:-1].strip()
        log.info("Reading keyword %s from \"%s\"", propname, filename)

        # rest of the file contains the property
        data = numpy.loadtxt(fileobj, comments='/',
                             dtype=_kw_dtype(propname))

    # reformat the property to contain the right shape
    grid[propname] = numpy.reshape(data, dims)


# rules on how to reformat some keywords; the rule is expressed as a function
# taking the actual data and then returning a tuple which is the new shape that
# can be passed to the reshape function. if a keyword is not in this list, then
# it will be written with a single entry on each line.
_SPECIAL_SHAPE = {
    'COORD': lambda ni, nj, nk: ((nj + 1) * (ni + 1), (2 * 3)),
}


# formats for keywords that have known value domains
_SPECIAL_FORMAT = {
    'COORD':   '%8.2f',
    'ZCORN':   '%7.2f',
    'ACTNUM':  '%d',
}


def _write_kw(fileobj, propname, values, lookup, dimens):
    """
    Write a section for a single keyword to an already open file.

    Parameters
    ----------
    fileobj : file-like object
        Where to serialize the data.
    propname : str
        Name of the property from the grid to write.
    values : numpy.ndarray
        Array with the values themselves.
    lookup : dict
        Mapping of properties to identifier string used in file.
    dimens : tuple of int
        Dimensions of the grid, (ni, nj, nk).

    Returns
    -------
    None
    """
    # structure of the values: if we can make a fixed number of columns,
    # then we attempt to do so, otherwise we just write everything as a
    # long stream (faster to load, when there are no exceptions)
    if propname in _SPECIAL_SHAPE:
        shape = _SPECIAL_SHAPE[propname](*dimens)
    else:
        shape = (numpy.prod(values.shape), 1)

    # make sure we get an appropriate number of digits on the keywords that
    # are known to us
    if propname in _SPECIAL_FORMAT:
        fmt = _SPECIAL_FORMAT[propname]
    else:
        # print integral types without inappropriate fraction digits
        if numpy.issubdtype(values.dtype, numpy.integer):
            fmt = '%d'
        else:
            fmt = '%11.6f'

    # write the keyword itself, on its own line
    fileobj.write(enc('{0:<8s}\n'.format(lookup(propname))))

    # replace NaN values (which we typically use to trap access to inactive
    # blocks) with zero (which Petrel uses a placeholder for inactive blocks)
    zero = 0 if numpy.issubdtype(values.dtype, numpy.integer) else 0.
    data = numpy.reshape(numpy.copy(numpy.ma.getdata(values)), shape)
    data[numpy.isnan(data)] = zero

    # data array itself; NumPy does the job for us
    numpy.savetxt(fileobj, data, fmt=fmt, delimiter='\t')

    # end of field for most fields in the loaded grid file
    fileobj.write(enc('{0:s}\n'.format(lookup('/'))))


def _write_meta(fileobj, subtype):
    """Write a header which enable us to recognize our own files later."""
    fileobj.write(_GEN_BGN)
    fileobj.write(enc(
        '-- Exported by: PyReSiTo ({0}) v1.0\r\n'.format(subtype)))
    fileobj.write(_GEN_END)


def _write_dimens_ecl(fileobj, dimens):
    """Write grid dimensions to an Eclipse wrapper file."""
    fileobj.write(enc('SPECGRID\n{0:d} {1:d} {2:d} 1 F\n/\n'.format(
        *dimens)))


def _write_dimens_ext(fileobj, dimens, lookup):
    """Write grid dimensions to a new external grid file."""
    # decode grid dimensions since we cannot expand inline in formatting tuple
    ni, nj, nk = dimens

    # write the keyword containg the
    fileobj.write(enc('{0:<8s}\n{1:d} {2:d} {3:d}\n{4:s}\n'.format(
        lookup('DIMENS'), ni, nj, nk, lookup('/'))))


# these keywords are written in the beginning of the file in this order,
# and then the rest of any properties we have stuffed into the grid object
# follows after.
_LEADING = ['COORD', 'ZCORN', 'ACTNUM']


def _write_multi(path, base, grid, lookup, dialect):
    """
    Generate files for the grid properties.

    Parameters
    ----------
    path : str
        Path to all files that are generated.
    base : str
        Prefix for all the files that are generated.
    grid : dict
        Dictionary of properties.
    lookup : function
        Lookup function for keyword strings.
    dialect : str
        Simulator that the file is intended for.

    Returns
    -------
    None
    """
    log.info('Writing multiple files')
    stem = os.path.join(path, base)

    # the size is written into a separate file, so it can be loaded from there
    dimens = grid['DIMENS']
    log.info('Grid size is %d x %d x %d', *dimens)
    log.info('Writing keyword DIMENS')
    with open('{0}_dimens.grdecl'.format(stem), 'wb') as fileobj:
        _write_dimens_ext(fileobj, dimens, lookup)

    # size is also written directly into the wrapper file, and not included
    with open('{0}.grdecl'.format(stem), 'wb') as wrapper:
        # header first
        log.info('Writing wrapper heading')
        _write_meta(wrapper, 'multi')

        # internal dimensioning is written specially with other keywords
        log.info('Writing size of grid')
        if dialect == 'ecl':
            _write_dimens_ecl(wrapper, dimens)
        else:
            assert False

        # start with all the predefined leading keywords, and then write the
        # rest of them in alphabetical order
        order = ([x for x in grid if x in _LEADING] +
                 sorted([x for x in grid if x not in _LEADING + ['DIMENS']]))

        # all other keywords than the size is written to each their own file;
        # use bytes formatted files since NumPy requires this to dump matrix
        for kw in order:
            log.info('Writing keyword %s', kw.upper())
            filename = '{0}_{1}.grdecl'.format(stem, kw.lower())
            with open(filename, 'wb') as fileobj:
                _write_kw(fileobj, kw, grid[kw], lookup, dimens)

            # write an include statement in the wrapper file; it is customary
            # to write only a relative filename and not a full path
            wrapper.write(
                enc('\n{0:<8s}\n\'{1:s}_{2:s}.grdecl\'\n{3:s}\n'.format(
                    lookup('INCLUDE'), base, kw.lower(), lookup('/'))))


def _write_single(path, base, grid, lookup, dialect):
    """
    Generate files for the grid properties.

    Parameters
    ----------
    path : str
        Path to all files that are generated.
    base : str
        Prefix for all the files that are generated.
    grid : dict
        Dictionary of properties.
    lookup : function
        Lookup function for keyword strings.
    dialect : str
        Simulator that the file is intended for.

    Returns
    -------
    None
    """
    log.info('Writing Single files')
    stem = os.path.join(path, base)
    dimens = grid['DIMENS']

    # size is also written directly into the file,
    with open('{0}.grdecl'.format(stem), 'wb') as fileobj:
        # header first
        log.info('Writing wrapper heading')
        _write_meta(fileobj, 'single')

        # internal dimensioning is written specially with other keywords
        log.info('Writing size of grid')
        if dialect == 'ecl':
            _write_dimens_ecl(fileobj, dimens)
        else:
            assert False
        #
        # log.info('Grid size is %d x %d x %d', *dimens)
        # log.info('Writing keyword DIMENS')
        # _write_dimens_ext(fileobj, dimens, lookup)

        # start with all the predefined leading keywords, and then write the
        # rest of them in alphabetical order
        order = ([x for x in grid if x in _LEADING] +
                 sorted([x for x in grid if x not in _LEADING + ['DIMENS']]))

        # all other keywords than the size is written to the file;
        # use bytes formatted files since NumPy requires this to dump matrix
        for kw in order:
            log.info('Writing keyword %s', kw.upper())
            _write_kw(fileobj, kw, grid[kw], lookup, dimens)


# a dialect contain the strings that are written for each keyword in the case
# that they are not the same as the property name. the end-of-field marker has
# been coded as a keyword with the symbol '/', as in Eclipse.
_DIALECT = {
    'ecl': {},
    'cmg': {
        'DIMENS':   'CORNER',
        'ACTNUM':   'NULL',
        'PORO':     'POR',
        'PERMX':    'PERMI',
        'PERMY':    'PERMJ',
        'PERMZ':    'PERMK',
        'MULTX':    'TRANSI',
        'MULTY':    'TRANSJ',
        'FIPNUM':   'ISECTOR',
        'NTG':      'NETGROSS',
        'EQLNUM':   'ITYPE',
        'SWATINIT': 'SWINIT ALL',
        '/':        ''
    }
}


def write(filename, grid, dialect='ecl', multi_file=True):
    """
    Write a grid to corner-point text format, formatted in such a way
    that it can easily be read again.

    Parameters
    ----------
    filename : str
        Name of the filename containing a wrapper for all the properties.
        This will also serve as the stem for all individual property files.
    grid : dict
        Grid object containing values for all properties.
    dialect : str
        Name of the simulator that we are writing files for. Currently, this must be 'ecl'.
    multi_file : bool
        Write properties in separate files, instead of putting everything in one large file.

    Returns
    -------
    None
    """
    # inner helper function that curries the dialect, and returns the keyword
    # from the translation dictionary if present, otherwise return unmodified.
    # this function is passed to all the other, and is used to lookup the
    # correct string to write for each keyword.
    trans_dict = _DIALECT[dialect]

    def _lookup(kw):
        return trans_dict[kw] if kw in trans_dict else kw

    # dismantle the filename into various parts; we are going to use the
    # relative filenames in the wrapper file
    path, base = os.path.split(filename)
    base, _ = os.path.splitext(base)

    # delegate to specialized routine, that directs writing of individual
    # properties to either each their file, or one common file
    if multi_file:
        _write_multi(path, base, grid, _lookup, dialect)
    else:
        _write_single(path, base, grid, _lookup, dialect)


def _stretches(data):
    """
    Identify stretches of data with equal values.

    Parameters
    ----------
    data : numpy.ndarray
        Array which is scanned for stretches of equal values.

    Returns
    -------
    generator
        Generator yielding tuples of (int, Any), where the int is the start index of the stretch and Any is the value of the stretch.
    """
    # picture cursors in between every number; this becomes the array of
    # numbers before and after every such cursor; looking backward and forwards
    before = data[0:-1]
    after = data[1:]

    # get a list of indices for the positions that change; add one since the
    # indices will be relative to the 'after' array, which is skewed one from
    # the original. we must add zero to this list, as the first element is
    # always new, but is never registered as such (nothing to differ from)
    changes = numpy.where(numpy.not_equal(before, after))[0] + 1
    changes = numpy.concatenate((numpy.array((0,), dtype=changes.dtype),
                                 changes))

    # changes is the start of each run; find the lengths of each run. here
    # we have the dual of the case above with the first item; the last item
    # runs until the end of the array, but there is no change after it so
    # we must register it manually
    length = numpy.diff(
        numpy.concatenate((changes, numpy.array((len(data),),
                                                dtype=changes.dtype))))

    # return a generator of every individual value and its associated
    # consecutive count in the stream
    for row, count in enumerate(length):
        yield (count, data[changes[row]])


def _write_compr_full(f_obj, data, fmt):
    """
    Write a data field as a keyword that can be included into an already
    opened grid file, assuming a full data array.

    Parameters
    ----------
    f_obj : io.IOBase
        File-like object to write to.
    data : numpy.ndarray
        Data array to be written.
    fmt : str
        Format of a single item, on the form used to specify formats in the
        built-in routines, but without percent or braces.

    Returns
    -------
    None
    """
    # format strings for writing a single value and multiple values,
    # respectively; we prepare these outside of the loop. as a special
    # case, we write nulls in a particularily short format
    single_fmt = '{{0:{0:s}}}\n'.format(fmt)
    multi_fmt = '{{0:d}}*{{1:{0:s}}}\n'.format(fmt)
    single_null = '0\n' if fmt[-1] == 'd' else '0.\n'
    multi_null = '{0:d}*0\n' if fmt[-1] == 'd' else '{0:d}*0.\n'

    # then write all the data in batch; length first, then the data,
    # using the data itself for criteria for what constitute a stretch
    for count, value in _stretches(data):
        if value:
            if count == 1:
                f_obj.write(enc(single_fmt.format(value)))
            else:
                f_obj.write(enc(multi_fmt.format(count, value)))
        else:
            if count == 1:
                f_obj.write(enc(single_null))
            else:
                f_obj.write(enc(multi_null.format(count)))


# pylint: disable=too-many-branches
def _write_compr_masked(f_obj, data, mask, fmt):
    """
    Write a data field as a keyword that can be included into an already
    opened grid file, assuming a masked (sparse) data array.

    Parameters
    ----------
    f_obj : io.IOBase
        File-like object to write to.
    data : numpy.ndarray
        Data array to be written.
    fmt : str
        Format of a single item, in the form used to specify formats in the
        built-in routines, but without percent or braces.

    Returns
    -------
    None
    """
    # format strings for writing a single value and multiple values. these
    # are the same as for the _write_compr_full routine, but we cannot make
    # them global variables since they depend on the format
    single_fmt = '{{0:{0:s}}}\n'.format(fmt)
    multi_fmt = '{{0:d}}*{{1:{0:s}}}\n'.format(fmt)
    single_null = '0\n' if fmt[-1] == 'd' else '0.\n'
    multi_null = '{0:d}*0\n' if fmt[-1] == 'd' else '{0:d}*0.\n'

    # Petrel reads these as masked values
    single_blank = 'NaN\n'
    multi_blank = '{0:d}*NaN\n'

    # we haven't written any values yet, so start reading from the beginning
    accum = 0

    # start out by enumerating stretches which is either masked, or unmasked
    # pylint: disable=too-many-nested-blocks
    for count_subset, is_blank in _stretches(mask):

        # if we are to write a stretch of blank values, then we don't need
        # any formatting for the number, just the special mask value
        if is_blank:
            if count_subset == 1:
                f_obj.write(enc(single_blank))
            else:
                f_obj.write(enc(multi_blank.format(count_subset)))

        # if we have a subset with non-blanks, then process this subset as
        # if it were an array in itself, which enables us to compress long
        # stretches of equal values there, too
        else:
            # fast path
            if count_subset == 1:
                value = data[accum]
                if value:
                    f_obj.write(enc(single_fmt.format(value)))
                else:
                    f_obj.write(enc(single_null))

            # more than one value in the subset
            else:
                subset = data[accum:(accum+count_subset)]
                for count, value in _stretches(subset):
                    if value:
                        if count == 1:
                            f_obj.write(enc(single_fmt.format(value)))
                        else:
                            f_obj.write(enc(multi_fmt.format(count, value)))
                    else:
                        if count == 1:
                            f_obj.write(enc(single_null))
                        else:
                            f_obj.write(enc(multi_null.format(count)))

        # keep track of how many items we have written totally
        accum += count_subset


def _write_compr_any(f_obj, keyw, cube, fmt):
    """
    Write a data field as a keyword that can be included into an already
    opened grid file.

    Parameters
    ----------
    f_obj : io.IOBase
        File-like object to write to.
    keyw : str
        Name of the keyword to put in the header; this should not be greater than eight characters.
    cube : numpy.ndarray
        Data cube to be written.
    fmt : str
        Format of a single item, in the form used to specify formats in the built-in routines, but without percent or braces.

    Returns
    -------
    None
    """
    # write the keyword first, on its own line
    f_obj.write(enc('{0:8s}\n'.format(keyw.upper())))

    # if the data is sparse, i.e. not defined over the entire field,
    # then use a different writing algorithm, then if it's full
    if hasattr(cube, 'mask'):
        _write_compr_masked(f_obj, numpy.ravel(cube.data),
                            numpy.ravel(cube.mask), fmt)
    else:
        _write_compr_full(f_obj, numpy.ravel(cube), fmt)

    # write a single slash at the end to terminate the field
    f_obj.write(enc('/\n'))


def write_compressed(fname, keyw, cube, *, fmt="12.6e"):
    """
    Write a data field as a keyword that can be included into a grid file.

    Parameters
    ----------
    fname : str
        Path to the output file to write to.
    keyw : str
        Name of the keyword to put in the header; this should not be greater than eight characters.
    cube : numpy.ndarray
        Data cube to be written.
    fmt : str
        Format of a single item, in the form used to specify formats in the built-in routines, but without percent or braces.

    Returns
    -------
    None
    """
    # if we specified a string, this is the path to the file, so open it
    # and start writing; otherwise, it is a file-like object that is already
    # opened, so just continue writing directly
    if isinstance(fname, str):
        with open(os.path.expanduser(fname), 'wb') as f_obj:
            _write_compr_any(f_obj, keyw, cube, fmt=fmt)
    else:
        _write_compr_any(fname, keyw, cube, fmt=fmt)


def shape(grdecl):
    """
    Get shape of field data cube.

    Parameters
    ----------
    grdecl : dict
        Corner-point grid structure.

    Returns
    -------
    tuple of int
        Shape of the field data cube as (num_k, num_j, num_i).
    """
    return tuple(reversed(grdecl['DIMENS']))


def read_prop(fname, dims, typ=numpy.float64, prop=None, *, mask=None):
    """
    Read a property from a text file into a dictionary. This is akin to
    Petrel's Import onto selection popup menu choice.

    Parameters
    ----------
    fname : str
        File name to load property from. Currently, this must be
        a file in the filesystem and cannot be a file-like object.
    dims : tuple of int
        Dimensions of the data cube, in a format compatible with
        the return of the shape function, i.e. (nk, nj, ni).
    typ : numpy.dtype
        Data type of the data to be loaded.
    prop : dict, optional
        Dictionary where the keyword will be added. If this is
        None, then a new dictionary will be created.
    mask : numpy.ndarray, optional
        Existing mask for inactive elements; this will be
        combined with the mask inherent in the data, if specified.
        If you have loaded the grid, this should be the ACTNUM
        field.

    Returns
    -------
    dict
        Dictionary where the property being read is added as a key.
    """
    # if output isn't given, then add to brand new dictionary
    if prop is None:
        prop = dict()

    # allocate memory
    num_cells = numpy.product(dims)
    data = numpy.empty((num_cells,), dtype=typ)

    # read raw array of values into memory first
    with ctx.closing(_OwningLexer(os.path.expanduser(fname))) as lex:

        # read the first keyword that appears in the file
        kw = dec(lex.expect(keyword))
        if not lex.at_eof():
            lex.expect(newline)

        # read numbers from file; each read may potentially fill a different
        # length of the array. use an almost identical type of loop for
        # integers and floating-point, but avoid using test inside the loop
        # or polymorphism through function pointer, to ease optimization
        num_read = 0
        if numpy.issubdtype(typ, numpy.integer):
            while num_read < num_cells:
                count, value = lex.expect_multi_card()
                data[num_read:(num_read+count)] = value
                num_read += count
        else:
            while num_read < num_cells:
                count, value = lex.expect_numbers()
                data[num_read:(num_read+count)] = value
                num_read += count

        # verify that we got exactly the number of corners we wanted
        if num_read > num_cells:
            raise GrdEclError(lex.path, lex.pos(),
                              "Expected {0:d} numbers, but got {1:d}".format(
                                  num_cells, num_read))

        # after all the data, there should be a sentinel slash
        lex.expect(endrec)

    # reformat the data into a natural kji/bfr hypercube
    data = numpy.reshape(data, dims)

    # inherit inactive cells by missing values in the file
    blanked = numpy.isnan(data)

    # if there is either specified a mask, or there is an inherit mask, then
    # we need to generate a masked array
    if (mask is not None) or numpy.any(blanked):
        # combine the two masks into one:
        if mask is not None:
            blanked = numpy.logical_or(blanked, mask)

        # update the values to not have anything where there is now a mask
        if numpy.issubdtype(data.dtype, numpy.integer):
            data[blanked] = 0
        else:
            data[blanked] = numpy.nan

        prop[kw] = numpy.ma.array(data=data, dtype=data.dtype,
                                  mask=blanked)

    # no mask at all, seems to be a full matrix
    else:
        prop[kw] = data

    return prop


def main(*args):
    """Read a data file to see if it parses OK."""
    # setup simple logging where we prefix each line with a letter code
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname).1s: %(message).76s")

    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("output", type=str, nargs='?', default=None)
    parser.add_argument("--dialect", choices=['ecl'], default='ecl')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--quiet", action='store_true')
    cmd_args = parser.parse_args(*args)

    # adjust the verbosity of the program
    if cmd_args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    if cmd_args.quiet:
        logging.getLogger(__name__).setLevel(logging.NOTSET)

    # process file
    grid = read(cmd_args.filename)

    # if an output file was specified, then write it
    if cmd_args.output is not None:
        write(cmd_args.output, grid, cmd_args.dialect, True)


# executable library
if __name__ == "__main__":
    main(sys.argv[1:])
