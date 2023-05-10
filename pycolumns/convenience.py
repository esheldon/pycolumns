import numpy as np
from .columns import Columns
from .defaults import DEFAULT_CACHE_MEM, DEFAULT_CHUNKSIZE
from . import util


def from_fits(
    coldir,
    filename,
    ext=1,
    native=False,
    little=True,
    lower=False,
    compression=None,
    chunksize=DEFAULT_CHUNKSIZE,
    cache_mem=DEFAULT_CACHE_MEM,
    verbose=False,
    yes=False,
):
    """
    Create or append to a columns database, reading from the input fits file.

    parameters
    ----------
    coldir: str
        Columns directory
    filename: string
        Name of the file to read
    ext: extension, optional
        The FITS extension to read, numerical or string. default 1
    native: bool, optional
        FITS files are in big endian byte order.
        If native is True, ensure the outpt is in native byte order.
        Default False.
    little: bool, optional
        FITS files are in big endian byte order.
        If little is True, convert to little endian byte order. Default
        True.
    lower: bool, optional
        if set to True, lower-case all names.  Default False.
    compression: list or dict, optional
        Either
            1. A list of names that get default compression
               see defaults.DEFAULT_COMPRESSION
            2. A dict with keys set to columns names, possibly with
               detailed compression settings.
    chunksize: dict, str or number
        The chunksize info for compressed columns.
        See TableSchema.from_array for a full explanation
    cache_mem: str or number
        Cache memory for index creation, default '1g' or one gigabyte.
    verbose: bool, optional
        If set to True, display information
    yes: bool, optional
        If set to True, do not prompt for confirmation when overwriting
        an existing directory
    """
    import fitsio

    if (native and np.little_endian) or little:
        byteswap = True
        if verbose:
            print('byteswapping')
    else:
        byteswap = False

    with fitsio.FITS(filename, lower=lower) as fits:
        hdu = fits[ext]

        one = hdu[0:0+1]

        if byteswap:
            util.byteswap_inplace(one)

        cols = Columns.create(
            coldir,
            yes=yes,
            cache_mem=cache_mem,
            verbose=verbose,
        )
        cols.from_array(
            one,
            compression=compression,
            chunksize=chunksize,
            append=False,
        )

        nrows = hdu.get_nrows()
        rowsize = one.itemsize

        # step size in bytes
        step_bytes = int(cols.cache_mem_gb * 1024**3)
        # step size in rows
        step = step_bytes // rowsize

        nstep = nrows // step
        nleft = nrows % step

        if nleft > 0:
            nstep += 1

        if verbose:
            print('Loading %s rows from file: %s' % (nrows, filename))

        for i in range(nstep):

            start = i * step
            stop = (i + 1) * step

            if stop > nrows:
                # not needed, but use for printouts
                stop = nrows

            if verbose:
                print(f'    {start}:{stop} of {nrows}')

            data = hdu[start:stop]

            if byteswap:
                util.byteswap_inplace(data)

            cols.append(data, verify=False)
            del data

    cols.verify()

    return cols
