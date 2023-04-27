import numpy as np
from ._column import Column as CColumn
from . import util


class Chunks(object):
    """
    A chunked file, with possible compression
    """
    def __init__(
        self,
        filename,
        chunks_filename,
        dtype,
        mode='r',
        compression=None,
        verbose=False,
    ):
        self._filename = filename
        self._chunks_filename = chunks_filename

        self._dtype = np.dtype(dtype)
        self._chunks_dtype = np.dtype(
            [('offset', 'i8'), ('nbytes', 'i8'),
             ('rowstart', 'i8'), ('nrows', 'i8')]
        )

        self._set_compression(compression)
        self._mode = mode
        self._verbose = verbose
        self._open_files()
        # super().__init__(filename, mode, verbose)
        self._set_nrows()

    @property
    def filename(self):
        """
        get the filename
        """
        return self._filename

    @property
    def chunks_filename(self):
        """
        get the filename
        """
        return self._chunks_filename

    @property
    def mode(self):
        """
        get the file open mode
        """
        return self._mode

    @property
    def compression(self):
        """
        get the file open mode
        """
        return self._compression

    @property
    def dtype(self):
        """
        get the numpy dtype
        """
        return self._dtype

    @property
    def chunks_dtype(self):
        """
        get the numpy dtype of the chunks file data
        """
        return self._chunks_dtype

    @property
    def size(self):
        """
        get numbe of rows
        """
        return self._nrows

    @property
    def nrows(self):
        """
        get number of rows
        """
        return self._nrows

    @property
    def nchunks(self):
        """
        get number of chunks
        """
        nchunks = 0
        if self._chunk_data is not None:
            nchunks = self._chunk_data.size

        return nchunks

    @property
    def verbose(self):
        """
        get the filename
        """
        return self._verbose

    def close(self):
        """
        close the data and chunks files
        """
        self._chunks_fobj.close()
        self._fobj.close()

    def append(self, data):
        """
        append data to the file
        """
        self._check_dtype(data)

        if self.compression:
            nbytes = self._append_compressed_data(data)
        else:
            nbytes = self._append_uncompressed_data(data)

        self._update_chunks_after_write(data.size, nbytes)

    def _append_uncompressed_data(self, data):
        # seek to end
        self._fobj.seek(0, 2)
        data.tofile(self._fobj)
        nbytes = data.nbytes
        return nbytes

    def _append_compressed_data(self, data):
        import blosc

        # seek to end
        self._fobj.seek(0, 2)

        if data.flags['C_CONTIGUOUS'] or data.flags['F_CONTIGUOUS']:
            # this saves memory
            compressed_bytes = blosc.compress_ptr(
                data.__array_interface__['data'][0],
                data.size,
                data.dtype.itemsize,
                **self.compression
            )
        else:
            dbytes = data.tobytes()
            compressed_bytes = blosc.compress(
                dbytes,
                data.dtype.itemsize,
                **self.compression
            )

        self._fobj.write(compressed_bytes)
        nbytes = len(compressed_bytes)
        return nbytes

    def read_chunk(self, chunk_index):
        """
        Read a chunk of data from the file
        """
        self._check_chunk(chunk_index)

        if self.compression:
            return self._read_compressed_chunk(chunk_index)
        else:
            return self._read_uncompressed_chunk(chunk_index)

    def _read_uncompressed_chunk(self, chunk_index):
        offset = self._chunk_data['offset'][chunk_index]
        nrows = self._chunk_data['nrows'][chunk_index]
        self._fobj.seek(offset, 0)
        return np.fromfile(
            self._fobj,
            dtype=self.dtype,
            count=nrows,
        )

    def _read_compressed_chunk(self, chunk_index):
        import blosc
        chunk = self._chunk_data[chunk_index]

        self._fobj.seek(chunk['offset'], 0)

        buff = bytearray(chunk['nbytes'])

        output = np.empty(chunk['nrows'], dtype=self.dtype)

        self._fobj.readinto(buff)
        blosc.decompress_ptr(buff, output.__array_interface__['data'][0])

        return output

    def _check_chunk(self, chunk_index):
        if self.nrows == 0:
            raise ValueError('file has no data')

        if chunk_index > self.nchunks - 1:
            raise ValueError(
                f'chunk {chunk_index} out of range [0, {self.nchunks-1}')

    def _update_chunks_after_write(self, nrows, nbytes):
        chunk = np.zeros(1, dtype=self.chunks_dtype)
        chunk['nbytes'] = nbytes
        chunk['nrows'] = nrows

        if self._chunk_data is None:
            self._chunk_data = chunk
        else:
            # note we probably don't want the chunks to be defined by what
            # gets appended, but rather keep the chunksize fixed
            # for compressed this would mean reading in the chunk that will
            # get appended, appending it, writing it back out with new
            # nbytes, nrows
            cd = self._chunk_data
            chunk['offset'][0] = cd['offset'][-1] + cd['nbytes'][-1]
            chunk['rowstart'][0] = cd['rowstart'][-1] + cd['nrows'][-1]

            self._chunk_data = np.hstack([self._chunk_data, chunk])

        print('appending chunk')
        self._chunks_fobj.append(chunk)

        self._set_nrows()

    def _set_compression(self, compression):
        """
        defaults get filled in
        """
        if not compression:
            self._compression = False
        else:
            self._compression = util.get_compression_with_defaults(
                compression,
                convert=True,
            )

    def _open_files(self):
        if self.verbose:
            print(
                f'opening chunks file {self.chunks_filename} '
                f'with mode: {self.mode}'
            )
        self._chunks_fobj = CColumn(
            self.chunks_filename,
            dtype=self.chunks_dtype,
            mode=self.mode,
            verbose=self.verbose,
        )
        if self._chunks_fobj.nrows > 0:
            self._chunk_data = self._chunks_fobj[:]
        else:
            self._chunk_data = None

        # needed to read/write bytes
        mode = self.mode
        if 'b' not in mode:
            mode = 'b' + mode

        self._fobj = open(self.filename, mode)

    # def read(self):
    #     """
    #     read all rows, creating the output
    #
    #     Returns
    #     -------
    #     data: array
    #         The output data
    #     """
    #     data = np.empty(self.nrows, dtype=self.dtype)
    #     super()._read_slice(data, 0)
    #     return data

    def read_into(self, data):
        """
        read rows from the start of the file, storing in the input array

        Returns
        -------
        data: array
            The output data
        """

        self._check_dtype(data)

        nrows = self.nrows
        if data.size > nrows:
            raise ValueError(
                f'input data rows {data.size} > file nrows {nrows}'
            )
        super()._read_slice(data, 0)

    # def read_slice(self, s):
    #     """
    #     read slice, creating the output
    #
    #     Parameters
    #     ----------
    #     s: slice
    #         The slice.  Must have a stop and start and no step
    #
    #     Returns
    #     -------
    #     data: array
    #         The output data
    #     """
    #     start, stop = self._extract_slice_start_stop(s)
    #     nrows = stop - start
    #     data = np.empty(nrows, dtype=self.dtype)
    #     super()._read_slice(data, start)
    #     return data

    def read_slice_into(self, data, s):
        """
        read rows into the input data

        Parameters
        ----------
        data: array
            The slice.  Must have a start
        s: slice
            The slice.  Must have a start and no step

        Returns
        -------
        None
        """

        self._check_dtype(data)

        start, stop = self._extract_slice_start_stop(s)
        nrows = stop - start
        if data.size != nrows:
            raise ValueError(f'data size {data.size} != slice nrows {nrows}')
        super()._read_slice(data, start)

    # def read_rows(self, rows):
    #     """
    #     read rows, creating the output
    #
    #     Parameters
    #     ----------
    #     rows: array
    #         The rows array
    #
    #     Returns
    #     -------
    #     data: array
    #         The output data
    #     """
    #     data = np.empty(rows.size, dtype=self.dtype)
    #     super()._read_rows(data, rows)
    #     # super()._read_rows_pages(data, rows)
    #     return data

    def read_rows_into(self, data, rows):
        """
        read rows into the input data

        Parameters
        ----------
        data: array
            Array in which to store the values
        rows: array
            The rows array

        Returns
        -------
        None
        """
        from .indices import Indices
        self._check_dtype(data)

        rows_use = util.extract_rows(rows, self.nrows)
        if not isinstance(rows_use, Indices):
            raise ValueError(f'git unexpected rows type {type(rows_use)}')

        super()._read_rows(data, rows_use)
        # super()._read_rows_pages(data, rows)

    def read_row_into(self, data, row):
        """
        read rows into the input data

        Parameters
        ----------
        data: array
            Array in which to store the values. Even though reading
            a single row, still must be length one array not scalar
        rows: array
            The rows array

        Returns
        -------
        None
        """
        self._check_dtype(data)
        if data.ndim == 0:
            raise ValueError('data must have ndim > 0')

        super()._read_row(data, row)
        # super()._read_row_pages(data, rows)

    def _read_rows(self, data, rows):
        chunk_indices = util.get_chunks(
            chunkrows_sorted=self._chunk_data['rowstart'],
            rows=rows,
        )
        # for this we assume rows are sorted
        h, _ = np.histogram(chunk_indices, np.arange(self.nchunks+1))

        w, = np.where(h > 0)
        start = 0

        for ci in w:
            num = h[ci]
            chunk_data = self.read_chunk(ci)

            end = start + num
            rowstart = self._chunk_data['rowstart'][ci]

            # this gives us rows within the chunk
            trows = rows[start:end] - rowstart

            data[start:end] = chunk_data[trows]

            start += num
        return data

    def __getitem__(self, arg):

        # returns either Indices or slice
        # converts slice with step to indices
        rows = util.extract_rows(arg, self.nrows)

        # TODO, make an optimized slice reader
        # for now, convert
        if isinstance(rows, slice):
            rows = np.arange(rows.start, rows.stop, rows.step)

        if isinstance(rows, slice):
            raise NotImplementedError('implement fast slice')
            nrows = rows.stop - rows.start
            data = np.empty(nrows, dtype=self.dtype)
            super()._read_slice(data, rows.start)
        else:
            data = np.empty(rows.size, dtype=self.dtype)

            if rows.ndim == 0:
                self._read_row(data, rows)
                data = data[0]
            else:
                self._read_rows(data, rows)

        return data

    def _extract_slice_start_stop(self, s):
        nrows = self.nrows

        start = s.start
        if start is None:
            start = 0
        stop = s.stop
        if stop is None:
            stop = nrows
        elif stop > nrows:
            stop = nrows
        return start, stop

    def _check_dtype(self, data):
        dtype = data.dtype
        if dtype != self.dtype:
            raise ValueError(
                f'input dtype {dtype} != file dtype {self.dtype}'
            )

    def _set_nrows(self):
        self._nrows = 0
        if self._chunk_data is not None:
            self._nrows = (
                self._chunk_data['rowstart'][-1]
                + self._chunk_data['nrows'][-1]
            )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __repr__(self):
        rep = _repr_template % {
            'filename': self.filename,
            'chunks_filename': self.chunks_filename,
            'mode': self.mode,
            'dtype': self.dtype,
            'nchunks': self.nchunks,
            'nrows': self.nrows,
        }
        return rep.strip()


_repr_template = """
Chunks:
    filename: %(filename)s
    chunks_filename: %(chunks_filename)s
    mode: %(mode)s
    dtype: %(dtype)s
    nchunks: %(nchunks)d
    nrows: %(nrows)d
"""


def test():
    import tempfile
    import os
    import pytest

    for compression in [False, True]:
        print('=' * 70)
        print('with compression:', compression)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.arange(60)
            fname = os.path.join(tmpdir, 'test.array')
            chunks_fname = os.path.join(tmpdir, 'test.chunks')
            with Chunks(
                filename=fname,
                chunks_filename=chunks_fname,
                dtype=data.dtype,
                mode='w+',
                compression=compression,
                verbose=True,
            ) as chunks:

                print('-' * 70)
                print('before append')
                print(chunks)

                sub1 = data[0:0 + 20]
                chunks.append(sub1)

                print('-' * 70)
                print('after append')
                print(chunks)
                assert chunks.nrows == sub1.size

                sub2 = data[20:20 + 20]
                chunks.append(sub2)

                print('-' * 70)
                print('after another append')
                print(chunks)
                assert chunks.nrows == sub1.size + sub2.size

                rsub1 = chunks.read_chunk(0)
                assert np.all(rsub1 == sub1)

                rsub2 = chunks.read_chunk(1)
                assert np.all(rsub2 == sub2)

                rsub1allslice = chunks[:]
                assert np.all(rsub1allslice == data[0:chunks.nrows])

                rslice1 = chunks[:20]
                assert np.all(rslice1 == data[:20])

                rslice2 = chunks[20:40]
                assert np.all(rslice2 == data[20:40])

                rows = np.array([3, 8, 17, 25])
                rr = chunks[rows]
                assert np.all(rr == data[rows])

                with pytest.raises(ValueError):
                    chunks.read_chunk(3)

                #
                # indata = col[2:8]
                # assert np.all(indata == data[2:8])
                #
                # indata = col[2:18:2]
                # assert np.all(indata == data[2:18:2])
                #
                # ind = [3, 5, 7]
                # indata = col[ind]
                # assert np.all(indata == data[ind])
                #
                # ind = 5
                # indata = col[ind]
                # assert np.all(indata == data[ind])
                #
                # s = slice(2, 8)
                # indata = np.zeros(s.stop - s.start, dtype=data.dtype)
                # col.read_slice_into(indata, s)
                # assert np.all(indata == data[s])
                #
                # ind = [3, 5, 7]
                # indata = np.zeros(len(ind), dtype=data.dtype)
                # col.read_rows_into(indata, ind)
                # assert np.all(indata == data[ind])
