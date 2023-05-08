import numpy as np
from ._column import Column as CColumn
from . import util
from .defaults import CHUNKS_DTYPE


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
        chunksize='1m',
        verbose=False,
    ):
        self._filename = filename
        self._chunks_filename = chunks_filename

        self._dtype = np.dtype(dtype)
        self._chunks_dtype = np.dtype(CHUNKS_DTYPE)

        self._set_compression(compression)
        self._mode = mode
        self._verbose = verbose
        self._open_files()
        self._set_nrows()
        self._set_row_chunksize(chunksize)

        self._clear_chunk_cache()

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
    def chunk_data(self):
        """
        get number of chunks
        """
        return self._chunk_data

    @property
    def nchunks(self):
        """
        get number of chunks
        """
        cd = self.chunk_data

        nchunks = 0
        if cd.has_data():
            nchunks = cd.size

        return nchunks

    @property
    def chunksize(self):
        """
        get string chunk size
        """
        return self._chunksize

    @property
    def row_chunksize(self):
        """
        get chunk size in rows
        """
        return self._row_chunksize

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
        self._fobj.close()

    def append(self, data):
        """
        Append data to the file. If the last chunk is not filled it
        will be filled before appending in a new chunk
        """

        data = util.get_data_with_conversion(data, self.dtype)

        cd = self.chunk_data
        if not cd.has_data():
            # nothing in file, so just append
            fill_last_chunk = False
        else:
            # see if nrows is less than the chunksize
            fill_last_chunk = (
                cd['nrows'][-1] != self.row_chunksize
            )

        if fill_last_chunk:
            # we need to append some to this chunk
            num_to_append = self.row_chunksize - cd['nrows'][-1]

            self._append_within_last_chunk(data[:num_to_append])

            if num_to_append == data.size:
                # we already appended all the data
                return

            adata = data[num_to_append:]
        else:
            adata = data

        self._append_in_chunks(adata)

    def _append_in_chunks(self, data):
        # Write data in chunks, the last one may be unfilled.
        nwrites = data.size // self.row_chunksize
        if data.size % self.row_chunksize != 0:
            nwrites += 1

        nrows_start = self.nrows
        nrows_written = 0
        for i in range(nwrites):
            start = i * self.row_chunksize
            end = (i + 1) * self.row_chunksize
            data_to_write = data[start:end]
            self._append(data_to_write)
            nrows_written += data_to_write.size

        nrec = self.nrows - nrows_start
        if nrec != nrows_written:
            raise RuntimeError(
                f'wrote {nrows_written} but nrows incremented by {nrec}'
            )

    def _append_within_last_chunk(self, data):
        """
        append data within chunk. The new data must fit within a chunk, if the
        input data overfills the chunk, an exception is raised
        """

        last_chunk_data = self.read_chunk(self.nchunks - 1)
        new_chunk_data = np.hstack([last_chunk_data, data])

        if new_chunk_data.size > self.row_chunksize:
            raise ValueError(
                f'appending last chunk got size {new_chunk_data.size} '
                f'> chunksize {self.row_chunksize}'
            )

        # overwrite old chunk data and append ne
        self._append(new_chunk_data, overwrite_last=True)

        # the cache will be out of date
        self._clear_chunk_cache()

        if False:
            import esutil as eu
            tmp = self.read_chunk(self.nchunks - 1)
            assert eu.numpy_util.compare_arrays(tmp, new_chunk_data)

    def _append(self, data, overwrite_last=False):
        """
        Append data, either appending an entirely new chunk or overwriting the
        last chunk when filling an incomplete one

        Parameters
        ----------
        data: array
            Data to be appended
        overwrite_last: bool, optional
            If set to True, start writing at the offset of the last chunk, to
            overwrite existing data.  This is needed when updating a compressed
            chunk.
        """

        cd = self.chunk_data

        if self.nrows == 0:
            offset = 0
        else:
            if overwrite_last:
                # filling in an existing chunk
                offset = cd['offset'][-1]
            else:
                # appending an entirely new chunk
                offset = cd['offset'][-1] + cd['nbytes'][-1]

        self._fobj.seek(offset)

        if self.compression:
            nbytes = self._write_compressed_data(data)
        else:
            nbytes = self._write_uncompressed_data(data)

        assert self._fobj.tell() == offset + nbytes, (
            'file tell == offset + nbytes'
        )

        self._update_chunks_after_write(
            data.size, nbytes, overwrite_last=overwrite_last,
        )

    def _write_uncompressed_data(self, data):
        data.tofile(self._fobj)
        nbytes = data.nbytes
        return nbytes

    def _write_compressed_data(self, data, is_compressed=False):
        if is_compressed:
            compressed_bytes = data
        else:
            compressed_bytes = self._get_compressed_data(data)

        self._fobj.write(compressed_bytes)
        self._fobj.flush()
        nbytes = len(compressed_bytes)
        return nbytes

    def _write_external_compressed_bytes(self, compressed_bytes, chunk_index):
        fname = self._get_external_filename(chunk_index)

        if True and self.verbose:
            print(f'     writing external: {fname}')

        with open(fname, 'wb') as fobj:
            fobj.write(compressed_bytes)

    def _read_external_compressed_bytes(self, chunk_index):
        fname = self._get_external_filename(chunk_index)
        with open(fname, 'rb') as fobj:
            compressed_bytes = fobj.read()
        return compressed_bytes

    def _get_external_filename(self, chunk_index):
        return f'{self.filename}.{chunk_index}'

    def _get_compressed_data(self, data):
        import blosc

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
        return compressed_bytes

    def _update_chunks_after_write(
        self, nrows, nbytes, overwrite_last=False,
    ):
        """
        Parameters
        ----------
        nrows: int
            Number of rows now in most recent written chunk
        nbytes: int
            Number of bytes now in most recent written chunk
        overwrite_last: bool
            If set to True, we overwrote the last chunk so
            also overwrite the chunk information
        """

        cd = self.chunk_data
        if overwrite_last:
            nrows_added = nrows - cd['nrows'][-1]
        else:
            nrows_added = nrows

        self.chunk_data.update_after_write(
            nrows=nrows, nbytes=nbytes, overwrite=overwrite_last,
        )

        self._check_file_position_after_append()

        old_nrows = self.nrows
        self._set_nrows()

        assert self.nrows == old_nrows + nrows_added, (
            'nrows == old_nrows + nrows'
        )

    def _check_file_position_after_append(self):
        ftell = self._fobj.tell()

        cd = self.chunk_data

        predicted = cd['offset'][-1] + cd['nbytes'][-1]

        if ftell != predicted:
            s = repr(self)
            raise RuntimeError(
                f'predicted file position {predicted} but got {ftell} '
                f'in column '
                f'\n{s}'
            )

    def read_chunk(self, chunk_index):
        """
        Read the indicated chunk

        Parameters
        ----------
        chunk_index: int
            The index of the chunk

        Returns
        -------
        array
        """
        # make a writeable copy
        return self._read_chunk(chunk_index).copy()

    def _read_chunk(self, chunk_index, writeable=False):
        """
        Read a chunk of data from the file

        This version does not make a copy, but returns a readonly view.
        """
        self._cache_chunk(chunk_index)

        if writeable:
            # we need to clear the cache since its going to get updated
            chunk = self._cached_chunk
            self._clear_chunk_cache()
            return chunk
        else:
            view = self._cached_chunk.view()
            view.flags['WRITEABLE'] = False
            return view

    def _cache_chunk(self, chunk_index):
        self._check_chunk(chunk_index)

        if chunk_index == self._cached_chunk_index:
            return

        if self.compression:
            chunk = self._read_compressed_chunk(chunk_index)
        else:
            chunk = self._read_uncompressed_chunk(chunk_index)

        cd = self.chunk_data

        expected = cd['nrows'][chunk_index]
        if chunk.size != expected:
            raise RuntimeError(
                f'read {chunk.size} for chunk {chunk_index} '
                f'but expected {expected}'
            )
        self._cached_chunk_index = chunk_index
        self._cached_chunk = chunk

    def _clear_chunk_cache(self):
        self._cached_chunk_index = -1
        self._cached_chunk = None

    def _read_uncompressed_chunk(self, chunk_index):
        cd = self.chunk_data

        offset = cd['offset'][chunk_index]
        nrows = cd['nrows'][chunk_index]
        self._fobj.seek(offset, 0)
        return np.fromfile(
            self._fobj,
            dtype=self.dtype,
            count=nrows,
        )

    def _read_compressed_chunk(self, chunk_index):
        import blosc

        compressed_bytes = self._read_compressed_bytes(chunk_index)

        cd = self.chunk_data
        chunk = cd[chunk_index]

        output = np.empty(chunk['nrows'], dtype=self.dtype)
        blosc.decompress_ptr(
            compressed_bytes, output.__array_interface__['data'][0]
        )
        return output

    def _read_compressed_bytes(self, chunk_index):
        cd = self.chunk_data

        chunk = cd[chunk_index]

        if chunk['is_external']:
            buff = self._read_external_compressed_bytes(chunk_index)
        else:
            fobj = self._fobj
            fobj.seek(chunk['offset'], 0)
            nbytes = chunk['nbytes']

            buff = bytearray(nbytes)

            nread = fobj.readinto(buff)
            if nread != len(buff):
                raise RuntimeError(
                    f'Expected to read {len(buff)} but read {nread}'
                )

        return buff

    def _check_chunk(self, chunk_index):
        if self.nrows == 0:
            raise ValueError('file has no data')

        if chunk_index > self.nchunks - 1:
            raise ValueError(
                f'chunk {chunk_index} out of range [0, {self.nchunks-1}')

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

    def _set_row_chunksize(self, chunksize):
        self._chunksize = chunksize
        try:
            end = chunksize[-1]
            if end == 'r':
                self._row_chunksize = int(chunksize[:-1])
                return
        except TypeError:
            pass

        chunksize_bytes = util.convert_to_bytes(chunksize)
        chunksize_bytes = int(chunksize_bytes)

        bytes_per_element = self.dtype.itemsize
        self._row_chunksize = chunksize_bytes // (bytes_per_element * 2)
        if self._row_chunksize < 1:
            raise ValueError(
                f'chunksize {chunksize} results in less than one row per chunk'
            )

    def _open_files(self):
        self._chunk_data = ChunkData(
            filename=self.chunks_filename,
            mode=self.mode,
        )

        # needed to read/write bytes
        mode = self.mode
        if 'b' not in mode:
            mode = 'b' + mode

        self._fobj = open(self.filename, mode)

    def _read_rows(self, data, rows):
        cd = self.chunk_data

        sortind = rows.sort_index
        if sortind is not None:
            # note original rows is not destroyed
            rows = rows[sortind]

        chunk_indices = util.get_chunks(
            chunkrows_sorted=cd['rowstart'],
            rows=rows,
        )
        # for this we assume rows are sorted
        h, _ = np.histogram(chunk_indices, np.arange(self.nchunks+1))

        w, = np.where(h > 0)
        start = 0

        for ci in w:
            num = h[ci]
            chunk = self._read_chunk(ci)

            end = start + num
            rowstart = cd['rowstart'][ci]

            # this gives us rows within the chunk
            trows = rows[start:end] - rowstart

            if sortind is not None:
                srows = sortind[start:end]
                data[srows] = chunk[trows]
            else:
                data[start:end] = chunk[trows]

            start += num

        return data

    def _set_rows(self, data, rows):
        cd = self.chunk_data

        if data.size == 1 and rows.size != 1:
            fill_scalar = True
        else:
            fill_scalar = False

        sortind = rows.sort_index
        if sortind is not None:
            # note original rows is not destroyed
            rows = rows[sortind]

        chunk_indices = util.get_chunks(
            chunkrows_sorted=cd['rowstart'],
            rows=rows,
        )
        # for this we assume rows are sorted
        h, _ = np.histogram(chunk_indices, np.arange(self.nchunks+1))

        w, = np.where(h > 0)
        start = 0

        for ci in w:
            num = h[ci]

            # settin writeable clears the cache
            chunk = self._read_chunk(ci, writeable=True)

            end = start + num
            rowstart = cd['rowstart'][ci]

            # this gives us rows within the chunk
            trows = rows[start:end] - rowstart

            if fill_scalar:
                chunk[trows] = data
            else:
                if sortind is not None:
                    srows = sortind[start:end]
                    # data[srows] = chunk[trows]
                    chunk[trows] = data[srows]
                else:
                    # data[start:end] = chunk[trows]
                    chunk[trows] = data[start:end]

            self._update_chunk(ci, chunk)

            start += num

    def _update_chunk(self, chunk_index, data):

        cd = self.chunk_data

        tnrows = cd['nrows'][chunk_index]
        tnbytes = cd['nbytes'][chunk_index]

        if data.size != tnrows:
            raise ValueError(f'data size {data.size} != chunk nrows {tnrows}')

        if not self.compression:
            if data.nbytes != tnbytes:
                raise ValueError(
                    f'data nbytes {data.nbytes} != chunk nbytes {tnbytes}'
                )

            self._fobj.seek(cd['offset'][chunk_index])
            nbytes = self._write_uncompressed_data(data)

        else:
            compressed_bytes = self._get_compressed_data(data)
            nbytes = len(compressed_bytes)

            is_ext = cd['is_external'][chunk_index]
            needs_extending = chunk_index != cd.size-1 and nbytes > tnbytes

            if is_ext or needs_extending:
                self._write_external_compressed_bytes(
                    compressed_bytes, chunk_index
                )
                cd.update_after_write(
                    nrows=cd['nrows'][chunk_index],
                    nbytes=None,  # not used due to is_external=True
                    overwrite=True,
                    chunk_index=chunk_index,
                    is_external=True,
                )
            else:
                # we can write directly into the chunk, just changing the
                # nbytes entry
                self._fobj.seek(cd['offset'][chunk_index])
                self._write_compressed_data(
                    compressed_bytes, is_compressed=True,
                )
                cd.update_after_write(
                    nrows=tnrows,
                    nbytes=nbytes,
                    overwrite=True,
                    chunk_index=chunk_index,
                )

    def __getitem__(self, arg):
        from .indices import Indices

        # returns either Indices or slice
        # converts slice with step to indices
        rows = util.extract_rows(arg, self.nrows)

        # TODO, make an optimized slice reader
        # for now, convert
        if isinstance(rows, slice):
            rows = Indices(np.arange(rows.start, rows.stop, rows.step))

        data = np.empty(rows.size, dtype=self.dtype)

        if rows.ndim == 0:
            send_rows = Indices([rows])
            self._read_rows(data, send_rows)
            data = data[0]
        else:
            self._read_rows(data, rows)

        return data

    def __setitem__(self, arg, data):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        from .indices import Indices

        data = util.get_data_with_conversion(data, self.dtype)

        # raise NotImplementedError(
        #     'updating compressed columns not yet supported'
        # )

        # returns either Indices or slice
        # converts slice with step to indices
        rows = util.extract_rows(arg, self.nrows)

        # TODO, make an optimized slice reader
        # for now, convert
        if isinstance(rows, slice):
            rows = Indices(np.arange(rows.start, rows.stop, rows.step))

        if rows.ndim == 0:
            send_rows = Indices([rows])
            self._set_rows(data, send_rows)
            data = data[0]
        else:
            self._set_rows(data, rows)

        return data

    def _set_nrows(self):
        cd = self.chunk_data

        self._nrows = 0

        if cd.has_data():
            self._nrows = cd['rowstart'][-1] + cd['nrows'][-1]
            assert self.nrows == cd['nrows'].sum(), (
                'nrows == sum nrows'
            )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __repr__(self):
        indent = '    '

        rep = [f'filename: {self.filename}']
        rep += [f'mode: {self.mode}']
        rep += [f'dtype: {self.dtype.str}']
        rep += [f'nrows: {self.nrows}']
        rep += [f'chunksize: {self.chunksize}']
        rep += [f'chunks_filename: {self.chunks_filename}']

        rep = [indent + r for r in rep]

        rep = ['Chunks:'] + rep
        return '\n'.join(rep)


class ChunkData(object):
    """
    Manage data about chunks in memory and in a file
    """
    def __init__(self, filename, mode='r'):
        self._filename = filename
        self._mode = mode
        self._dtype = np.dtype(CHUNKS_DTYPE)

        self._fobj = CColumn(
            self.filename,
            dtype=self.dtype,
            mode=self.mode,
        )

        self._data = None
        if self._fobj.nrows > 0:
            self._data = self._fobj[:]

    def update_after_write(
        self,
        nrows,
        nbytes,
        overwrite=False,
        chunk_index=None,
        is_external=False,
    ):
        """
        Parameters
        ----------
        nrows: int
            Number of rows now in most recent written chunk
        nbytes: int
            Number of bytes now in most recent written chunk
        overwrite: bool
            If set to True, we overwrote the chunk indicated
            by chunk_index so also overwrite the chunk information
        chunk_index: int, optional
            Defaults to last
        is_external: bool, optional
            If set to True, indicates this chunk is stored externally
            from the main file, happens when updates cause the chunk
            to grow in compressed size.
        """

        if overwrite:
            assert self._data is not None, (
                'cannot overwrite last when starting first chunk'
            )

            if chunk_index is None:
                chunk_index = self.size - 1

            self['nrows'][chunk_index] = nrows

            if is_external:
                # we don't update nbytes, since it is now in its own file
                # and we don't need to keep track of it
                self['is_external'][chunk_index] = is_external
            else:
                self['nbytes'][chunk_index] = nbytes

            self._fobj.update_row(chunk_index, self[chunk_index])
        else:
            chunk = np.zeros(1, dtype=self.dtype)

            chunk['nbytes'] = nbytes
            chunk['nrows'] = nrows

            if self._data is None:
                self._data = chunk
            else:

                chunk['offset'] = self['offset'][-1] + self['nbytes'][-1]
                chunk['rowstart'] = self['rowstart'][-1] + self['nrows'][-1]

                self._data = np.hstack([self._data, chunk])

            self._fobj.append(chunk)

    @property
    def filename(self):
        """
        get the filename
        """
        return self._filename

    @property
    def mode(self):
        """
        get the file open mode
        """
        return self._mode

    @property
    def dtype(self):
        """
        get the numpy dtype of the chunks file data
        """
        return self._dtype

    @property
    def size(self):
        """
        get the numpy dtype of the chunks file data
        """
        self.ensure_has_data()
        return self._data.size

    def has_data(self):
        """
        Returns true if data is present
        """
        return self._data is not None

    def ensure_has_data(self):
        """
        Raises exception if not data present
        """
        if not self.has_data():
            raise RuntimeError('no chunk data exists')

    def __getitem__(self, arg):
        self.ensure_has_data()
        return self._data[arg]


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
