"""
not used yet
"""


def mergesort_index(source, sink, order, chunksize, tmpdir):
    """
    just as fast
    """
    import tempfile
    import numpy as np
    from .sfile import SimpleFile

    nchunks = source.size//chunksize
    nleft = source.size % chunksize

    if nleft > 0:
        nchunks += 1

    # store sorted chunks into files of size n
    mergers = []

    for i in range(nchunks):
        start = i*chunksize
        end = (i+1)*chunksize

        chunk_data = source[start:end].copy()
        chunk_data.sort(order=order)

        tmpf = tempfile.mktemp(dir=tmpdir, suffix='.sf')
        tmpsf = SimpleFile(tmpf, mode='w+')
        tmpsf.write(chunk_data)
        del chunk_data

        data = {
            'current_index': 0,
            'sf': tmpsf,
        }
        mergers.append(data)

    # merge onto sink
    stack_tops = np.zeros(len(mergers), dtype=source.dtype)

    for i, data in enumerate(mergers):
        stack_tops[i] = data['sf'][data['current_index']]
        data['current_index'] += 1

    isink = 0

    while len(mergers) > 0:

        imin = stack_tops[order].argmin()

        sink[isink] = stack_tops[imin]
        isink += 1

        data = mergers[imin]
        if data['current_index'] == data['sf'].size:
            ind = [i for i in range(stack_tops.size) if i != imin]
            stack_tops = stack_tops[ind]

            del mergers[imin]
        else:
            stack_tops[imin] = data['sf'][data['current_index']]
            data['current_index'] += 1


def cache_mergesort(source, sink, order, chunksize, tmpdir):
    """
    maybe 15% faster?
    """
    import tempfile
    import numpy as np
    from .sfile import SimpleFile

    nchunks = source.size//chunksize
    nleft = source.size % chunksize

    if nleft > 0:
        nchunks += 1

    cache_size = chunksize // nchunks

    # store sorted chunks into files of size n
    mergers = []

    for i in range(nchunks):
        start = i*chunksize
        end = (i+1)*chunksize

        chunk_data = source[start:end].copy()
        chunk_data.sort(order=order)

        tmpf = tempfile.mktemp(dir=tmpdir, suffix='.sf')
        tmpsf = SimpleFile(tmpf, mode='w+')
        tmpsf.write(chunk_data)

        cache = tmpsf._mmap[:cache_size].copy()
        data = {
            # location where we will start next cache chunk
            'main_index': cache.size,
            'cache_index': 0,
            'sf': tmpsf,
            'cache': cache,
        }
        mergers.append(data)

    # merge onto sink
    stack_tops = np.zeros(len(mergers), dtype=source.dtype)
    scratch = np.zeros(chunksize, dtype=source.dtype)

    for i, data in enumerate(mergers):
        stack_tops[i] = data['cache'][data['cache_index']]
        data['cache_index'] += 1

    sink_start = 0
    iscratch = 0

    while len(mergers) > 0:

        dowrite = False

        imin = stack_tops[order].argmin()

        scratch[iscratch] = stack_tops[imin]
        iscratch += 1

        data = mergers[imin]

        copy_new_top = True
        if data['cache_index'] == data['cache'].size:

            if data['main_index'] == data['sf'].size:
                # there is no more data left for this chunk
                ind = [i for i in range(stack_tops.size) if i != imin]
                stack_tops = stack_tops[ind]

                mergers[imin]['sf'].close()
                del mergers[imin]
                copy_new_top = False
            else:
                # we have more data, lets load some into the cache
                main_index = data['main_index']
                next_index = main_index + cache_size
                data['cache'] = (
                    data['sf']._mmap[main_index:next_index].copy()
                )
                data['cache_index'] = 0
                data['main_index'] = main_index + data['cache'].size

        if copy_new_top:
            stack_tops[imin] = data['cache'][data['cache_index']]
            data['cache_index'] += 1

        if iscratch == scratch.size or len(mergers) == 0:
            dowrite = True

        if dowrite:
            num_wrote = iscratch
            sink_end = sink_start + num_wrote
            sink[sink_start:sink_end] = scratch[:num_wrote]
            sink_start = sink_end
            iscratch = 0


def _do_test(func, tmpdir, seed=999, num=1_000_000, chunksize_mbytes=500):
    import tempfile
    import os
    import shutil
    import numpy as np
    import esutil as eu
    from . import sfile

    valname = 'val'

    rng = np.random.RandomState(seed)
    data = np.zeros(num, dtype=[('index', 'i8'), (valname, 'f8')])
    data['index'] = np.arange(num)
    data['val'] = rng.uniform(size=num)

    infile = tempfile.mktemp(dir=tmpdir, prefix='infile-', suffix='.sf')
    outfile = tempfile.mktemp(dir=tmpdir, prefix='outfile-', suffix='.sf')

    if os.path.exists(infile):
        os.remove(infile)
    if os.path.exists(outfile):
        os.remove(outfile)

    with sfile.SimpleFile(infile, mode='w+') as sf:
        sf.write(data)

    if func == 'inplace':
        outfile = infile
        with sfile.SimpleFile(outfile, mode='r+') as sf:
            sf._mmap.sort(order=valname)
    else:
        shutil.copy(infile, outfile)

        with sfile.SimpleFile(infile, mode='r') as sfin:
            with sfile.SimpleFile(outfile, mode='r+') as sfout:

                chunksize_bytes = chunksize_mbytes * 1024 * 1024

                bytes_per_element = sfin.dtype.itemsize
                chunksize = chunksize_bytes//bytes_per_element

                print(f'num: {num} chunksize: {chunksize}')

                func(
                    source=sfin._mmap,
                    sink=sfout._mmap,
                    order=valname,
                    chunksize=chunksize,
                    tmpdir=tmpdir,
                )

    sdata = sfile.read(outfile)
    data.sort(order=valname, kind='mergesort')
    assert eu.numpy_util.compare_arrays(sdata, data)


def test(seed=999, num=1_000_000, keep=False, chunksize_mbytes=500):
    import tempfile

    if keep:
        _do_test(
            func=mergesort_index,
            tmpdir='.',
            seed=seed,
            num=num,
            chunksize_mbytes=chunksize_mbytes,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            _do_test(
                func=mergesort_index,
                tmpdir=tmpdir,
                seed=seed,
                num=num,
                chunksize_mbytes=chunksize_mbytes,
            )


def test_cache(seed=999, num=1_000_000, keep=False, chunksize_mbytes=500):
    import tempfile

    if keep:
        _do_test(
            func=cache_mergesort,
            tmpdir='.',
            seed=seed,
            num=num,
            chunksize_mbytes=chunksize_mbytes,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            _do_test(
                func=cache_mergesort,
                tmpdir=tmpdir,
                seed=seed,
                num=num,
                chunksize_mbytes=chunksize_mbytes,
            )


def test_inplace(seed=999, num=1_000_000, keep=False, chunksize_mbytes=500):
    import tempfile

    if keep:
        _do_test(
            func='inplace',
            tmpdir='.',
            seed=seed,
            num=num,
            chunksize_mbytes=chunksize_mbytes,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            _do_test(
                func='inplace',
                tmpdir=tmpdir,
                seed=seed,
                num=num,
                chunksize_mbytes=chunksize_mbytes,
            )
