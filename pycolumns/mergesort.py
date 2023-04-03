"""
not used yet
"""


def cache_mergesort(source, sink, chunksize):
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
    print('cache_size:', cache_size)

    with tempfile.TemporaryDirectory() as tmpdir:

        # store sorted chunks into files of size n
        mergers = []

        for i in range(nchunks):
            start = i*chunksize
            end = (i+1)*chunksize

            chunk_data = source[start:end].copy()
            chunk_data.sort()

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

            imin = stack_tops.argmin()

            scratch[iscratch] = stack_tops[imin]
            iscratch += 1

            data = mergers[imin]

            copy_new_top = True
            if data['cache_index'] == data['cache'].size:

                if data['main_index'] == data['sf'].size:
                    # there is no more data left for this chunk
                    ind = [i for i in range(stack_tops.size) if i != imin]
                    stack_tops = stack_tops[ind]

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


def scratch_mergesort(source, sink, chunksize):
    """
        approach:
            break the source into files of size n
            sort each of these files
            merge these onto the sink
    """
    import tempfile
    import numpy as np
    from .sfile import SimpleFile

    nchunks = source.size//chunksize
    nleft = source.size % chunksize

    if nleft > 0:
        nchunks += 1

    with tempfile.TemporaryDirectory() as tmpdir:

        # store sorted chunks into files of size n
        mergers = []

        for i in range(nchunks):
            start = i*chunksize
            end = (i+1)*chunksize

            chunk_data = source[start:end].copy()
            chunk_data.sort()

            tmpf = tempfile.mktemp(dir=tmpdir, suffix='.sf')
            tmpsf = SimpleFile(tmpf, mode='w+')
            tmpsf.write(chunk_data)

            data = {
                'current_index': 0,
                'sf': tmpsf,
            }
            mergers.append(data)

        # merge onto sink
        stack_tops = np.zeros(len(mergers), dtype=source.dtype)
        scratch = np.zeros(chunksize, dtype=source.dtype)

        for i, data in enumerate(mergers):
            stack_tops[i] = data['sf'][data['current_index']]
            data['current_index'] += 1

        sink_start = 0
        iscratch = 0

        while len(mergers) > 0:

            dowrite = False

            imin = stack_tops.argmin()

            scratch[iscratch] = stack_tops[imin]
            iscratch += 1

            data = mergers[imin]
            if data['current_index'] == data['sf'].size:
                ind = [i for i in range(stack_tops.size) if i != imin]
                stack_tops = stack_tops[ind]

                del mergers[imin]
            else:
                stack_tops[imin] = data['sf'][data['current_index']]
                data['current_index'] += 1

            if iscratch == scratch.size:
                dowrite = True

            if len(mergers) == 0:
                # this is also the last chunk; we will write and quite
                # the loop after this
                dowrite = True

            if dowrite:
                nwrote = iscratch
                sink_end = sink_start + nwrote
                sink[sink_start:sink_end] = scratch[:nwrote]
                sink_start = sink_end
                iscratch = 0


def mergesort(source, sink, chunksize):
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

    with tempfile.TemporaryDirectory() as tmpdir:

        # store sorted chunks into files of size n
        mergers = []

        for i in range(nchunks):
            start = i*chunksize
            end = (i+1)*chunksize

            chunk_data = source[start:end].copy()
            chunk_data.sort()

            tmpf = tempfile.mktemp(dir=tmpdir, suffix='.sf')
            tmpsf = SimpleFile(tmpf, mode='w+')
            tmpsf.write(chunk_data)

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

            imin = stack_tops.argmin()

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


def test_old():
    import os
    import shutil
    from .sfile import SimpleFile

    infile = 'testsmall.sf'
    outfile = 'testsmall-sorted.sf'

    if os.path.exists(outfile):
        os.remove(outfile)

    shutil.copy(infile, outfile)

    sf = SimpleFile(infile)
    sfout = SimpleFile(outfile, mode='r+')

    chunk_size_mbytes = 500
    # chunk_size_mbytes = 50
    # chunk_size_mbytes = 100
    chunk_size_bytes = chunk_size_mbytes*1_000_000

    bytes_per_element = sf.dtype.itemsize
    chunksize = chunk_size_bytes//bytes_per_element

    mergesort(sf._mmap, sfout._mmap, chunksize)


def test(seed=999, num=1_000_000):
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

    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tempfile.mktemp(dir=tmpdir, prefix='infile-', suffix='.sf')
        outfile = tempfile.mktemp(dir=tmpdir, prefix='outfile-', suffix='.sf')

        if os.path.exists(infile):
            os.remove(infile)
        if os.path.exists(outfile):
            os.remove(outfile)

        with sfile.SimpleFile(infile, mode='w+') as sf:
            sf.write(data)

        shutil.copy(infile, outfile)

        with sfile.SimpleFile(infile, mode='r') as sfout:
            with sfile.SimpleFile(outfile, mode='w+') as sfout:

                chunk_size_mbytes = 500
                # chunk_size_mbytes = 50
                # chunk_size_mbytes = 100
                chunk_size_bytes = chunk_size_mbytes*1_000_000

                bytes_per_element = sf.dtype.itemsize
                chunksize = chunk_size_bytes//bytes_per_element

                mergesort(
                    source=sf._mmap,
                    sink=sfout._mmap,
                    order=valname,
                    chunksize=chunksize,
                    tmpdir=tmpdir,
                )

        sdata = sfile.read(outfile)
        data.sort(order=valname)
        assert eu.numpy_util.compare_arrays(sdata, data)
