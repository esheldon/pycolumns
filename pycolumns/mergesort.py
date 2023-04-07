
def create_mergesort_index(infile, outfile, chunksize, tmpdir, verbose=False):
    import tempfile
    import numpy as np
    from .sfile import Slicer
    from .column import get_index_dtype
    import time

    atm0 = time.time()
    with Slicer(infile) as source:
        index_dtype = get_index_dtype(source.dtype)

        nchunks = source.nrows // chunksize
        nleft = source.nrows % chunksize

        if nleft > 0:
            nchunks += 1
        # TODO easy way to get a cache size but we should make this tunable
        cache_size = chunksize // nchunks

        if verbose:
            print('index_dtype:', index_dtype)
            print('nrows:', source.nrows)
            print('chunksize:', chunksize)
            print('cache size:', cache_size)

        # store sorted chunks into files of size n
        mergers = []

        for i in range(nchunks):
            tm0 = time.time()

            if verbose:
                print(f'chunk {i+1}/{nchunks} ', end='', flush=True)

            start = i * chunksize
            end = (i + 1) * chunksize

            chunk_value_data = source[start:end]
            # might be fewer than requested in slice
            chunk_num = chunk_value_data.size

            chunk_data = np.zeros(chunk_num, dtype=index_dtype)

            s = chunk_value_data.argsort()
            # chunk_data['index'] = np.arange(start, start+chunk_num)
            # chunk_data['value'] = chunk_value_data
            chunk_data['index'] = np.arange(start, start+chunk_num)[s]
            chunk_data['value'] = chunk_value_data[s]
            del s
            del chunk_value_data

            tmpf = tempfile.mktemp(dir=tmpdir, suffix='.sf')
            slicer = Slicer(tmpf, mode='w+')
            slicer.write(chunk_data)

            # cache from lower chunk of data
            cache = chunk_data[:cache_size].copy()
            del chunk_data

            data = {
                # location where we will start next cache chunk
                'main_index': cache.size,
                'cache_index': 0,
                'slicer': slicer,
                'cache': cache,
            }
            mergers.append(data)
            if verbose:
                print('time: %.3g sec' % (time.time() - tm0))

    # merge onto sink
    stack_tops = np.zeros(len(mergers), dtype=index_dtype)
    scratch = np.zeros(chunksize, dtype=index_dtype)

    for i, data in enumerate(mergers):
        stack_tops[i] = data['cache'][data['cache_index']]
        data['cache_index'] += 1

    iscratch = 0

    if verbose:
        print('Doing mergesort')

    mtm0 = time.time()
    with Slicer(outfile, mode='w+') as sink:
        nwritten = 0
        it = 0
        while len(mergers) > 0:
            it += 1
            imin = stack_tops['value'].argmin()

            scratch[iscratch] = stack_tops[imin]
            iscratch += 1

            data = mergers[imin]

            copy_new_top = True
            if data['cache_index'] == data['cache'].size:

                if data['main_index'] == data['slicer'].size:
                    # there is no more data left for this chunk
                    ind = [i for i in range(stack_tops.size) if i != imin]
                    stack_tops = stack_tops[ind]

                    mergers[imin]['slicer'].close()
                    del mergers[imin]
                    copy_new_top = False
                else:
                    # we have more data, lets load some into the cache
                    main_index = data['main_index']
                    next_index = main_index + cache_size
                    data['cache'] = (
                        data['slicer'][main_index:next_index]
                    )
                    data['cache_index'] = 0
                    data['main_index'] = main_index + data['cache'].size

            if copy_new_top:
                stack_tops[imin] = data['cache'][data['cache_index']]
                data['cache_index'] += 1

            if iscratch == scratch.size or len(mergers) == 0:
                num2write = iscratch
                nwritten += num2write
                if verbose:
                    print(f'\nwriting {nwritten}/{source.size}')
                sink.write(scratch[:num2write])
                iscratch = 0
            else:
                if verbose and it % (chunksize // 20) == 0:
                    print('.', end='', flush=True)

    if verbose:
        print('merge time: %.3g min' % ((time.time() - mtm0)/60))
        print('total time: %.3g min' % ((time.time() - atm0)/60))


def _do_test(
    tmpdir, seed=999, num=1_000_000, chunksize_mbytes=5,
):
    import tempfile
    import numpy as np
    import esutil as eu
    from . import sfile

    valname = 'value'

    rng = np.random.RandomState(seed)
    values = rng.uniform(size=num)

    check_data = np.zeros(num, dtype=[('index', 'i8'), (valname, 'f8')])
    check_data['index'] = np.arange(num)
    check_data[valname] = values

    infile = tempfile.mktemp(dir=tmpdir, prefix='infile-', suffix='.sf')
    outfile = tempfile.mktemp(dir=tmpdir, prefix='outfile-', suffix='.sf')

    print('writing:', infile)
    sfile.write(infile, values)

    chunksize_bytes = chunksize_mbytes * 1024 * 1024

    bytes_per_element = check_data.dtype.itemsize
    chunksize = chunksize_bytes // bytes_per_element
    print('chunksize:', chunksize)

    create_mergesort_index(
        infile=infile,
        outfile=outfile,
        chunksize=chunksize,
        tmpdir=tmpdir,
    )

    sdata = sfile.read(outfile)
    check_data.sort(order=valname, kind='mergesort')
    assert eu.numpy_util.compare_arrays(
        sdata,
        check_data,
        ignore_missing=False,
        verbose=True,
    )


def test(seed=999, num=1_000_000, chunksize_mbytes=5, keep=False):
    import tempfile
    if keep:
        _do_test(
            tmpdir='.', seed=seed, num=num,
            chunksize_mbytes=chunksize_mbytes,
        )
    else:
        with tempfile.TemporaryDirectory(dir='.') as tmpdir:
            _do_test(
                tmpdir=tmpdir, seed=seed, num=num,
                chunksize_mbytes=chunksize_mbytes,
            )
