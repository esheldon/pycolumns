
def create_mergesort_index(
    source,
    isink,
    ssink,
    chunksize,
    tmpdir,
    verbose=False,
):
    import tempfile
    import numpy as np
    from ._column import Column as CColumn
    import time

    atm0 = time.time()

    nrows = source.nrows

    nchunks = nrows // chunksize
    nleft = nrows % chunksize

    if nleft > 0:
        nchunks += 1
    # TODO easy way to get a cache size but we should make this tunable
    cache_size = chunksize // nchunks

    if verbose:
        print('dtype:', source.dtype)
        print('index_dtype:', source.index_dtype)
        print('nrows:', nrows)
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

        tm0read = time.time()
        if verbose:
            print('reading ', end='', flush=True)

        chunk_value_data = source[start:end]

        if verbose:
            _ptime(tm0read)

        # might be fewer than requested in slice
        chunk_num = chunk_value_data.size

        schunk_data = np.zeros(chunk_num, dtype=source.index_dtype)

        if verbose:
            print('argsort ', end='', flush=True)

        tm0sort = time.time()
        s = chunk_value_data.argsort()
        if verbose:
            _ptime(tm0sort)

        ichunk_data = np.arange(start, start+chunk_num)[s]
        schunk_data = chunk_value_data[s]
        del s
        del chunk_value_data

        if verbose:
            print('writing ', end='', flush=True)

        tm0write = time.time()
        itmpf = tempfile.mktemp(dir=tmpdir, suffix='.index')
        ifobj = CColumn(itmpf, dtype=source.index_dtype, mode='w+')
        ifobj.append(ichunk_data)

        stmpf = tempfile.mktemp(dir=tmpdir, suffix='.sort')
        sfobj = CColumn(stmpf, dtype=source.dtype, mode='w+')
        sfobj.append(schunk_data)

        if verbose:
            _ptime(tm0write)

        # cache from lower chunk of data
        icache = ichunk_data[:cache_size].copy()
        scache = schunk_data[:cache_size].copy()

        del ichunk_data
        del schunk_data

        data = {
            # location where we will start next cache chunk
            'main_index': icache.size,
            'cache_index': 0,
            'ifile': ifobj,
            'sfile': sfobj,
            'icache': icache,
            'scache': scache,
        }
        mergers.append(data)
        if verbose:
            _ptime(tm0, end='\n')

    # merge onto sink

    stack_itops = np.zeros(len(mergers), dtype=source.index_dtype)
    stack_stops = np.zeros(len(mergers), dtype=source.dtype)
    iscratch = np.zeros(chunksize, dtype=source.index_dtype)
    sscratch = np.zeros(chunksize, dtype=source.dtype)

    for i, data in enumerate(mergers):
        stack_itops[i] = data['icache'][data['cache_index']]
        stack_stops[i] = data['scache'][data['cache_index']]
        data['cache_index'] += 1

    scratch_ind = 0

    if verbose:
        print('Doing mergesort')

    mtm0 = time.time()

    nwritten = 0
    it = 0
    while len(mergers) > 0:
        it += 1
        imin = stack_stops.argmin()

        iscratch[scratch_ind] = stack_itops[imin]
        sscratch[scratch_ind] = stack_stops[imin]
        scratch_ind += 1

        data = mergers[imin]

        copy_new_top = True
        if data['cache_index'] == data['icache'].size:

            if data['main_index'] == data['ifile'].nrows:
                # there is no more data left for this chunk
                ind = [i for i in range(stack_itops.size) if i != imin]
                stack_itops = stack_itops[ind]
                stack_stops = stack_stops[ind]

                mergers[imin]['ifile'].close()
                mergers[imin]['sfile'].close()
                del mergers[imin]
                copy_new_top = False
            else:
                # we have more data, lets load some into the cache
                main_index = data['main_index']
                next_index = main_index + cache_size
                data['icache'] = data['ifile'][main_index:next_index]
                data['scache'] = data['sfile'][main_index:next_index]
                data['cache_index'] = 0
                data['main_index'] = main_index + data['icache'].size

        if copy_new_top:
            stack_itops[imin] = data['icache'][data['cache_index']]
            stack_stops[imin] = data['scache'][data['cache_index']]
            data['cache_index'] += 1

        if scratch_ind == iscratch.size or len(mergers) == 0:
            num2write = scratch_ind
            nwritten += num2write
            if verbose:
                print(f'\nwriting {nwritten}/{nrows}')

            isink.append(iscratch[:num2write])
            ssink.append(sscratch[:num2write])

            scratch_ind = 0
        else:
            if verbose and it % (chunksize // 20) == 0:
                print('.', end='', flush=True)

    if verbose:
        print('merge time: %.3g min' % ((time.time() - mtm0)/60))
        print('total time: %.3g min' % ((time.time() - atm0)/60))


def create_mergesort_index_old(
    source, outfile, chunksize, tmpdir, verbose=False,
):
    import tempfile
    import numpy as np
    import fitsio
    from .column import get_index_dtype
    import time

    atm0 = time.time()

    nrows = source.nrows
    dtype = source.dtype
    index_dtype = get_index_dtype(dtype)

    nchunks = nrows // chunksize
    nleft = nrows % chunksize

    if nleft > 0:
        nchunks += 1
    # TODO easy way to get a cache size but we should make this tunable
    cache_size = chunksize // nchunks

    if verbose:
        print('index_dtype:', index_dtype)
        print('nrows:', nrows)
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

        tm0read = time.time()
        if verbose:
            print('reading ', end='', flush=True)

        chunk_value_data = source[start:end]

        if verbose:
            _ptime(tm0read)

        # might be fewer than requested in slice
        chunk_num = chunk_value_data.size

        chunk_data = np.zeros(chunk_num, dtype=index_dtype)

        if verbose:
            print('argsort ', end='', flush=True)

        tm0sort = time.time()
        s = chunk_value_data.argsort()
        if verbose:
            _ptime(tm0sort)

        # chunk_data['index'] = np.arange(start, start+chunk_num)
        # chunk_data['value'] = chunk_value_data
        chunk_data['index'] = np.arange(start, start+chunk_num)[s]
        chunk_data['value'] = chunk_value_data[s]
        del s
        del chunk_value_data

        if verbose:
            print('writing ', end='', flush=True)

        tm0write = time.time()
        tmpf = tempfile.mktemp(dir=tmpdir, suffix='.sf')
        fobj = fitsio.FITS(tmpf, 'rw')
        fobj.write(chunk_data)

        if verbose:
            _ptime(tm0write)

        # cache from lower chunk of data
        cache = chunk_data[:cache_size].copy()
        del chunk_data

        data = {
            # location where we will start next cache chunk
            'main_index': cache.size,
            'cache_index': 0,
            'fits': fobj,
            'cache': cache,
        }
        mergers.append(data)
        if verbose:
            _ptime(tm0, end='\n')

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
    with fitsio.FITS(outfile, mode='rw', clobber=True) as sink:
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

                if data['main_index'] == data['fits'][1].nrows:
                    # there is no more data left for this chunk
                    ind = [i for i in range(stack_tops.size) if i != imin]
                    stack_tops = stack_tops[ind]

                    mergers[imin]['fits'].close()
                    del mergers[imin]
                    copy_new_top = False
                else:
                    # we have more data, lets load some into the cache
                    main_index = data['main_index']
                    next_index = main_index + cache_size
                    data['cache'] = (
                        data['fits'][1][main_index:next_index]
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
                    print(f'\nwriting {nwritten}/{nrows}')

                if len(sink) == 1:
                    # nothing written yet
                    sink.write(scratch[:num2write])
                else:
                    sink[1].append(scratch[:num2write])

                iscratch = 0
            else:
                if verbose and it % (chunksize // 20) == 0:
                    print('.', end='', flush=True)

    if verbose:
        print('merge time: %.3g min' % ((time.time() - mtm0)/60))
        print('total time: %.3g min' % ((time.time() - atm0)/60))


def _ptime(tm0, end=' '):
    import time
    tm = time.time() - tm0
    print(f'({tm/60:.2g} min)', end=end, flush=True)
