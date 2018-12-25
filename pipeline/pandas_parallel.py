"""
Requirements:
    pip install pandas msgpack dask tqdm loguru

Usage:
    import pandas as pd
    from pandas_parallel import pandas_parallel

    df = pd.DataFrame.from_records([
        ['A', 1, 2],
        ['B', 3, 4],
        ['C', 4, 5],
        ['A', 2, 3],
    ], columns=['x', 'y', 'z'])

    # 若数据每一行可单独处理，可使用`sequence`方式(默认方式)分片
    # 其他参数:
    #   npartitions: 指定分区数，默认自动选择最佳值
    #   partition_size: 指定分区大小，默认自动选择最佳值
    @pandas_parallel(progress_bar=True)
    def myfunc_seq(df):
        df['sum'] = df[['y', 'z']].sum(axis=1)
        return df

    # 若数据需要分组处理，可使用`hash`方式分片
    @pandas_parallel(partitioner='hash', partition_column='x', progress_bar=True)
    def myfunc_hash(df):
        return df.groupby('x').sum()

    # 处理超大数据集，设置iterator为True，函数会将结果写入文件，并返回pandas iterator对象
    # 参考：pd.read_csv(iterator=True)
    @pandas_parallel(iterator=True, progress_bar=True)
    def myfunc_big(df):
        df['sum'] = df[['y', 'z']].sum(axis=1)
        return df

    print(myfunc_seq(df))
    print(myfunc_hash(df))
    # 可将dataframe或iterator作为参数
    print(list(myfunc_big(df)))  # 函数返回值为iterator对象
"""
import tempfile
import atexit
import os.path
import functools
import multiprocessing
from collections import defaultdict

import tqdm
import dask.distributed as dd
import pandas as pd
from loguru import logger as LOG


TMPDIR_PREFIX = 'pandas_parallel_'
_GLOBAL_LOCAL_CLUSTER = None


def _atexit_close_cluster():
    global _GLOBAL_LOCAL_CLUSTER
    if _GLOBAL_LOCAL_CLUSTER is not None:
        _GLOBAL_LOCAL_CLUSTER.close()


def get_or_create_cluster():
    global _GLOBAL_LOCAL_CLUSTER
    if _GLOBAL_LOCAL_CLUSTER is None:
        _GLOBAL_LOCAL_CLUSTER = dd.LocalCluster()
        atexit.register(_atexit_close_cluster)
        LOG.info('local cluster running at {}', _GLOBAL_LOCAL_CLUSTER.dashboard_link)
    return _GLOBAL_LOCAL_CLUSTER


def is_dataframe(x):
    return isinstance(x, pd.DataFrame)


def getdefault_npartitions(total=None):
    npartitions = 100 * multiprocessing.cpu_count()
    if total:
        if total < 0:
            raise ValueError('total must >= 0')
        npartitions = min(total // 10000, npartitions)
    else:
        npartitions = min(1000, npartitions)
    return max(1, npartitions)


def cloudpickle_safe_lru_cache(*args, **kwargs):
    # https://github.com/cloudpipe/cloudpickle/issues/178
    _decorator = functools.lru_cache(*args, **kwargs)

    def decorator(f):
        f_wrapped = None

        def wrapped(*args, **kwargs):
            nonlocal f_wrapped
            if f_wrapped is None:
                f_wrapped = _decorator(f)
            return f_wrapped(*args, **kwargs)
        return wrapped

    return decorator


class ParallelDataFrame:
    def __init__(self, partitioner=None, partitioner_options=None):
        if partitioner is None:
            partitioner = SequencePartitioner
        if partitioner_options is None:
            partitioner_options = {}
        if isinstance(partitioner, str):
            if partitioner not in BUILTIN_PARTITIONERS:
                raise ValueError('unknown partitioner {}'.format(partitioner))
            self.partitioner_class = BUILTIN_PARTITIONERS[partitioner]
        else:
            self.partitioner_class = partitioner
        self.partitioner_options = partitioner_options
        self.partitioner = self.partitioner_class(**partitioner_options)

    def stream_split(self, data, tmpdir):
        is_iterator = not is_dataframe(data)
        if is_iterator:
            spliter = getattr(self.partitioner, 'stream_split_iterator', None)
            if spliter is None:
                spliter = self.partitioner.split_iterator
        else:
            spliter = getattr(self.partitioner, 'stream_split', None)
            if spliter is None:
                spliter = self.partitioner.split
        LOG.info('split dataframe use {}', self.partitioner)
        for i, df_part in enumerate(spliter(data)):
            part_path = os.path.join(tmpdir, str(i))
            df_part.to_pickle(part_path)
            yield part_path


class SequencePartitioner:
    def __init__(self, npartitions=None, partition_size=None):
        if npartitions is not None and npartitions <= 0:
            raise ValueError('npartitions must > 0')
        self.npartitions = npartitions
        if partition_size is not None and partition_size <= 0:
            raise ValueError('partition_size must > 0')
        self.partition_size = partition_size
        if npartitions is not None and partition_size is not None:
            raise ValueError('can not accept both npartitions and partition_size')

    def __repr__(self):
        return '<{} npartitions={}, partition_size={}>'\
            .format(self.__class__.__name__, self.npartitions, self.partition_size)

    def _compute_partition_locs(self, total, nparts, psize, remain):
        assert nparts * psize + remain == total, 'partition bug!!'
        partitions = []
        for i in range(nparts):
            if i < remain:
                begin = i * (psize + 1)
                end = begin + psize + 1
            else:
                begin = i * psize + remain
                end = begin + psize
            partitions.append((begin, end))
        return partitions

    def _compute_partition_locs_by_npartitions(self, total, npartitions):
        """split exactly {npartitions} partitions"""
        psize, remain = divmod(total, npartitions)
        return self._compute_partition_locs(total, npartitions, psize, remain)

    def _compute_partition_locs_by_partition_size(self, total, partition_size):
        """split and make every partition size < {partition_size}"""
        psize = partition_size
        npartitions, remain = divmod(total, psize)
        if remain > 0:
            psize = partition_size - 1
            npartitions, remain = divmod(total, psize)
        return self._compute_partition_locs(total, npartitions, psize, remain)

    def _get_partition_locs(self, total):
        partition_locs = None
        if self.npartitions is not None:
            partition_locs = self._compute_partition_locs_by_npartitions(total, self.npartitions)
        if self.partition_size is not None:
            partition_locs = self._compute_partition_locs_by_partition_size(
                total, self.partition_size)
        if partition_locs is None:
            npartitions = getdefault_npartitions(total)
            partition_locs = self._compute_partition_locs_by_npartitions(total, npartitions)
        return partition_locs

    def stream_split(self, df):
        partition_locs = self._get_partition_locs(len(df))
        for begin, end in partition_locs:
            yield df.iloc[begin:end]

    def split(self, df):
        return list(enumerate(self.stream_split(df)))

    def stream_split_iterator(self, iterator):
        for df in iterator:
            yield from self.split(df)

    def split_iterator(self, iterator):
        return list(self.stream_split_iterator(iterator))


class HashPartitioner:
    def __init__(self, partition_column=None, npartitions=None):
        if not partition_column:
            raise ValueError('partition_column is required by HashPartitioner')
        self.partition_column = partition_column
        if npartitions is not None and npartitions <= 0:
            raise ValueError('npartitions must > 0')
        self.npartitions = npartitions

    def __repr__(self):
        return '<{} npartitions={}, partition_column={}>'\
            .format(self.__class__.__name__, self.npartitions, self.partition_column)

    def _get_npartitions(self):
        return getdefault_npartitions() if self.npartitions is None else self.npartitions

    def _stream_split(self, df, npartitions):
        hash_keys = df[self.partition_column].apply(hash)
        for i in range(npartitions):
            df_part = df[hash_keys % npartitions == i]
            if not df_part.empty:
                yield i, df_part

    def stream_split(self, df):
        npartitions = self._get_npartitions()
        LOG.info('HashPartitioner npartitions={}', npartitions)
        for i, df_part in self._stream_split(df, npartitions):
            yield df_part

    def split(self, df):
        return list(self.stream_split(df))

    def stream_split_iterator(self, iterator):
        npartitions = self._get_npartitions()
        partition_counts = defaultdict(lambda: 0)
        with tempfile.TemporaryDirectory(prefix=TMPDIR_PREFIX) as tmpdir:
            LOG.info('HashPartitioner npartitions={}, tmpdir={}', npartitions, tmpdir)
            for i in range(npartitions):
                os.makedirs(os.path.join(tmpdir, str(i)))
            for df in iterator:
                for i, df_part in self._stream_split(df, npartitions):
                    n = partition_counts[i]
                    part_path = os.path.join(tmpdir, str(i), str(n))
                    df_part.to_pickle(part_path)
                    partition_counts[i] += 1
            for i, count in partition_counts.items():
                chunks = []
                for n in range(count):
                    part_path = os.path.join(tmpdir, str(i), str(n))
                    chunks.append(pd.read_pickle(part_path))
                yield pd.concat(chunks)

    def split_iterator(self, iterator):
        return list(self.stream_split_iterator(iterator))


BUILTIN_PARTITIONERS = {
    'sequence': SequencePartitioner,
    'hash': HashPartitioner,
}


_TASK_END = '_TASK_END_'.format(hash(object()))
_TASK_FAILED = '_TASK_FAILED'.format(hash(object()))


def _get_result_iterator(range_n_tasks, result_queue):
    result_file = tempfile.NamedTemporaryFile(prefix=TMPDIR_PREFIX, delete=False)
    for _ in range_n_tasks:
        part_result_path = result_queue.get()
        if part_result_path == _TASK_FAILED:
            raise RuntimeError('Task failed')
        df_result = pd.read_pickle(part_result_path)
        df_result.to_msgpack(result_file.name, append=True)
    return pd.read_msgpack(result_file.name, iterator=True)


def _get_result_dataframe(range_n_tasks, result_queue):
    result_dfs = []
    for _ in range_n_tasks:
        part_result_path = result_queue.get()
        if part_result_path == _TASK_FAILED:
            raise RuntimeError('Task failed')
        df_result = pd.read_pickle(part_result_path)
        result_dfs.append(df_result)
    return pd.concat(result_dfs)


def pandas_parallel(
    scheduler='processes',
    partitioner='sequence',
    progress_bar=False,
    iterator=False,
    cluster=None,
    **partitioner_options,
):
    parallel_df = ParallelDataFrame(
        partitioner=partitioner, partitioner_options=partitioner_options)

    def decorator(f):

        def worker(task_queue, result_queue, args, kwargs):
            while True:
                df_path = task_queue.get()
                if df_path == _TASK_END:
                    task_queue.put(df_path)
                    return
                try:
                    df = pd.read_pickle(df_path)
                    df = f(df, *args, **kwargs)
                    result_path = df_path + '.result'
                    df.to_pickle(result_path)
                except BaseException as ex:
                    LOG.exception(ex)
                    result_queue.put(_TASK_FAILED)
                    raise
                else:
                    result_queue.put(result_path)

        def wrapped(df, *args, **kwargs):
            if cluster is None:
                my_cluster = get_or_create_cluster()
            else:
                my_cluster = cluster
            client = dd.Client(my_cluster)
            try:
                task_queue = dd.Queue()
                result_queue = dd.Queue()
                workers = []
                for _ in range(len(my_cluster.workers)):
                    w = client.submit(worker, task_queue, result_queue, args, kwargs)
                    workers.append(w)
                with tempfile.TemporaryDirectory(prefix=TMPDIR_PREFIX) as tmpdir:
                    LOG.info('pandas parallel tmpdir={}', tmpdir)
                    n_tasks = 0
                    for df_path in parallel_df.stream_split(df, tmpdir):
                        task_queue.put(df_path)
                        n_tasks += 1
                    task_queue.put(_TASK_END)
                    range_n_tasks = range(n_tasks)
                    if progress_bar:
                        range_n_tasks = tqdm.tqdm(range_n_tasks, ncols=80, ascii=True)
                    if iterator:
                        return _get_result_iterator(range_n_tasks, result_queue)
                    else:
                        return _get_result_dataframe(range_n_tasks, result_queue)
            finally:
                client.close()
        return wrapped

    return decorator
