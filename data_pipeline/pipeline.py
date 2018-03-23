"""
Pipeline class to create pipelines for data manipulation
"""

# Author: Tiago Alves

from data_pipe import DataPipe

class Pipeline:
    """Defines a pipeline for manipulating data.

    Allows a sequence of transformations to be applied,
    offering an API to save and abstract those transformations for similar
    datasets.

    Parameters
    ----------
    in_memory: bool, optional
        True if all commands in the pipeline should be performed in memory.
        If false, will load the data partitioned in chunks. This allows
        manipulating big datasets, but considerably increase the execution
        time and on-disk space.

    chunk_size: int, optional
        If ``in_memory`` is set as True, this parameter sets how many lines
        should be processed at each iteration of the pipeline. Bigger chunk
        sizes decreases execution time, but increases memory usage. If not
        provided, the chunk_size will be estimated.

    Attributes
    ----------
    pipeline: list
        List of commands given to the pipeline

    built: bool
        Controls if the pipeline is alreay built

    Examples
    --------
    >>> from data_pipeline import Pipeline
    >>> from data_pipeline.data_pipe import DataPipe as dp

    >>> pipeline = Pipeline()

    >>> pipeline.build([
    >>>    anonimyze("cpf"),
    >>>    drop_sparse(),
    >>>    drop_duplicates(),
    >>>    set_one_hot(),
    >>>    remove_outliers(),
    >>>    keep_numeric(),
    >>>    select("age > 18"),
    >>>    fill_null("mean"),
    >>>    update(lambda data: data.transpose()),
    >>>    normalize(),
    >>>    split_temporal("date")
    >>> ])

    >>> data = dp.load("file")
    >>> train, test = pipeline.run(data)
    """


    def __init__(self, in_memory: bool = True, chunk_size: int = -1) -> None:
        self.in_memory = in_memory
        self.chunk_size = chunk_size
        self.built = False

    def build(self, commands: list) -> None:
        """Builds the pipeline from a list of commands.
        Every command in the must receive and return a DataPipe object or
        a tuple of DataPipes. If a command returns more than a single DataPipe,
        the pipeline will split itself and all following commands will be
        run for each new pipeline.

        Parameters
        ----------
        commands: list
            List of commands to execute. Every command must receive and return 
            a DataPipe object or a tuple of DataPipes.
        """
        pass

    def run(self, datapipe: DataPipe) -> tuple:
        """Runs the pipeline for the given DataPipe. The pipeline must be
        built first.

        Parameters
        ----------
        datapipe: DataPipe
            data_pipe.DataPipe object
    
        Returns
        ----------
        data: tuple of DataPipes transformed. If no split opperation is 
        performed, it will contain only a single DataPipe, otherwise the 
        number of datapipes follows the number of splits performed.
        """
        pass

    def apply(self, datapipe: DataPipe) -> tuple:
        """Similar to the ``run`` method, but it does not create new 
        information, i.e., this method will not generate new anonymization 
        keys or one hot enconding columns.

        Parameters
        ----------
        datapipe: DataPipe
            data_pipe.DataPipe object
        
        Returns
        ----------
        data: tuple of DataPipes transformed. If no split opperation is 
        performed, it will contain only a single DataPipe, otherwise the 
        number of datapipes follows the number of splits performed.
        """
        pass

    def save(self, filename: str):
        """Dump the pipeline to a file.
    
        Parameters
        ----------
        filename: str
            Path to where to save the pipeline on disk.
        """
        pass
        
    def loads(self, filename: str):
        """Load a saved pipeline from a file.
    
        Parameters
        ----------
        filename: str
            Path to where to retrieve the pipeline from disk.
        """
        pass

    def print(self) -> None:
        """Print the pipeline commands as string.
        """
        pass

    def plot(self) -> None:
        """Plot the pipeline commands as a tree.
        """
        pass
