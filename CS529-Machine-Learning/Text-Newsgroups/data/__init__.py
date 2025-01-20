from scipy.sparse import csr_matrix as csr, save_npz, load_npz
from numpy import ndarray, loadtxt, savetxt, float as f, int as d
from pathlib import Path
from typing import Union

#this class is used to load the dataset which is in csv format
#then change them into two different formats: csv or npz
class Loader:

    #this is s constructor of the class
    def __init__(self, path: Union[Path, str] = None, verbose: bool = False):
        """
        Class Constructor

        Parameters
        path : Union[Path, str] = None
            Base path prefix for all load and save operations.
        verbose: bool = False
            Report log to std out.
        """
        self.root_path = None
        self.verbose = verbose

        if path is not None:
            self.set_root(path)
        else:
            self.set_root('.')

    #this function is used to determine the root directory
    def set_root(self, path: Union[Path, str]):
        """
        Sets the path prefix for load and save operations.

        Parameters
        path : Union[Path, str]
            Base path prefix for all load and save operations.
        
        Raises
        RuntimError
            If an existing root path does not point to a directory
        """
        if isinstance(path, str):
            self.root_path = Path(path)
        else:
            self.root_path = path

        if self.root_path.exists() and not self.root_path.is_dir():
            raise RuntimeError('Given path is not a directory.')

    #this function is used to create the directory
    def _create_directory(self):
        """
        Creates the path prefix directory.
        """
        self.root_path.mkdir(mode=0o700, parents=True, exist_ok=True)

    #this function is used to load the file
    #the loading of the file is done in both the format: csv and npz
    def load(self, path: Union[Path, str]):
        """
        Loads a matrix from file.

        Parameters
        path : Union[Path, str]
            Relative file path from root_path

        Returns
        Union[scipy.sparse._data_matrix, numpy.ndarray]
           Returns a sparse matrix for .npz files or a ndarray for .csv files.

        Raises
        ValueError
            If given file name does not have .npz or .csv suffix.
        FileNotFoundError
            If root_path + path does not exist
        """
        if self.verbose:
            print('Preparing to load data.')
        if isinstance(path, str):
            path = self.root_path.joinpath(Path(path))
        else:
            path = self.root_path.joinpath(path)
        if path.exists():
            if self.verbose:
                print('Loading...', end='', flush=True)
            if path.suffix == '.csv':
                return self._load_csv(path)
            elif path.suffix == '.npz':
                return self._load_npz(path)
            else:
                raise ValueError(f'Unknown file type {path.suffix}')
        else:
            raise FileNotFoundError(f'{path} does not exist.')

    #this function is used to save the outcome in the csv or npz format
    #if the file name is not provided it will be named as data
    def save(self, data: Union[csr, ndarray], path: Union[Path, str] = None, overwrite: bool = False):
        """
        Saves a matrix to file.

        Parameters
        data : Union[scipy.sparse.csr_matrix, numpy.ndarray]
            The data to be written to file.
        path : Union[Path, str] = None
            The name of the file. The file name will be adjusted with the proper suffix for the matrix type. If path is None it is save to root_path + data[.npz/.csv]
        overwrite : bool = False
            Overwrite existing files.

        Raises
        IsADirectoryError
            If path is a directory.
        ValueError
            If data is not a csr sparse matrix or ndarray
        FileExistsError
            If overwrite is not True and root_path + path already exists. 
        """
        if self.verbose:
            print('Preparing to save.')
        if path is None:
            if isinstance(data, csr):
                path = self.root_path.joinpath(Path('data.npz'))
            elif isinstance(data, ndarray):
                path = self.root_path.joinpath(Path('data.csv'))
        else:
            path = self.root_path.joinpath(path)

        if path.is_dir():
            raise IsADirectoryError(f'{path} is a directory.')

        if not self.root_path.exists():
            self._create_directory()

        # Fix the file suffix
        if isinstance(data, csr):
            path = path.with_suffix('.npz')
        elif isinstance(data, ndarray):
            path = path.with_suffix('.csv')
        else:
            raise ValueError('Unknown data type. {type(data)}')

        # Always overwrite default names
        if not overwrite and path.name not in ['data.csv', 'data.npz'] and path.exists():
            raise FileExistsError(f'{path} already exists.')
        if self.verbose:
            print('Saving...', end='', flush=True)
        if path.suffix == '.npz':
            self._save_npz(data, path)
        else:
            self._save_csv(data, path)

    #this function is used to save the file in the csv format
    def _save_csv(self, data: ndarray, path: Path):
        """
        Helper function for save.

        Parameters
        data : ndarray
            A ndarray of type numpy.int or numpy.float values.
        path : Path
            Path of the file to be saved.

        Raises
        ValueError
            If data.dtype is not numpy.float or numpy.int
        """
        if data.dtype == d:
            fmt = '%d'
        elif data.dtype == f:
            fmt = '%g'
        else:
            raise ValueError('Unknown format for data.')
        savetxt(path, data, fmt=fmt, delimiter=',')
        if self.verbose:
            print('Done.')

    #this function is used to load the csv file
    def _load_csv(self, path: Union[Path, str]):
        """
        Helper function for load.

        Parameters
        path : Path
            Path of the file to be loaded.

        Returns
            A numpy.ndarray from the given path.
        """
        d = loadtxt(path, delimiter=',')
        if self.verbose:
            print('Done.')
        return self.get_sparse(d)

    #this function is used to save the data in npz format which is used to reduce the sparsity
    def _save_npz(self, data: csr, path: Path):
        """
        Helper function for save.

        Parameters
        data : scipy.sparse.csr_matrix
            A sparse matrix in cst format.
        path : Path
            Path of the file to be saved.
        """
        save_npz(path, data, True)
        if self.verbose:
            print('Done.')

    #this function is used to load the data in the npz format which is used to reduce the sparsity
    def _load_npz(self, path: Union[Path, str]):
        """
        Helper function for load.

        Parameters
        path : Path
            Path of the file to be loaded.

        Returns
            A scipy.sparse._data_matrix from the given path.
        """
        d = load_npz(path)
        if self.verbose:
            print('Done.')
        return d

    #this function is used to convert the ndarry into csr to reduce the sparcity
    def get_sparse(self, data: ndarray):
        """
        Creates a scipy.sparse.csr_matrix from numpy.ndarray

        Parameters
        data : numpy.ndarray
            The data for the sparse matrix.

        Returns
            A scipy.sparse.csr_matrix from data.
        """
        return csr(data)


class Plotter:
    def __init__(self):
        pass
