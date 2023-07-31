import time
import typing
import logging

import numpy as np
import torch
import scipy

from models.base import BaseModel

logger = logging.getLogger(__name__)


class ELSA(BaseModel):
    def __init__(self):
        self._model = None

    def name(self) -> str:
        return "elsa"

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _torch_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def fit(self, training=False):
        device = self._torch_device()
        self.set_seed(self.config.seed)
        self._model = ELSAModule(n_items=self.dataset.get_total_items(), device=device, n_dims=64)
        checkpoint_path = self._checkpoint_path(extension='pth')
        if training:
            X = self.dataset.get_train_data()
            self._model.fit(X, batch_size=1024, epochs=5)
            self._create_checkpoints_dir()
            torch.save(self._model.state_dict(), checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)
            self._model.load_state_dict(state_dict)

    def predict(self, X: scipy.sparse.csr_matrix, **kwargs):
        X_predict = self._model.predict(X, batch_size=32).cpu().numpy()
        X_predict[X.nonzero()] = 0
        self._apply_filters(X_predict, **kwargs)

        return X_predict


# class copied from https://github.com/recombee/ELSA/blob/main/elsa/elsa.py and renamed to keep name ELSA for instance of RepSys's BaseModel
class ELSAModule(torch.nn.Module):
    """
    Scalable Linear Shallow Autoencoder for Collaborative Filtering
    """

    def __init__(self, n_items: int, n_dims: int, device: torch.device = None, lr: float = 0.1):
        """
        Train model with given training data

        Parameters
        ----------
        n_items : int
            Number of items recognized by ELSA (in the paper denotes as 'I')
        n_dims : int
            Size of the items' latent space (in the paper denotes as 'r')
        device : torch.device, optional
            ELSA's weights will allocated on this device
        """
        super(ELSAModule, self).__init__()
        W = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_items, n_dims])).detach().clone())
        self.__W_list = torch.nn.ParameterList([W])
        self.__device = device or torch.device("cuda")
        self.__items_cnt = n_items
        self.__optimizer = torch.optim.NAdam(self.parameters(), lr=lr)
        # NMSELoss is implemented by us in this file, because PyTorch implements only MSELoss
        self.__nmse = NMSELoss()
        self.__cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.to(self.__device)

    def train_step(self, x, y):
        self.zero_grad()
        output = self(x)
        loss = self.__nmse(output, y)
        loss.backward()
        self.__optimizer.step()
        return loss, output

    def fit(
        self,
        train_data: typing.Union[
            torch.utils.data.dataloader.DataLoader,
            np.ndarray,
            torch.Tensor,
            scipy.sparse.csr_matrix,
        ],
        epochs: int = 20,
        batch_size: int = None,
        shuffle: bool = False,
        validation_data: typing.Union[
            torch.utils.data.dataloader.DataLoader,
            np.ndarray,
            torch.Tensor,
            scipy.sparse.csr_matrix,
        ] = None,
        verbose=True,
    ):
        """
        Train model with given training data

        Parameters
        ----------
        train_data : torch.utils.data.dataloader.DataLoader or np.ndarray or torch.Tensor or scipy.sparse.csr_matrix
            DataLoader is expected to return (batch size, number of items) tensor allocated on the same device as ELSA (specified in constructor)
            Data are expected to have users in rows and items in columns. Number of columns (items) in given data must be the same as number items recognized by ELSA (specified in constructor).
        epochs : int
            Number of epochs to train
        batch_size : int
            Required when train_data or validation are in numpy, torch or scipy format. Not required when dataloader is passed.
        shuffle : bool
            Shuffle train and validation data (if given) at the start of each epoch
        validation_data : torch.utils.data.dataloader.DataLoader or np.ndarray or torch.Tensor or scipy.sparse.csr_matrix, optional
            DataLoader is expected to return (batch size, number of items) tensor allocated on the same device as ELSA (specified in constructor)
            Data are expected to have users in rows and items in columns. Number of columns (items) in given data must be the same as number items recognized by ELSA (specified in constructor).
        verbose : bool, optional
            Print the progress of training and evaluation

        Returns
        -------
        dict
            Contains list of metrics for each epoch, namely 'nmse_train' with 'cosine_train' and 'nmse_val' with 'cosine_val' when validation_data are given
        """
        train_dataloader = self.__convert_data_to_dataloader(train_data, batch_size, shuffle, self.__device, "train_data")
        if validation_data is not None:
            validation_dataloader = self.__convert_data_to_dataloader(validation_data, batch_size, shuffle, self.__device, "validation_data")
        else:
            validation_dataloader = None

        total_steps = len(train_dataloader)
        if verbose:
            print("************************** [START] **************************")
            print(f"Runing on {self.__device}.")
            print("Total steps {:n}".format(total_steps))

        losses = {"nmse_train": [], "cosine_train": []}
        if validation_dataloader is not None:
            losses = {**losses, **{"nmse_val": [], "cosine_val": []}}
        for epoch_index in range(1, epochs + 1):
            epoch_start = time.time()
            nmse_losses_per_epoch = []
            cosine_losses_per_epoch = []
            for step, io_batch in enumerate(train_dataloader, start=1):
                # Check that io_batch.shape[1] is the same as number of items in ELSA. If not, print reasonable error
                if io_batch.shape[1] != self.__items_cnt:
                    raise ValueError(
                        f"Number of items recognized by ELSA model is {self.__items_cnt}, but given data/dataloader yields data with second dimension {io_batch.shape[1]}."
                    )
                loss, predictions = self.train_step(io_batch, io_batch)  # Input is also output, since ELSA is an autoencoder
                nmse_losses_per_epoch.append(loss.item())
                cosine_losses_per_epoch.append(1 - torch.mean(self.__cosine(io_batch, predictions), dim=-1).item())
                if verbose:
                    log_dict = {
                        "Epoch": f"{epoch_index}/{epochs}",
                        "Step": f"{step}/{total_steps}",
                        "nmse_train": round(np.mean(nmse_losses_per_epoch), 8),
                        "cosine_train": round(np.mean(cosine_losses_per_epoch), 4),
                        "time": f"{(time.time()-epoch_start):2f}s",
                    }
                    print("\r" + self.__get_progress_log(log_dict), end="")

            losses["nmse_train"].append(np.mean(nmse_losses_per_epoch))
            losses["cosine_train"].append(np.mean(cosine_losses_per_epoch))

            train_end = time.time()
            log_dict = {
                "Epoch": f"{epoch_index}/{epochs}",
                "nmse_train": round(losses["nmse_train"][-1], 8),
                "cosine_train": round(losses["cosine_train"][-1], 4),
                "training time": f"{(train_end - epoch_start):2f}s",
            }

            if validation_dataloader is not None:
                epoch_start = time.time()
                nmse_losses_per_epoch = []
                cosine_losses_per_epoch = []
                with torch.no_grad():
                    for step, io_batch in enumerate(validation_dataloader, start=1):
                        output = self(io_batch)
                        nmse_losses_per_epoch.append(
                            self.__nmse(output, io_batch).item()
                        )
                        cosine_losses_per_epoch.append(1 - torch.mean(self.__cosine(io_batch, output), dim=-1).item())

                losses["nmse_val"].append(np.mean(nmse_losses_per_epoch))
                losses["cosine_val"].append(np.mean(cosine_losses_per_epoch))

                log_dict = {
                    **log_dict,
                    **{
                        "nmse_val": losses["nmse_val"][-1],
                        "cosine_val": losses["cosine_val"][-1],
                        "val time": f"{(time.time() - train_end):2f}s",
                    },
                }

            if verbose:
                print("\r" + self.__get_progress_log(log_dict))
        if verbose:
            print("\n************************** [END] **************************")
        return losses

    def set_device(self, device: typing.Union[str, torch.device]) -> None:
        """
        Set device (like cuda or cpu).
        If given device is different than the currectly used devices, all tensors will be moved

        Parameters
        ----------
        device : str or torch.device
        """
        if isinstance(device, str):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise ValueError(f"Device must be specified by string or by torch.device instance, but '{type(device)}' was given.")
        self.__device = device
        for i, W in enumerate(self.__W_list):
            if W.device != self.__device:
                self.__W_list[i] = W.to(self.__device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Give tensor 'x' to forward pass ELSA by computing (xAA^T - x) where A is a tensor of item embeddings

        Parameters
        ----------
        x : torch.Tensor
            Input two-dimensional tensor where second dimension is required to be 'num_items'

        Returns
        -------
        torch.Tensor
            Predicted tensor with the same shape as input tensor 'x'
        """
        A = torch.nn.functional.normalize(self.__get_weights(), dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return xAAT - x

    def get_items_embeddings(self, as_numpy: bool = False) -> typing.Union[torch.Tensor, np.ndarray]:
        """
        Get embeddings for all items as 2-dimensional Tensor (or numpy array if as_numpy=True) with dimensions (n_items, n_dims)

        Parameters
        ----------
        as_numpy: bool, optional
            If set, method moves the embeddings tensor to CPU (if not already present) and converts it to numpy array

        Returns
        -------
        torch.Tensor or np.ndarray
            2-dimensional tensor (or array) with dimensions (n_items, n_dims) as given in the constructor
            Tensor is allocated on the device specified in the constructor (if 'set_device' method was not called)
        """
        if as_numpy:
            return self.__get_weights().detach().cpu().numpy()
        else:
            return self.__get_weights().detach()

    def similar_items(
        self,
        N: int,
        batch_size: int,
        sources: typing.Union[np.ndarray, torch.Tensor] = None,
        candidates: typing.Union[np.ndarray, torch.Tensor] = None,
        verbose: bool = True,
    ) -> tuple:
        """
        Calculate a list of similar items measured by a cosine similarity of item embeddings

        Parameters
        ----------
        N : int
            The number of similar items to return
        batch_size : int
            Number of source items computed in one batch
        sources : np.ndarray or torch.Tensor, optional
            One dimension array of item ids to select for which items the similar items should be computed
        candidates : np.ndarray or torch.Tensor, optional
            One dimension array of item ids to select which items that can be returned as one of the the most similar items

        Returns
        -------
        tuple
            Tuple of (itemids, scores) torch tensors allocated on cpu
                The dimensions both tensors are (num_items, N) where num_items is a number of items recognized by the model
                or (len(sources), N) when parameter 'sources' is passedF
        """
        item_embeddings = self.get_items_embeddings()
        if sources is None:
            sources = torch.arange(item_embeddings.shape[0], device=self.__device)
        elif isinstance(sources, np.ndarray):
            sources = torch.from_numpy(sources).to(self.__device)
        elif isinstance(sources, torch.Tensor):
            sources = sources.to(self.__device)
        else:
            raise ValueError(f"Supported datatypes for 'sources' parameter are 'np.ndarray' and 'torch.Tensor', but {type(sources)} was given.")
        if sources.ndim > 1:
            raise ValueError(f"Parameter 'sources' is expected to be 1-dimensional array, but {sources.ndim}-dimensional array was given.")

        sources_embeddings = item_embeddings[sources]
        if candidates is None:
            candidates = torch.arange(item_embeddings.shape[0], device=self.__device)
        elif isinstance(candidates, np.ndarray):
            candidates = torch.from_numpy(candidates).to(self.__device)
        elif isinstance(candidates, torch.Tensor):
            candidates = candidates.to(self.__device)
        else:
            raise ValueError(f"Supported datatypes for 'candidates' parameter are 'np.ndarray' and 'torch.Tensor', but {type(candidates)} was given.")
        if candidates.ndim > 1:
            raise ValueError(f"Parameter 'candidates' is expected to be 1-dimensional array, but {candidates.ndim}-dimensional array was given.")

        candidates_embeddings = item_embeddings[candidates]
        items_cnt = sources_embeddings.shape[0]
        source_batches_cnt = items_cnt // batch_size + int(items_cnt % batch_size > 0)
        if verbose:
            print(f"Number of batches with size {batch_size} to compute cosine similarity and predict TopK is {source_batches_cnt}")

        neighbour_scores_list = []
        neighbour_indexes_list = []
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        source_items_processed_cnt = 0
        for i in range(source_batches_cnt):
            start_index = i * batch_size
            batch_source_indexes = sources[start_index : start_index + batch_size]
            batch_elems = sources_embeddings[start_index : start_index + batch_size]
            batch_len = batch_elems.shape[0]

            cosine_similarites_in_batch = cos(
                torch.unsqueeze(batch_elems, 2),
                torch.unsqueeze(candidates_embeddings.T, 0),
            )
            # Remove cos(a, a) = 1 from matrix by substracting -2 (to get -1) when source and candidate items are the same
            cosine_similarites_in_batch -= 2 * (batch_source_indexes[:, None] == candidates).byte()
            (
                neighbour_scores,
                neighbour_indexes_to_candidates,
            ) = cosine_similarites_in_batch.topk(N)
            neighbour_scores_list.append(neighbour_scores.cpu())
            neighbour_indexes_list.append(candidates[neighbour_indexes_to_candidates].cpu())
            source_items_processed_cnt += batch_len
            if verbose:
                print(
                    f"\rBatch {i + 1}/{source_batches_cnt}, number of source items processed: {source_items_processed_cnt}",
                    end="",
                )
        if verbose:
            print("")

        return torch.vstack(neighbour_indexes_list), torch.vstack(neighbour_scores_list)

    def predict(
        self,
        data: typing.Union[
            torch.utils.data.dataloader.DataLoader,
            np.ndarray,
            torch.Tensor,
            scipy.sparse.csr_matrix,
        ],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Predict output for all data from dataloader

        Parameters
        ----------
        data : torch.utils.data.dataloader.DataLoader or np.ndarray or torch.Tensor or scipy.sparse.csr_matrix

        batch_size : int

        Returns
        -------
        torch.Tensor
            Predicted values for given data
        """
        shuffle = False
        dataloader = self.__convert_data_to_dataloader(data, batch_size, shuffle, self.__device, "data")
        return torch.vstack(list(self.predict_generator(dataloader, batch_size)))

    def predict_generator(
        self,
        data: typing.Union[
            torch.utils.data.dataloader.DataLoader,
            np.ndarray,
            torch.Tensor,
            scipy.sparse.csr_matrix,
        ],
        batch_size: int,
    ) -> typing.Generator[torch.Tensor, None, None]:
        """
        Create and return generator that passes data from dataloader by batches to 'forward' method.

        Parameters
        ----------
        data : torch.utils.data.dataloader.DataLoader or np.ndarray or torch.Tensor or scipy.sparse.csr_matrix

        batch_size : int

        Returns
        -------
        generator
            Return predicted tensors by batches
        """
        shuffle = False
        dataloader = self.__convert_data_to_dataloader(data, batch_size, shuffle, self.__device, "data")
        for input_batch in dataloader:
            # Check that io_batch.shape[1] is the same as number of items in ELSA. If not, print reasonable error
            if input_batch.shape[1] != self.__items_cnt:
                raise ValueError(
                    f"Number of items recognized by ELSA model is {self.__items_cnt}, but given data/dataloader yields data with second dimension {input_batch.shape[1]}."
                )
            yield self.forward(input_batch).detach()

    def __get_weights(self):
        return torch.vstack([param.to(self.__device) for param in self.__W_list])

    @staticmethod
    def __convert_data_to_dataloader(
        data: typing.Union[
            torch.utils.data.dataloader.DataLoader,
            np.ndarray,
            torch.Tensor,
            scipy.sparse.csr_matrix,
        ],
        batch_size: int,
        shuffle: bool,
        device: torch.device,
        parameter_name: str,
    ) -> torch.utils.data.dataloader.DataLoader:
        """
        Take input data and converted to torch dataloader (if it is not already) located on the same device as the model

        Parameters
        ----------
        data : torch.utils.data.dataloader.DataLoader or np.ndarray or torch.Tensor or scipy.sparse.csr_matrix

        batch_size : int
            If given data are not in the dataloader (yet), batch_size is required to create it. Otherwise can be None
        shuffle : bool
            Shuffle data
        device : torch.device
            Torch device used as a target device of dataloader
        parameter_name : str
            Name of the parameter with the data in the parent method (calling this method). Printed when the data are not in the correct format

        Returns
        -------
        torch.utils.data.dataloader.DataLoader
        """
        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            dataloader = data
            if batch_size is not None and dataloader.batch_size != batch_size:
                raise ValueError(
                    f"Given parameter '{parameter_name}' is already a dataloader with predefined batch_size={dataloader.batch_size}, but given batch_size is {batch_size}. When passing dataloader, do not use parameter 'batch_size'."
                )
        elif isinstance(data, scipy.sparse.csr_matrix):
            if batch_size is None:
                raise ValueError(
                    f"Batch size cannot be None for given '{parameter_name}' with datatype '{type(data)}'. Use 'batch_size' parameter to specify it"
                )
            dataset = SparseMatrixDataset(data, device=device)
            # DataLoader by defaults extracts individual rows of sparse matrix with shape (1, num_items) and batch them to (batch_size, 1, num_items)
            # But we want to have (batch_size, num_items) on the input. The solution is to use `collate_fn` function and stack rows to one matrix
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda x: torch.vstack(x),
            )
        elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            if batch_size is None:
                raise ValueError(
                    f"Batch size cannot be None for given '{parameter_name}' with datatype '{type(data)}'. Use 'batch_size' parameter to specify it"
                )
            dataset = DenseMatrixDataset(data, device=device)
            dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError(f"Datatype '{type(data)}' for given '{parameter_name} is not supported")
        return dataloader

    @staticmethod
    def __get_progress_log(values: dict):
        return "; ".join(f"{key}: {val}" for key, val in values.items())


class NMSELoss(torch.nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(input, dim=-1),
            torch.nn.functional.normalize(target, dim=-1),
            reduction='mean'
        )


class DenseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, dense_matrix: typing.Union[np.ndarray, np.matrix], device: torch.device):
        if isinstance(dense_matrix, np.matrix):
            self.ndarray = dense_matrix.A
        else:
            self.ndarray = dense_matrix
        self.__device = device

    def __len__(self):
        return self.ndarray.shape[0]

    def __getitem__(self, idx):
        """
        Extract a row of a dense matrix, probably allocated on the CPU, and move it to given device (preferable GPU).
        """
        return torch.Tensor(self.ndarray[idx]).to(self.__device)


class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_matrix: scipy.sparse.csr_matrix, device: torch.device):
        self.csr_matrix = sparse_matrix
        self.__device = device

    def __len__(self):
        return self.csr_matrix.shape[0]

    def __getitem__(self, idx):
        """
        Extract a row of a sparse matrix converted to sparse coo tensor allocated on the CPU.
        To same memory bandwidth, it moves data in sparse format to given device (preferably GPU) and convert it to dense there
        """
        scipy_coo = self.csr_matrix[idx].tocoo()
        torch_coo = torch.sparse_coo_tensor(
            np.vstack([scipy_coo.row, scipy_coo.col]),
            scipy_coo.data.astype(np.float32),
            scipy_coo.shape,
        )
        return torch_coo.to(self.__device).to_dense()
