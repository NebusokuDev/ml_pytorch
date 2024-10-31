from logging import getLogger


def choose_device(logger=None):
    """
    デバイスを選択する関数。
    - torch_directmlがインポートできる場合は、そのデバイスを返す。
    - それ以外の場合は、CUDAが利用可能であればCUDAデバイスを、そうでなければCPUデバイスを返す。

    Args:
        logger (Logger, optional): ロガーインスタンス。指定しない場合はデフォルトのロガーを使用。

    Returns:
        torch.device: 選択されたデバイス
    """
    import torch
    logger = logger or getLogger(__name__)

    try:
        import torch_directml
        return torch_directml.device()
    except ModuleNotFoundError:
        logger.warning("torch_directml not found. Falling back to CUDA or CPU.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as err:
        logger.error(f"Error while selecting device: {err}")
        return torch.device("cpu")  # エラー時はCPUデバイスを返す
