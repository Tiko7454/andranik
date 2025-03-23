import os
import kagglehub
from tqdm import tqdm
from ui.fen import FEN
from ml.main import get_model_result
from main import MODEL_PATH

path = kagglehub.dataset_download("koryakinp/chess-positions")
print("Path to dataset files:", path)

elements = 0
true_results = 0
bad = []

directory = "test"
total = len(os.listdir(os.path.join(path, directory)))
with tqdm(total=total, desc="Processing") as pbar:
    for img_name in os.listdir(os.path.join(path, directory)):
        elements += 1
        fen = img_name.split(".")[0].replace("-", "/")
        img_path = os.path.join(path, directory, img_name)
        chess_board = get_model_result(img_path, MODEL_PATH)
        result_fen = FEN(chess_board)
        if result_fen != fen:
            bad.append(img_path)
            # print(result_fen._main_fen)
            # print(fen)
        else:
            true_results += 1
        prc = true_results/elements
        pbar.set_postfix({
            "Accuracy": f"{prc:.4f}"
        })
        pbar.update()

# print(bad)
print(f"{true_results} / {elements} = {true_results/elements*100}%")
