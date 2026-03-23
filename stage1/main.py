# import pandas as pd
# from model_loader import load_model
# from analyzer import analyze
# from config import DATA_PATH, OUTPUT_PATH

# def main():
#     model, tokenizer, device = load_model()

#     df = pd.read_csv(DATA_PATH)

#     if "Dialogue" not in df.columns:
#         raise ValueError("Missing Dialogue column")

#     df["PLAINTIFF"] = df["Dialogue"].apply(
#         lambda x: analyze(x, "plaintiff", model, tokenizer, device)
#     )

#     df["DEFENDANT"] = df["Dialogue"].apply(
#         lambda x: analyze(x, "defendant", model, tokenizer, device)
#     )

#     df.to_csv(OUTPUT_PATH, index=False)
#     print("Saved!")

# if __name__ == "__main__":
#     main()

import torch

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))