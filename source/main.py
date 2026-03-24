import argparse

def main():
    parser = argparse.ArgumentParser(description="ECB Statement Segmenter")
    parser.add_argument("--csv",        default="statements_sample.csv")
    parser.add_argument("--train",      action="store_true",
                        help="Fine-tune the transformer")
    parser.add_argument("--model_dir",  default="./ecb_segmenter_model")
    parser.add_argument("--infer",      action="store_true",
                        help="Run inference on the latest statement")
    parser.add_argument("--out_csv",    default="ecb_labeled_paragraphs.csv")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()

    

if __name__ == "__main__":
    main()