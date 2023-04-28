import argparse

# ----------------------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser(description="options of PCN")

    # make parser for train (part of this is used for test)
    parser.add_argument("--final_num_points", default=16384, type=int)
    parser.add_argument("--latent_dim", default=1920, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--optimizer", default="Adam", help="if you want to choose other optimization, you must change the code.")
    parser.add_argument("--lr", default=1e-3, help="learning rate", type=float)
    parser.add_argument("--lr_schduler", default=False, type=bool)
    parser.add_argument("--weight_G_loss", default=0.95, help="0 means do not use else loss (e.g. CD)", type=float)
    parser.add_argument("--dataset_dir", default="./../PCN/data/BridgeCompletion")
    parser.add_argument("--save_dir", default="./checkpoint")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--device", default="cuda")

    # make parser for test
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--select_result", default="best") # you can select best or normal
    parser.add_argument("--result_subset", default="bridge")
    parser.add_argument("--result_eval", default="test")
    parser.add_argument("--year", default="2022")
    parser.add_argument("-d", "--date", type=str)
    return parser
# ----------------------------------------------------------------------------------------
