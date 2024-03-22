import argparse
import data_convert.megvii_converter as megvii_converter
from data_convert.create_gt_database import create_groundtruth_database


def megvii_data_prep(root_path, info_prefix, dataset_name, out_dir):
    megvii_converter.create_megvii_infos(root_path, info_prefix)
    create_groundtruth_database(dataset_name,
                                root_path,
                                info_prefix,
                                f"{out_dir}/{info_prefix}_infos_train.pkl")


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="MegviiDataset", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="/",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="/",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="megvii")


args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == "megvii":

        megvii_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name="MegviiDataset",
            out_dir=args.out_dir
        )
